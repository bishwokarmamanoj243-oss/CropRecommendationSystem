from __future__ import annotations

import sqlite3
import click
import json
import time
import shutil
import os
from functools import wraps
from pathlib import Path

from flask import Flask, abort, flash, g, redirect, render_template, request, session, url_for
from werkzeug.security import check_password_hash, generate_password_hash

from ml import FEATURE_ORDER, load_and_validate, predict_top, train_and_save, ensure_model


def create_app(test_config=None):
    app = Flask(__name__)
    app.config.from_mapping(SECRET_KEY="change-this-secret", DATABASE=str(Path(app.root_path) / "crop.db"))
    if test_config:
        app.config.update(test_config)

    def db():
        if "db" not in g:
            g.db = sqlite3.connect(app.config["DATABASE"])
            g.db.row_factory = sqlite3.Row
        return g.db

    @app.teardown_appcontext
    def close_db(_error=None):
        connection = g.pop("db", None)
        if connection is not None:
            connection.close()

    def initialize_db():
        connection = db()
        connection.executescript("""
            CREATE TABLE IF NOT EXISTS users (
              id INTEGER PRIMARY KEY, username TEXT UNIQUE NOT NULL, password TEXT NOT NULL,
              role TEXT NOT NULL CHECK(role IN ('farmer','admin')), status TEXT NOT NULL DEFAULT 'active');
            CREATE TABLE IF NOT EXISTS recommendations (
              id INTEGER PRIMARY KEY, user_id INTEGER NOT NULL, query TEXT NOT NULL,
              recommendation TEXT NOT NULL, confidence REAL NOT NULL, timestamp TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
              FOREIGN KEY(user_id) REFERENCES users(id));
            CREATE TABLE IF NOT EXISTS helpdesk_tickets (
              id INTEGER PRIMARY KEY, user_id INTEGER NOT NULL, subject TEXT NOT NULL, message TEXT NOT NULL,
              status TEXT NOT NULL DEFAULT 'open', assigned_to TEXT, created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
              FOREIGN KEY(user_id) REFERENCES users(id));
            CREATE TABLE IF NOT EXISTS feedback (
              id INTEGER PRIMARY KEY, recommendation_id INTEGER NOT NULL, user_id INTEGER NOT NULL,
              rating INTEGER NOT NULL CHECK(rating IN (-1, 1)), note TEXT, created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
              UNIQUE(recommendation_id, user_id));
            CREATE TABLE IF NOT EXISTS market_prices (
              crop TEXT PRIMARY KEY, price REAL NOT NULL, trend TEXT NOT NULL, updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP);
            CREATE TABLE IF NOT EXISTS advisories (
              id INTEGER PRIMARY KEY, title TEXT NOT NULL, body TEXT NOT NULL,
              severity TEXT NOT NULL DEFAULT 'info', active INTEGER NOT NULL DEFAULT 1,
              updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP);
        """)
        existing_columns = {row[1] for row in connection.execute("PRAGMA table_info(recommendations)")}
        if "location" not in existing_columns:
            connection.execute("ALTER TABLE recommendations ADD COLUMN location TEXT")
        user_columns = {row[1] for row in connection.execute("PRAGMA table_info(users)")}
        if "status" not in user_columns:
            connection.execute("ALTER TABLE users ADD COLUMN status TEXT NOT NULL DEFAULT 'active'")
        for crop, price, trend in (("rice", 31.5, "steady"), ("maize", 24.0, "up"), ("wheat", 28.0, "steady"), ("cotton", 52.0, "down")):
            connection.execute("INSERT OR IGNORE INTO market_prices(crop,price,trend) VALUES (?,?,?)", (crop, price, trend))
        connection.execute("INSERT OR IGNORE INTO advisories(id,title,body,severity) VALUES (1,?,?,?)",
                           ("Weekly crop health check", "Inspect leaves and stems weekly. Confirm any suspected disease with your local agricultural extension office.", "info"))
        # Local demonstration accounts. Existing accounts are never overwritten.
        for username, password, role in (
            ("demo_admin", "admin123", "admin"),
            ("demo_farmer", "farmer123", "farmer"),
        ):
            if not connection.execute("SELECT 1 FROM users WHERE username=?", (username,)).fetchone():
                connection.execute("INSERT INTO users(username,password,role) VALUES (?,?,?)",
                                   (username, generate_password_hash(password), role))
        connection.commit()

    with app.app_context():
        initialize_db()
        ensure_model()

    @app.context_processor
    def language_helpers():
        from translations import MESSAGES
        language = session.get("language", "en")
        return {"language": language, "t": lambda key: MESSAGES.get(language, MESSAGES["en"]).get(key, MESSAGES["en"].get(key, key))}

    @app.route("/language/<code>")
    def set_language(code):
        if code in {"en", "ne"}:
            session["language"] = code
        return redirect(request.referrer if request.referrer and request.referrer.startswith(request.host_url) else url_for("index"))

    @app.cli.command("create-admin")
    @click.option("--username", prompt=True)
    @click.option("--password", prompt=True, hide_input=True, confirmation_prompt=True)
    def create_admin(username, password):
        """Create a local administrator account."""
        try:
            db().execute("INSERT INTO users(username,password,role) VALUES (?,?,?)",
                         (username, generate_password_hash(password), "admin"))
            db().commit()
            click.echo("Administrator created.")
        except sqlite3.IntegrityError:
            raise click.ClickException("That username is already in use.")

    def login_required(view):
        @wraps(view)
        def wrapped(*args, **kwargs):
            if "user_id" not in session:
                return redirect(url_for("login"))
            return view(*args, **kwargs)
        return wrapped

    def admin_required(view):
        @wraps(view)
        def wrapped(*args, **kwargs):
            if session.get("role") != "admin":
                abort(403)
            return view(*args, **kwargs)
        return wrapped

    def farmer_context():
        from translations import MESSAGES
        messages = MESSAGES.get(session.get("language", "en"), MESSAGES["en"])
        alert = db().execute("SELECT title, body, severity FROM advisories WHERE active=1 ORDER BY updated_at DESC LIMIT 1").fetchone()
        guidance = {
            "en": {"irrigation": "Check soil moisture every 2–3 days; water early morning when the topsoil is dry.", "fertilizer": "Split nitrogen applications; apply phosphorus and potassium according to your soil test.", "no_alert": "No active regional pest alerts."},
            "ne": {"irrigation": "हरेक २–३ दिनमा माटोको चिस्यान जाँच्नुहोस्; माथिल्लो माटो सुख्खा भए बिहानै पानी दिनुहोस्।", "fertilizer": "नाइट्रोजनलाई भागमा प्रयोग गर्नुहोस्; माटो परीक्षणअनुसार फस्फोरस र पोटासियम दिनुहोस्।", "no_alert": "हाल कुनै सक्रिय क्षेत्रीय कीरा चेतावनी छैन।"},
        }[session.get("language", "en")]
        return {"weather_note": messages["weather_note"], "irrigation": guidance["irrigation"],
                "fertilizer": guidance["fertilizer"],
                "pest_alert": (f"{alert['title']}: {alert['body']}" if alert else guidance["no_alert"])}

    @app.route("/")
    def index():
        return render_template("index.html", feature_order=FEATURE_ORDER, **farmer_context())

    @app.route("/register", methods=("GET", "POST"))
    def register():
        if request.method == "POST":
            username, password = request.form.get("username", "").strip(), request.form.get("password", "")
            if not username or not password:
                flash("Username and password are required.")
            else:
                try:
                    db().execute("INSERT INTO users(username,password,role) VALUES (?,?,?)", (username, generate_password_hash(password), "farmer"))
                    db().commit()
                    return redirect(url_for("login"))
                except sqlite3.IntegrityError:
                    flash("That username is already in use.")
        return render_template("auth.html", action="Register")

    @app.route("/login", methods=("GET", "POST"))
    def login():
        if request.method == "POST":
            user = db().execute("SELECT * FROM users WHERE username=?", (request.form.get("username", ""),)).fetchone()
            if user and user["status"] != "active":
                flash("This account has been suspended. Please contact support.")
            elif user and check_password_hash(user["password"], request.form.get("password", "")):
                session.clear(); session.update(user_id=user["id"], username=user["username"], role=user["role"])
                return redirect(url_for("index"))
            flash("Invalid username or password.")
        return render_template("auth.html", action="Login")

    @app.route("/logout")
    def logout():
        session.clear(); return redirect(url_for("index"))

    @app.route("/recommend", methods=("POST",))
    @login_required
    def recommend():
        try:
            values = {name: float(request.form[name]) for name in FEATURE_ORDER}
            if not 0 <= values["ph"] <= 14:
                raise ValueError("Soil pH must be between 0 and 14.")
            if values["rainfall"] < 0:
                raise ValueError("Rainfall cannot be negative.")
            if any(value < 0 for value in values.values()):
                raise ValueError("Values cannot be negative.")
        except KeyError:
            flash("All seven fields are required."); return redirect(url_for("index"))
        except ValueError as exc:
            flash(str(exc) if str(exc) else "All fields must be valid numbers."); return redirect(url_for("index"))
        matches = predict_top(values)
        crop, confidence = matches[0]
        location = request.form.get("location", "").strip()[:120]
        cursor = db().execute("INSERT INTO recommendations(user_id,query,recommendation,confidence,location) VALUES (?,?,?,?,?)",
                     (session["user_id"], json.dumps(values), crop, confidence, location))
        db().commit()
        price = db().execute("SELECT price, trend FROM market_prices WHERE crop=?", (crop,)).fetchone()
        estimate = {"yield": round(1.8 + (confidence * 1.7), 1), "price": price["price"] if price else None,
                    "trend": price["trend"] if price else "unavailable"}
        return render_template("result.html", values=values, crop=crop, confidence=confidence, matches=matches,
                               estimate=estimate, recommendation_id=cursor.lastrowid, **farmer_context())

    @app.route("/history")
    @login_required
    def history():
        rows = db().execute("SELECT * FROM recommendations WHERE user_id=? ORDER BY timestamp DESC", (session["user_id"],)).fetchall()
        return render_template("history.html", rows=rows)

    @app.route("/help", methods=("GET", "POST"))
    @login_required
    def helpdesk():
        if request.method == "POST":
            subject, message = request.form.get("subject", "").strip(), request.form.get("message", "").strip()
            if subject and message:
                db().execute("INSERT INTO helpdesk_tickets(user_id,subject,message) VALUES (?,?,?)", (session["user_id"], subject, message))
                db().commit(); flash("Your question has been sent to the agricultural support queue.")
                return redirect(url_for("helpdesk"))
            flash("Please provide both a subject and your question.")
        tickets = db().execute("SELECT * FROM helpdesk_tickets WHERE user_id=? ORDER BY created_at DESC", (session["user_id"],)).fetchall()
        return render_template("help.html", tickets=tickets)

    @app.route("/feedback/<int:recommendation_id>", methods=("POST",))
    @login_required
    def feedback(recommendation_id):
        rating = 1 if request.form.get("rating") == "up" else -1
        db().execute("INSERT OR REPLACE INTO feedback(recommendation_id,user_id,rating,note) VALUES (?,?,?,?)",
                     (recommendation_id, session["user_id"], rating, request.form.get("note", "")[:500]))
        db().commit(); flash("Thank you—your feedback helps us review recommendation quality.")
        return redirect(url_for("history"))

    @app.route("/admin")
    @login_required
    @admin_required
    def admin():
        report, data = ensure_model(), load_and_validate()
        connection = db()
        stats = {"farmers": connection.execute("SELECT COUNT(*) FROM users WHERE role='farmer'").fetchone()[0],
                 "queries": connection.execute("SELECT COUNT(*) FROM recommendations WHERE date(timestamp)=date('now')").fetchone()[0],
                 "coverage": connection.execute("SELECT COUNT(DISTINCT location) FROM recommendations WHERE location IS NOT NULL AND location != ''").fetchone()[0]}
        crops = connection.execute("SELECT recommendation, COUNT(*) AS count FROM recommendations GROUP BY recommendation ORDER BY count DESC").fetchall()
        tickets = connection.execute("SELECT h.*, u.username FROM helpdesk_tickets h JOIN users u ON u.id=h.user_id ORDER BY h.created_at DESC LIMIT 10").fetchall()
        farmers = connection.execute("SELECT id,username,role,status FROM users ORDER BY id DESC LIMIT 30").fetchall()
        prices = connection.execute("SELECT * FROM market_prices ORDER BY crop").fetchall()
        feedback = connection.execute("SELECT COUNT(*) AS responses, COALESCE(ROUND(AVG(rating) * 100), 0) AS score FROM feedback").fetchone()
        recent_locations = connection.execute("SELECT location, COUNT(*) count FROM recommendations WHERE location IS NOT NULL AND location != '' GROUP BY location ORDER BY count DESC LIMIT 5").fetchall()
        advisories = connection.execute("SELECT * FROM advisories ORDER BY updated_at DESC").fetchall()
        return render_template("admin.html", report=report, rows=data.to_dict("records"), stats=stats, crops=crops, tickets=tickets, farmers=farmers, prices=prices, feedback=feedback, recent_locations=recent_locations, advisories=advisories)

    @app.route("/admin/user/<int:user_id>/status", methods=("POST",))
    @login_required
    @admin_required
    def user_status(user_id):
        status = request.form.get("status")
        if status in {"active", "suspended"} and user_id != session["user_id"]:
            db().execute("UPDATE users SET status=? WHERE id=?", (status, user_id)); db().commit()
        return redirect(url_for("admin"))

    @app.route("/admin/price", methods=("POST",))
    @login_required
    @admin_required
    def update_price():
        crop = request.form.get("crop", "").strip().lower()[:50]
        trend = request.form.get("trend", "steady")
        try: price = float(request.form.get("price", ""))
        except ValueError: price = -1
        if crop and price >= 0 and trend in {"up", "steady", "down"}:
            db().execute("INSERT INTO market_prices(crop,price,trend,updated_at) VALUES (?,?,?,CURRENT_TIMESTAMP) ON CONFLICT(crop) DO UPDATE SET price=excluded.price,trend=excluded.trend,updated_at=CURRENT_TIMESTAMP", (crop, price, trend)); db().commit()
            flash("Market price updated.")
        else: flash("Enter a crop, a non-negative price, and a valid trend.")
        return redirect(url_for("admin"))

    @app.route("/admin/advisory", methods=("POST",))
    @login_required
    @admin_required
    def create_advisory():
        title, body = request.form.get("title", "").strip(), request.form.get("body", "").strip()
        severity = request.form.get("severity", "info")
        if title and body and severity in {"info", "watch", "urgent"}:
            db().execute("INSERT INTO advisories(title,body,severity) VALUES (?,?,?)", (title[:100], body[:500], severity)); db().commit(); flash("Regional advisory published.")
        else: flash("Provide an advisory title, message, and severity.")
        return redirect(url_for("admin"))

    @app.route("/admin/ticket/<int:ticket_id>", methods=("POST",))
    @login_required
    @admin_required
    def assign_ticket(ticket_id):
        db().execute("UPDATE helpdesk_tickets SET status='assigned', assigned_to=? WHERE id=?", (request.form.get("expert", "Agricultural specialist")[:100], ticket_id))
        db().commit(); return redirect(url_for("admin"))

    @app.route("/operations")
    @login_required
    @admin_required
    def operations():
        report = ensure_model()
        started = time.monotonic()
        predict_top({"N": 90, "P": 42, "K": 43, "temperature": 20.8, "humidity": 82, "ph": 6.5, "rainfall": 202.9})
        latency = round((time.monotonic() - started) * 1000, 1)
        model = __import__("joblib").load(Path(app.root_path) / "models" / "rf_crop_model.pkl")
        importance = sorted(zip(FEATURE_ORDER, model.feature_importances_), key=lambda item: item[1], reverse=True)
        disk = shutil.disk_usage(app.root_path)
        system = {"cpu": "Not available in local Flask mode", "ram": "Not available in local Flask mode", "disk": round((disk.used / disk.total) * 100, 1), "db_size": round(os.path.getsize(app.config["DATABASE"]) / 1024, 1)}
        return render_template("operations.html", report=report, latency=latency, importance=importance, system=system)

    @app.route("/admin/train", methods=("POST",))
    @login_required
    @admin_required
    def retrain():
        report = train_and_save()
        flash("Training completed using the current authoritative CSV.")
        return render_template("training.html", report=report)

    return app


if __name__ == "__main__":
    create_app().run(debug=True)
