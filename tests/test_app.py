import sqlite3
from pathlib import Path

import pytest
from werkzeug.security import generate_password_hash

from app import create_app
from ml import FEATURE_ORDER, load_and_validate, predict


@pytest.fixture()
def app():
    database = Path(__file__).parent / "_runtime_test.db"
    if database.exists():
        database.unlink()
    return create_app({"TESTING": True, "SECRET_KEY": "test", "DATABASE": str(database)})


@pytest.fixture()
def client(app):
    return app.test_client()


def add_user(app, username, role="farmer"):
    with app.app_context():
        connection = sqlite3.connect(app.config["DATABASE"])
        connection.execute("INSERT INTO users(username,password,role) VALUES (?,?,?)",
                           (username, generate_password_hash("secret"), role))
        connection.commit(); connection.close()


def login(client, username):
    return client.post("/login", data={"username": username, "password": "secret"}, follow_redirects=True)


def payload(**changes):
    values = {"N": "90", "P": "42", "K": "43", "temperature": "20.8", "humidity": "82", "ph": "6.5", "rainfall": "202.9"}
    values.update(changes)
    return values


def test_authoritative_csv_shape():
    data = load_and_validate()
    assert list(data.columns[:-1]) == FEATURE_ORDER
    # These assertions intentionally reflect the exact supplied rows; do not add synthetic rows.
    assert len(data) == 66
    assert data.label.nunique() == 22


def test_prediction_uses_trained_artifacts():
    crop, confidence = predict({key: float(value) for key, value in payload().items()})
    assert isinstance(crop, str)
    assert 0 <= confidence <= 1


def test_invalid_fields_rejected(client, app):
    add_user(app, "farmer"); login(client, "farmer")
    assert b"Soil pH must be between" in client.post("/recommend", data=payload(ph="15"), follow_redirects=True).data
    assert b"Rainfall cannot be negative" in client.post("/recommend", data=payload(rainfall="-1"), follow_redirects=True).data
    incomplete = payload(); incomplete.pop("N")
    assert b"All seven fields are required" in client.post("/recommend", data=incomplete, follow_redirects=True).data


def test_farmer_prediction_is_saved_and_private(client, app):
    add_user(app, "one"); add_user(app, "two")
    login(client, "one")
    response = client.post("/recommend", data=payload(), follow_redirects=True)
    assert b"Recommended Crop" in response.data
    assert b"rice" in client.get("/history").data.lower()
    client.get("/logout"); login(client, "two")
    assert b"rice" not in client.get("/history").data.lower()


def test_non_admin_cannot_open_admin_pages(client, app):
    add_user(app, "farmer"); login(client, "farmer")
    assert client.get("/admin").status_code == 403
    assert client.post("/admin/train").status_code == 403
