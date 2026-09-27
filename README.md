# Crop Recommendation System

Flask application trained only from `data/crop_recommendation.csv`. It validates the CSV, compares Decision Tree, Gaussian Naive Bayes, SVM, and Random Forest, and deploys a reproducible Random Forest model with `random_state=42`.

## Important data note

The supplied literal CSV contains **66 rows and 22 crop labels**, not the 69 rows and 23 labels stated elsewhere in the supplied brief. This project preserves those exact rows and reports the computed totals; no rows were added, removed, or synthesized.

## Run

```powershell
python -m pip install -r requirements.txt
python app.py
```

Register a farmer account through the site. To create an administrator for local use, run the following once (use a strong password in real deployment):

```powershell
flask --app app create-admin
```

## Demo accounts

| Role | Username | Password |
| --- | --- | --- |
| Administrator | `demo_admin` | `admin123` |
| Farmer | `demo_farmer` | `farmer123` |

These accounts are for local demonstration only. Change or remove them before any production deployment.

For a production deployment, set a non-default `SECRET_KEY`, use a managed database, and create administrator accounts through a controlled setup process.

## Test

```powershell
python -m pytest
```
