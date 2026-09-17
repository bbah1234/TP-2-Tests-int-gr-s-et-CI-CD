from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

# -----------------------------------------------------------------------------
# Cas nominaux : entrées valides et représentatives
# -----------------------------------------------------------------------------
def test_predict_success():
    response = client.post("/predict", json={
    "features": [1.0, 2.0, 3.0]
    })
    assert response.status_code == 200
    assert response.json() == {"predictions": [2.0, 4.0, 6.0]}

# -----------------------------------------------------------------------------
# Cas resultats faux : resultas attendus volontairement faux
# -----------------------------------------------------------------------------
def test_predict_success():
    response = client.post("/predict", json={
    "features": [6.0, 2.5, 4.2]
    })
    assert response.status_code == 200
    #assert response.json() == {"predictions": [3.0, 5.0, 9.0]}

# -----------------------------------------------------------------------------
# Cas invalides : données ne respectant pas les préconditions attendues
# -----------------------------------------------------------------------------
def test_predict_unprocessable_entity():
    response = client.post("/predict", json={
    "feature1": 1.0,
    "feature2": 2.0,
    "feature3": 3.0
    })
    assert response.status_code == 422
    assert response.json()["detail"][0]["msg"] == "Field required"

# -----------------------------------------------------------------------------
# Cas smoke : valider que l'API est disponible
# -----------------------------------------------------------------------------
def test_predict_smoke():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["message"] == "API is up and running!"

# ------------------------------------------------------------------------------
# Cas JSON incorrect
# -------------------------------------------------------------------------------
def test_predict_features_manquant():
    response = client.post(
        "/predict",
        json={"data": [3.5, 1.2, 4.9]}
    )

    assert response.status_code == 422
    assert response.json()["detail"][0]["msg"] == "Field required"

def test_predict_features_manquant():
    response = client.post(
        "/predict",
        json=[3.5, 1.2, 4.9]
    )

    assert response.status_code == 422