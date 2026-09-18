from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

# -----------------------------------------------------------------------------
# 1) Cas nominal : prédiction correcte avec [1.0, 2.0, 3.0]
# -----------------------------------------------------------------------------
def test_predict_success():
    response = client.post(
        "/predict",
        json={"features": [1.0, 2.0, 3.0]}
    )

    assert response.status_code == 200
    assert response.json() == {"predictions": [2.0, 4.0, 6.0]}


# -----------------------------------------------------------------------------
# 2) Cas résultat incorrect : résultat attendu volontairement faux
# -----------------------------------------------------------------------------
def test_predict_incorrect_result():
    response = client.post(
        "/predict",
        json={"features": [6.0, 2.5, 4.2]}
    )

    assert response.status_code == 200

    # On vérifie que l'API ne retourne pas ce résultat volontairement faux
    assert response.json() != {"predictions": [3.0, 5.0, 9.0]}


# -----------------------------------------------------------------------------
# 3) Cas invalide : champ "features" manquant
# -----------------------------------------------------------------------------
def test_predict_features_manquant():
    response = client.post(
        "/predict",
        json={"data": [3.5, 1.2, 4.9]}
    )

    assert response.status_code == 422
    assert response.json()["detail"][0]["msg"] == "Field required"


# -----------------------------------------------------------------------------
# 4) Cas JSON incorrect : tableau envoyé directement
# -----------------------------------------------------------------------------
def test_predict_invalid_json():
    response = client.post(
        "/predict",
        json=[3.5, 1.2, 4.9]
    )

    assert response.status_code == 422


# -----------------------------------------------------------------------------
# Cas smoke : vérifier que l'API est disponible
# -----------------------------------------------------------------------------
def test_predict_smoke():
    response = client.get("/")

    assert response.status_code == 200
    assert response.json()["message"] == "API is up and running!"

