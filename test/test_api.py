import sys 
sys.path.append("src")
from fastapi.testclient import TestClient
from src.fast_Api import app 

client = TestClient(app)

def test_root_endpoint():
    response = client.get("/")
    assert response.status_code == 200

def test_valid_question():
    response = client.post("/ask", json={"question": "What is RAG?", "n_results": 2})
    assert response.status_code == 200
    assert "answer" in response.json() or "message" in response.json()

def test_empty_question_returns_400():
    response = client.post("/ask", json={"question": "", "n_results": 2})
    assert response.status_code == 400

def test_wrong_type_returns_422():
    response = client.post("/ask", json={"question": "test", "n_results": "three"})
    assert response.status_code == 422

def test_missing_field_returns_422():
    response = client.post("/ask", json={"n_results": 3})
    assert response.status_code == 422

def test_negative_n_results_returns_400():
    response = client.post("/ask", json={"question": "test", "n_results": -1})
    assert response.status_code == 400