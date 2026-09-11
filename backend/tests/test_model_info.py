def test_get_model_info(client):
    response = client.get("/api/v1/model/info")
    assert response.status_code == 200
    data = response.json()
    assert data["architecture"] == "ResidualEmotionCNN"
    assert data["num_classes"] == 7
    assert "happy" in data["classes"]
    assert "neutral" in data["classes"]
    assert data["input_shape"] == [1, 1, 48, 48]
    assert data["status"] == "production"
    assert "model_name" in data
    # Verify no local Windows paths leak in the response
    for k, v in data.items():
        if isinstance(v, str):
            assert "C:\\" not in v and "c:/" not in v.lower()
