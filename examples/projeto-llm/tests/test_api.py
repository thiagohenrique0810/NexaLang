"""Tests for TurboServe API endpoints."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

fastapi = pytest.importorskip("fastapi")

from fastapi.testclient import TestClient
from turboserve.api.server import app


client = TestClient(app)


class TestHealthEndpoint:
    def test_health_returns_ok(self):
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"


class TestCompileEndpoint:
    def test_compile_valid_tl(self):
        source = 'model "gpt2" { weights full fp16 target cpu }'
        response = client.post("/compile", json={"source": source})
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert "plan" in data
        assert "ir" in data

    def test_compile_invalid_syntax(self):
        response = client.post("/compile", json={"source": "invalid { broken"})
        assert response.status_code == 400

    def test_compile_full_config(self):
        source = '''model "llama3-8b" {
  weights quantized int4
  kv_cache turboquant bits=3
  attention flash streaming=true
  decode speculative drafter="tiny-draft"
  scheduler continuous_batching max_batch=32 max_ctx=8192
  target gpu "cuda"
}'''
        response = client.post("/compile", json={"source": source})
        assert response.status_code == 200
        data = response.json()
        ir_models = data["ir"]["models"]
        assert len(ir_models) == 1
        assert ir_models[0]["name"] == "llama3-8b"
        assert ir_models[0]["weights"]["quant_type"] == "int4"


class TestGenerateEndpointWithoutModel:
    def test_generate_without_model_returns_503(self):
        response = client.post("/generate", json={
            "prompt": "Hello world",
            "max_tokens": 10,
        })
        assert response.status_code == 503

    def test_chat_without_model_returns_503(self):
        response = client.post("/chat", json={
            "messages": [{"role": "user", "content": "hi"}],
        })
        assert response.status_code == 503


class TestMetricsEndpoint:
    def test_metrics(self):
        response = client.get("/metrics")
        assert response.status_code == 200
        data = response.json()
        assert "ready" in data


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
