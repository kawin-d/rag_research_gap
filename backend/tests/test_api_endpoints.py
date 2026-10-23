import sys
from pathlib import Path
from fastapi.testclient import TestClient

# Add backend directory to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.main import app

client = TestClient(app)


def test_api():
    print("\n--- Running FastAPI End-to-End API Integration Tests ---")

    # 1. Health check
    res = client.get("/")
    assert res.status_code == 200
    print("[PASS] GET / -> status:", res.json()["status"])

    # 2. List papers (seed if empty)
    res = client.get("/api/papers")
    assert res.status_code == 200
    papers_data = res.json()
    if papers_data["total"] == 0:
        client.post("/api/sample/load")
        res = client.get("/api/papers")
        papers_data = res.json()
    print(f"[PASS] GET /api/papers -> Total papers: {papers_data['total']}")

    # 3. RAG Q&A
    q_payload = {
        "question": "What is Low-Rank Adaptation (LoRA) and what are its limitations?",
        "top_k": 3
    }
    res = client.post("/api/rag/ask", json=q_payload)
    assert res.status_code == 200
    rag_data = res.json()
    print(f"[PASS] POST /api/rag/ask -> Answer received with {len(rag_data['citations'])} citations.")

    # 4. Gap Clusters
    res = client.get("/api/gaps/clusters")
    assert res.status_code == 200
    clusters = res.json()
    print(f"[PASS] GET /api/gaps/clusters -> {len(clusters)} clusters returned. Top cluster score: {clusters[0]['gap_score']}")

    # 5. Gap Markdown Report
    top_id = clusters[0]["cluster_id"]
    res = client.get(f"/api/gaps/report/{top_id}?format=markdown")
    assert res.status_code == 200
    assert "Research Gap Report" in res.text
    print(f"[PASS] GET /api/gaps/report/{top_id}?format=markdown -> Valid markdown report generated.")

    # 6. Evaluation Metrics & General Benchmark Questions
    res = client.get("/api/evaluation/metrics")
    assert res.status_code == 200
    metrics = res.json()
    print(f"[PASS] GET /api/evaluation/metrics -> Question Coverage@5: {metrics['question_coverage_at_5']}, Precision: {metrics['expert_gap_precision']}")

    res_bench = client.get("/api/evaluation/benchmark-questions")
    assert res_bench.status_code == 200
    bench_qs = res_bench.json()
    assert len(bench_qs) == 10
    print(f"[PASS] GET /api/evaluation/benchmark-questions -> {len(bench_qs)} research gap benchmark questions loaded.")

    # 7. Baseline Comparison (General Research Inquiry)
    res = client.post("/api/evaluation/baseline-comparison?question=What+are+the+primary+limitations+and+methodologies+discussed?")
    assert res.status_code == 200
    print("[PASS] POST /api/evaluation/baseline-comparison -> Comparison succeeded.")

    # 8. Interactive LLM Provider Selector & Security Tests
    status_res = client.get("/api/rag/llm-status")
    assert status_res.status_code == 200
    st = status_res.json()
    assert "provider" in st
    assert "model" in st
    assert "supported_models" in st
    assert "has_key" in st
    # Verify strict security: raw keys must NEVER appear in JSON response
    for k in ["api_key", "key", "secret", "GROQ_API_KEY", "OPENAI_API_KEY"]:
        assert k not in st, f"Security Violation: '{k}' exposed in status response!"
    print(f"[PASS] GET /api/rag/llm-status -> Secure response (Active: {st['provider']}, Model: {st['model']})")

    # Switch to Groq
    cfg_res = client.post("/api/rag/llm-config", json={"provider": "groq", "model": "openai/gpt-oss-120b"})
    assert cfg_res.status_code == 200
    assert cfg_res.json()["provider"] == "groq"
    assert cfg_res.json()["model"] == "openai/gpt-oss-120b"
    print("[PASS] POST /api/rag/llm-config -> Switched to Groq (openai/gpt-oss-120b)")

    # Switch to OpenAI
    cfg_res2 = client.post("/api/rag/llm-config", json={"provider": "openai", "model": "gpt-4o-mini"})
    assert cfg_res2.status_code == 200
    assert cfg_res2.json()["provider"] == "openai"
    assert cfg_res2.json()["model"] == "gpt-4o-mini"
    print("[PASS] POST /api/rag/llm-config -> Switched to OpenAI (gpt-4o-mini)")

    # Switch to Mock
    cfg_res3 = client.post("/api/rag/llm-config", json={"provider": "mock"})
    assert cfg_res3.status_code == 200
    assert cfg_res3.json()["provider"] == "mock"
    print("[PASS] POST /api/rag/llm-config -> Switched to Mock")

    # Restore Groq as active
    client.post("/api/rag/llm-config", json={"provider": "groq", "model": "openai/gpt-oss-120b"})

    print("\n=========================================")
    print(" ALL API ENDPOINTS VERIFIED AND PASSING! ")
    print("=========================================\n")


if __name__ == "__main__":
    test_api()
