# src/test_llm.py
import requests, json

BASE = "http://localhost:1234"

def check_models():
    r = requests.get(f"{BASE}/v1/models", timeout=10)
    print("models status:", r.status_code)
    print(r.json())

def test_chat():
    payload = {
        "model": "mistral-7b-instruct-v0.2",
        "messages": [{"role":"user", "content":"Summarize this in one line: LM Studio is running."}],
        "temperature": 0.0
    }
    r = requests.post(f"{BASE}/v1/chat/completions", json=payload, timeout=60)
    print("chat status:", r.status_code)
    print(json.dumps(r.json(), indent=2)[:2000])

def test_embed():
    payload = {"model": "mistral-7b-instruct-v0.2", "input": "hello world"}
    r = requests.post(f"{BASE}/v1/embeddings", json=payload, timeout=30)
    print("embed status:", r.status_code)
    print("embedding length:", len(r.json().get("data", [])[0].get("embedding", [])))

if __name__ == "__main__":
    check_models()
    test_chat()
    test_embed()
