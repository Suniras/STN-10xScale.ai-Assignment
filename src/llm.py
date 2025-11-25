import requests

BASE = "http://localhost:1234"

def generate_answer(prompt: str):
    payload = {
        "model": "mistral-7b-instruct-v0.2",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0
    }
    r = requests.post(f"{BASE}/v1/chat/completions", json=payload, timeout=120)
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]
