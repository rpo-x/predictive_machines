#!/usr/bin/env python3
# There are prerequisites (in terminal):
# 1. brew install ollama
# 2. ollama serve
# 3. ollama pull llama3
import requests
import json

API_URL = "http://localhost:11434/api/generate"
MODEL = "llama3"


def chat():
    while True:
        prompt = input("You: ").strip()
        if prompt.lower() in {"exit", "quit"}:
            break

        with requests.post(
            API_URL, json={"model": MODEL, "prompt": prompt}, stream=True
        ) as r:
            for line in r.iter_lines():
                if line:
                    data = json.loads(line)
                    if "response" in data:
                        print(data["response"], end="", flush=True)
        print()


if __name__ == "__main__":
    chat()
