"""Smoke-Test: eine Chat-Anfrage gegen einen OpenAI-kompatiblen Endpunkt.

Beispiel (Ollama):
  python benchmark/smoke_test.py --base-url http://localhost:11434/v1 --model llama3.2:3b
"""

import argparse
import sys

import httpx


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://localhost:11434/v1")
    ap.add_argument("--model", default="llama3.2:3b")
    args = ap.parse_args()

    payload = {
        "model": args.model,
        "messages": [
            {"role": "user", "content": "Erkläre in einem Satz, was Knowledge Distillation ist."}
        ],
        "max_tokens": 128,
        "temperature": 0.7,
    }
    try:
        resp = httpx.post(f"{args.base_url}/chat/completions", json=payload, timeout=120)
        resp.raise_for_status()
    except httpx.HTTPError as e:
        print(f"Anfrage fehlgeschlagen: {e}", file=sys.stderr)
        print("Läuft der Server? (ollama serve / docker compose up)", file=sys.stderr)
        return 1

    data = resp.json()
    print("Antwort:\n")
    print(data["choices"][0]["message"]["content"].strip())
    usage = data.get("usage", {})
    print(
        f"\nTokens: prompt={usage.get('prompt_tokens')} "
        f"completion={usage.get('completion_tokens')}"
    )
    if not usage:
        print(
            "WARNUNG: kein 'usage'-Feld -> Token-Monitoring braucht dann einen "
            "lokalen Tokenizer-Fallback.",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
