"""
Ollama (Llama 3) client — talks to local Ollama REST API.
"""
import json
import requests
from typing import Dict, Any, Optional

DEFAULT_BASE_URL = "http://localhost:11434"
DEFAULT_MODEL = "llama3:8b"


class LlamaClient:
    def __init__(
        self,
        base_url: str = DEFAULT_BASE_URL,
        model: str = DEFAULT_MODEL,
        temperature: float = 0.3,
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.temperature = temperature

    def is_available(self) -> bool:
        try:
            r = requests.get(f"{self.base_url}/api/tags", timeout=5)
            return r.status_code == 200
        except Exception:
            return False

    def list_models(self) -> list[str]:
        try:
            r = requests.get(f"{self.base_url}/api/tags", timeout=5)
            data = r.json()
            return [m["name"] for m in data.get("models", [])]
        except Exception:
            return []

    def chat(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: Optional[float] = None,
    ) -> str:
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "stream": False,
            "options": {
                "temperature": temperature or self.temperature,
            },
            "format": "json",
        }
        r = requests.post(
            f"{self.base_url}/api/chat",
            json=payload,
            timeout=300,
        )
        r.raise_for_status()
        return r.json()["message"]["content"]

    def chat_json(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: Optional[float] = None,
    ) -> Dict[str, Any]:
        raw = self.chat(system_prompt, user_prompt, temperature)
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            start = raw.find("{")
            end = raw.rfind("}") + 1
            if start != -1 and end > start:
                return json.loads(raw[start:end])
            raise ValueError(f"Could not parse JSON from model response:\n{raw[:500]}")
