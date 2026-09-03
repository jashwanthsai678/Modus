"""The one place this project calls a model. Everything else imports this.

STANDALONE ON PURPOSE. This is meant to be dropped into someone else's
repository, so it cannot depend on that repository's own LLM wrapper — it needs
its own, minimal, and boring: one HTTP call to an OpenAI-compatible endpoint,
JSON-mode requested, retried once on a parse failure with the error shown back
to the model.

A LABEL ON EVERY CALL, ALWAYS. `label` names the call in every log line and in
every exception. On a run judging forty candidates, "the model response was not
valid JSON" with no label is not a diagnosis.
"""
import json
import os
import re
from typing import Optional

DEFAULT_BASE_URL = "https://openrouter.ai/api/v1/chat/completions"

_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)


def _strip_fences(text: str) -> str:
    return _FENCE.sub("", text or "").strip()


async def call_json(prompt: str, *, model: str, label: str,
                    system: Optional[str] = None, required: tuple = (),
                    temperature: float = 0.4, max_tokens: int = 1500,
                    attempts: int = 3, timeout_s: float = 60.0) -> dict:
    """One JSON-returning call. Raises with the label attached on final failure.

    RETRIED WITH THE ERROR SHOWN BACK. A parse failure is a normal event on a
    forty-candidate run — some models ignore json_object mode, some wrap the
    answer in prose — and re-asking with the exact error attached fixes it far
    more often than re-rolling the same prompt blind.
    """
    import httpx

    api_key = (os.environ.get("OPENROUTER_API_KEY") or "").strip()
    if not api_key:
        raise RuntimeError(
            f"[{label}] OPENROUTER_API_KEY is not set. Model CICD calls every "
            f"candidate through OpenRouter; without a key nothing can run.")

    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    last_error = None
    async with httpx.AsyncClient(timeout=timeout_s) as http:
        for attempt in range(1, attempts + 1):
            try:
                response = await http.post(
                    DEFAULT_BASE_URL,
                    headers={"Authorization": f"Bearer {api_key}",
                            "Content-Type": "application/json",
                            "HTTP-Referer": "https://github.com/", "X-Title": "Model CICD"},
                    json={"model": model, "messages": messages,
                         "temperature": temperature, "max_tokens": max_tokens,
                         "response_format": {"type": "json_object"}})
                response.raise_for_status()
                payload = response.json()
                choice = (payload.get("choices") or [{}])[0]
                text = ((choice.get("message") or {}).get("content") or "").strip()
                if not text:
                    raise ValueError(f"empty response (finish_reason="
                                     f"{choice.get('finish_reason')})")
                data = json.loads(_strip_fences(text))
                missing = [k for k in required if k not in data]
                if missing:
                    raise ValueError(f"missing required key(s): {missing}")
                return data
            except Exception as exc:                # noqa: BLE001
                last_error = exc
                if attempt < attempts:
                    messages.append({"role": "user",
                                     "content": f"That was not valid — {exc}. "
                                                f"Return ONLY the JSON object, "
                                                f"nothing else."})
    raise RuntimeError(f"[{label}] {model} failed after {attempts} attempt(s): "
                      f"{last_error}")
