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


class RateLimitedError(RuntimeError):
    """The platform said 429. This is a REAL, OBSERVED signal — not a
    guessed rate limit — surfaced distinctly so a leaderboard can show
    "this candidate got rate limited during even a light bench" rather than
    folding it into a generic failure."""


class ModelUnavailableError(RuntimeError):
    """The platform rejected the request at the model level — 400 or 404.

    CARRIES THE PLATFORM'S OWN MESSAGE, NOT A PARAPHRASE, and that is the
    whole point of this class. Checked live against OpenRouter, the status
    code alone says almost nothing:

        404  "This model is unavailable for free. The paid version is
              available now - use this slug instead: minimax/minimax-m3"
                                              -> gone for good, and the
                                                 message names the fix
        404  "Provider returned error"  metadata: {provider_name: Nvidia}
                                              -> the UPSTREAM had a moment;
                                                 the model is fine
        400  "vendor/x is not a valid model ID"
                                              -> never existed

    So two 404s mean opposite things, and the most clear-cut death is a 400.
    This class therefore makes no claim about permanence — it just preserves
    what the platform said. Deciding which of these justifies dropping a
    candidate is policy, and it lives in `health.py`."""


def _strip_fences(text: str) -> str:
    return _FENCE.sub("", text or "").strip()


def _raise_if_rate_limited(response) -> None:
    if response.status_code == 429:
        raise RateLimitedError("rate limited (429 Too Many Requests)")


def _platform_error(response) -> str:
    """The platform's own words for why it refused. Falls back to raw text,
    and never raises — this runs on an already-failing path."""
    try:
        payload = response.json()
    except Exception:                               # noqa: BLE001
        return (getattr(response, "text", "") or "")[:300]
    error = payload.get("error") if isinstance(payload, dict) else None
    if not isinstance(error, dict):
        return str(error or payload)[:300]
    message = str(error.get("message") or "")
    provider = (error.get("metadata") or {}).get("provider_name")
    if provider:
        # Named because it's the tell that distinguishes "this model is gone"
        # from "this model's host had a bad second".
        message = f"{message} (upstream provider: {provider})"
    return message[:300]


def _raise_if_model_rejected(response) -> None:
    if response.status_code in (400, 404):
        raise ModelUnavailableError(
            f"HTTP {response.status_code}: {_platform_error(response)}")


async def call_json(prompt: str, *, model: str, label: str,
                    base_url: str = DEFAULT_BASE_URL,
                    api_key_env: str = "OPENROUTER_API_KEY",
                    system: Optional[str] = None, required: tuple = (),
                    temperature: float = 0.4, max_tokens: int = 1500,
                    attempts: int = 3, timeout_s: float = 60.0,
                    cache=None, usage_sink: Optional[dict] = None) -> dict:
    """One JSON-returning call. Raises with the label attached on final failure.

    RETRIED WITH THE ERROR SHOWN BACK. A parse failure is a normal event on a
    forty-candidate run — some models ignore json_object mode, some wrap the
    answer in prose — and re-asking with the exact error attached fixes it far
    more often than re-rolling the same prompt blind.

    `base_url`/`api_key_env` PARAMETERIZE THE ENDPOINT, THIS FILE STAYS
    STANDALONE. This project calls other platforms too (see `catalogue.py`'s
    PROVIDERS), but this file must not import that — it's meant to be
    dropped into someone else's repo on its own. A caller that knows which
    platform a candidate came from (`sandbox.py`) resolves the URL and key
    env var itself and passes them in; every default here reproduces exactly
    what this function always did — OpenRouter, `OPENROUTER_API_KEY`.

    `cache` IS DUCK-TYPED, FOR THE SAME REASON. Anything exposing
    `fingerprint(**kwargs) -> str`, `get(key)` and `put(key, value,
    model=...)` works — `modelcicd/cache.py` is one such thing, but this
    file does not import it, so dropping this module into another repo
    still needs nothing but httpx. `None`, the default, is exactly the
    behavior this function has always had: every call goes to the network.

    ONLY A SUCCESS IS EVER STORED. The write below sits after the `required`
    check on the one path that returns — a rate limit, a parse failure, a
    timeout, an HTTP error all leave the cache untouched and get retried on
    the next run. Caching a failure would pin a candidate to a bad afternoon.

    `usage_sink`, WHEN GIVEN, IS FILLED ON A GENUINE SUCCESS ONLY — never on
    a cache hit (nothing was actually bought) and never on failure. Carries
    `promptTokens`/`completionTokens` straight from the platform's own
    `usage` block (every provider here is OpenAI-compatible, so the shape is
    the same one across all of them) and `attempts`, the 1-based try this
    succeeded on — a model that needed a retry to produce valid JSON is a
    real reliability signal, currently invisible to every caller.
    """
    import httpx

    cache_key = None
    if cache is not None:
        cache_key = cache.fingerprint(
            model=model, prompt=prompt, system=system, temperature=temperature,
            max_tokens=max_tokens, base_url=base_url, required=tuple(required))
        cached = cache.get(cache_key)
        if cached is not None:
            return cached

    api_key = (os.environ.get(api_key_env) or "").strip()
    if not api_key:
        raise RuntimeError(
            f"[{label}] {api_key_env} is not set. Without a key nothing can run.")

    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    last_error = None
    async with httpx.AsyncClient(timeout=timeout_s) as http:
        for attempt in range(1, attempts + 1):
            try:
                response = await http.post(
                    base_url,
                    headers={"Authorization": f"Bearer {api_key}",
                            "Content-Type": "application/json",
                            "HTTP-Referer": "https://github.com/", "X-Title": "Modus"},
                    json={"model": model, "messages": messages,
                         "temperature": temperature, "max_tokens": max_tokens,
                         "response_format": {"type": "json_object"}})
                _raise_if_rate_limited(response)
                _raise_if_model_rejected(response)
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
                if cache is not None and cache_key:
                    cache.put(cache_key, data, model=model)
                if usage_sink is not None:
                    usage = payload.get("usage") or {}
                    usage_sink["promptTokens"] = usage.get("prompt_tokens")
                    usage_sink["completionTokens"] = usage.get("completion_tokens")
                    usage_sink["attempts"] = attempt
                return data
            except (RateLimitedError, ModelUnavailableError):
                # Fail fast — retrying immediately just hits the same limit
                # again (and a 404 will never stop being a 404), while the
                # "that wasn't valid JSON" retry message below would be
                # nonsensical for an HTTP error anyway.
                raise
            except Exception as exc:                # noqa: BLE001
                last_error = exc
                if attempt < attempts:
                    messages.append({"role": "user",
                                     "content": f"That was not valid — {exc}. "
                                                f"Return ONLY the JSON object, "
                                                f"nothing else."})
    raise RuntimeError(f"[{label}] {model} failed after {attempts} attempt(s): "
                      f"{last_error}")
