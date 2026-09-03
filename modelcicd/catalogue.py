"""What models exist right now, what they cost, and where to run them.

A SCHEDULED POLLER, NOT AN AGENT. OpenRouter publishes its whole catalogue as
structured JSON on a public endpoint — price, context length, JSON-mode support,
modality, all of it. An agent that "goes and reads the pricing page" adds
latency, cost, and a failure mode a `GET` request does not have: it can misread
a number and poison the shortlist silently. Nothing here calls a model.

MULTIPLE PROVIDERS, EACH TRUSTED FOR WHAT IT ACTUALLY PUBLISHES. OpenRouter
proxies most platforms through one endpoint with one consistent shape,
including price — that's still the default. Groq and Fireworks are also
registered here, each with its OWN row-normalizer (`_rows_groq`,
`_rows_fireworks`), because their `/models` endpoints do not necessarily
publish machine-readable pricing or JSON-mode support the way OpenRouter's
does. UNKNOWN IS NOT FREE, AND UNKNOWN IS NOT JSON-CAPABLE EITHER: a field
either provider's API doesn't expose comes back `None`/`False`, never
guessed — the same "wrong in the direction of excluding a maybe-good
candidate, not including a maybe-broken one" bias the price guardrail
already applies. A project picks which of these providers to actually
search (`project.providers`); adding a further provider is a new entry in
PROVIDERS plus its own `_rows_<name>` normalizer.

THE DEDUPE PROBLEM. The same weights hosted on two platforms are two different
artefacts — different quantisation, different context windows, different
JSON-mode support — and collapsing them into one row throws away the thing worth
measuring. So a logical model keeps its hosting options rather than replacing
them, and `key()` is a heuristic that errs toward SPLITTING: an extra row costs
a reader a moment; a wrongly merged row costs a wrong measurement.
"""
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

PROVIDERS = {
    "openrouter": {"url": "https://openrouter.ai/api/v1/models",
                   "key_env": "OPENROUTER_API_KEY", "needs_key": False},
    "groq": {"url": "https://api.groq.com/openai/v1/models",
             "key_env": "GROQ_API_KEY", "needs_key": True},
    "fireworks": {"url": "https://api.fireworks.ai/inference/v1/models",
                  "key_env": "FIREWORKS_API_KEY", "needs_key": True},
}

# Suffixes that describe ROUTING or PACKAGING, stripped when forming the
# logical key. `chat` and `turbo` are deliberately absent — they are identity
# for some model families (deepseek-chat vs deepseek-r1; gpt-3.5-turbo), and
# stripping them merged distinct models in early testing.
_ROUTING_TAGS = re.compile(
    r"[:@](free|nitro|floor|beta|extended|online|thinking|preview)$", re.I)
_PACKAGING = re.compile(
    r"-(instruct|it|hf|gguf|awq|gptq|fp8|fp16|bf16|q4|q8|versatile|"
    r"specdec|latest)\b", re.I)


def key(model_id: str) -> str:
    """The logical model behind a platform-specific id. A heuristic, documented
    as one: wrong in the direction of splitting rather than merging."""
    text = (model_id or "").strip().lower()
    text = _ROUTING_TAGS.sub("", text)
    if "/" in text:
        text = text.rsplit("/", 1)[-1]
    text = _PACKAGING.sub("", text)
    return re.sub(r"[^a-z0-9.]+", "-", text).strip("-")


def _price(value) -> Optional[float]:
    """Per-million-token price, or None for unknown.

    A NEGATIVE VALUE IS A SENTINEL, NOT A PRICE. OpenRouter returns "-1" for its
    own routed meta-models ("auto", "fusion") meaning "depends where this is
    routed". Multiplied through arithmetically that becomes -$1,000,000/M, which
    sorts CHEAPEST and clears every ceiling — the one bug in a cost guardrail
    that actually spends money.
    """
    try:
        per_million = float(value) * 1_000_000
    except (TypeError, ValueError):
        return None
    return round(per_million, 4) if per_million >= 0 else None


def _rows_openrouter(payload: dict) -> list:
    out = []
    for m in payload.get("data") or []:
        arch = m.get("architecture") or {}
        pricing = m.get("pricing") or {}
        top = m.get("top_provider") or {}
        out.append({
            "provider": "openrouter", "id": m.get("id"), "name": m.get("name"),
            "context": m.get("context_length") or top.get("context_length"),
            "price_in": _price(pricing.get("prompt")),
            "price_out": _price(pricing.get("completion")),
            "input_modalities": arch.get("input_modalities")
                                or ([arch.get("modality")] if arch.get("modality") else []),
            "output_modalities": arch.get("output_modalities") or [],
            # Every candidate is scored through a JSON-returning call; a model
            # without this cannot be judged, only guessed at.
            "json_mode": "response_format" in (m.get("supported_parameters") or []),
        })
    return out


def _rows_openai_compatible(provider: str, payload: dict) -> list:
    """Groq and Fireworks both expose an OpenAI-compatible `/models` list —
    just `{"data": [{"id": ...}, ...]}`, with no guaranteed price or
    JSON-mode field. Whatever context field IS present (naming varies) is
    used; price and json_mode default to unknown/false rather than guessed —
    an unpriced or unconfirmed-JSON model is excluded downstream by the same
    guardrails that already exclude any other unpriced model, never silently
    treated as free or capable."""
    out = []
    for m in payload.get("data") or []:
        if not m.get("id"):
            continue
        context = (m.get("context_window") or m.get("context_length")
                  or m.get("max_context_length"))
        out.append({
            "provider": provider, "id": m["id"], "name": m.get("id"),
            "context": context, "price_in": None, "price_out": None,
            "input_modalities": ["text"], "output_modalities": ["text"],
            "json_mode": False,
        })
    return out


def fetch(provider: str, *, timeout: float = 20.0) -> list:
    """One platform's catalogue. Returns [] and says why on failure — a platform
    being down must not take the whole poll with it."""
    spec = PROVIDERS.get(provider)
    if not spec:
        print(f"[modelcicd] unknown provider {provider!r}")
        return []
    headers = {}
    api_key = (os.environ.get(spec["key_env"]) or "").strip()
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    elif spec["needs_key"]:
        print(f"[modelcicd] {provider}: no {spec['key_env']} set — skipped.")
        return []
    try:
        import httpx
        response = httpx.get(spec["url"], headers=headers, timeout=timeout)
        response.raise_for_status()
        payload = response.json()
    except Exception as exc:                        # noqa: BLE001
        print(f"[modelcicd] {provider}: catalogue unreachable ({str(exc)[:140]})")
        return []
    normalize = (_rows_openrouter if provider == "openrouter"
                else lambda p: _rows_openai_compatible(provider, p))
    return [r for r in normalize(payload) if r.get("id")]


def poll(providers: Optional[list] = None) -> dict:
    """Every platform, normalised and grouped into logical models."""
    rows, counts = [], {}
    for provider in (providers or list(PROVIDERS)):
        got = fetch(provider)
        counts[provider] = len(got)
        rows += got

    models: dict = {}
    for row in rows:
        entry = models.setdefault(key(row["id"]), {"key": key(row["id"]), "hosts": []})
        entry["hosts"].append(row)

    for entry in models.values():
        priced = [h["price_out"] for h in entry["hosts"] if h["price_out"] is not None]
        contexts = [h["context"] for h in entry["hosts"] if h.get("context")]
        entry["cheapest_out"] = min(priced) if priced else None
        entry["max_context"] = max(contexts) if contexts else None
        entry["providers"] = sorted({h["provider"] for h in entry["hosts"]})
        entry["json_mode"] = any(h.get("json_mode") for h in entry["hosts"])
        entry["name"] = next((h.get("name") for h in entry["hosts"] if h.get("name")),
                             entry["key"])

    return {"fetchedAt": datetime.now(timezone.utc).isoformat(),
            "counts": counts, "models": models}


def cheapest_host_id(model: dict) -> Optional[str]:
    """The id `client.py` should actually call if this model wins — the cheapest
    host that publishes a real price."""
    hosts = [h for h in model.get("hosts") or [] if h.get("price_out") is not None]
    hosts.sort(key=lambda h: h["price_out"])
    return hosts[0]["id"] if hosts else None


def save(catalogue: dict, path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(catalogue, indent=2, ensure_ascii=False), encoding="utf-8")


def load(path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


DEFAULT_PATH = Path(__file__).resolve().parent.parent / "out" / "catalogue.json"


def load_or_poll(path=DEFAULT_PATH, *, refresh: bool = False) -> dict:
    """The cached catalogue if one is on disk and a refresh wasn't asked for,
    otherwise a fresh poll (saved for next time). One implementation shared by
    the CLI and the scheduler, so both agree on what "the catalogue" is."""
    p = Path(path)
    if not refresh and p.exists():
        return load(p)
    cat = poll()
    save(cat, p)
    return cat
