"""The one place this project calls a USER'S OWN endpoint, not a model.

Mirrors `client.py` — one HTTP call, one label on every request — but this one
calls whatever a use case's `endpoint:` block points at (hosted or localhost,
no distinction needed, both are just a URL) to capture what that feature
returns RIGHT NOW for a test case's input. `bench.py` judges that answer
through the same rubric as every candidate model, so the leaderboard shows a
"current (your endpoint)" row alongside them. This module never routes a
candidate model through the endpoint — it only ever captures a baseline.
"""
import json
import os
import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .config import Endpoint

_ENV_REF = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")


def _expand_env(value: str) -> str:
    """Substitutes ${VAR} from the environment — the same pattern this project
    already uses for API keys, so a header value never has to be committed."""
    return _ENV_REF.sub(lambda m: os.environ.get(m.group(1), ""), value or "")


def _dig(data: dict, dotted_path: str):
    """Pulls a nested field out of a parsed JSON body, e.g. 'data.reply'."""
    cur = data
    for part in dotted_path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            raise KeyError(f"response has no field '{dotted_path}' "
                           f"(stopped at {part!r})")
        cur = cur[part]
    return cur


async def call(endpoint: "Endpoint", input_text: str, *, label: str,
               timeout_s: float = 30.0) -> str:
    """One call to a use case's own endpoint. Raises with the label attached
    on failure, exactly like `client.call_json` does for a candidate model."""
    import httpx

    headers = {k: _expand_env(v) for k, v in (endpoint.headers or {}).items()}
    body = {endpoint.input_field: input_text}

    async with httpx.AsyncClient(timeout=timeout_s) as http:
        try:
            response = await http.request(endpoint.method, endpoint.url,
                                          headers=headers, json=body)
            response.raise_for_status()
            data = response.json()
        except Exception as exc:                    # noqa: BLE001
            raise RuntimeError(f"[{label}] {endpoint.url} failed: {exc}") from exc

    try:
        value = _dig(data, endpoint.response_field)
    except KeyError as exc:
        raise RuntimeError(f"[{label}] {exc} in response from {endpoint.url}") from exc
    return value if isinstance(value, str) else json.dumps(value)
