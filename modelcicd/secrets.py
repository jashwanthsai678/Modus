"""The one place this project writes an API key — into `.env`, the same file
`OPENROUTER_API_KEY` already lives in. No new secret-storage mechanism: the
interface (CLI and dashboard) can now ASK for a key instead of requiring you
to hand-edit this file, but where it ends up is exactly where it always did.

NEVER RETURNS A VALUE. `status()` reports whether a key is set, never what
it is — nothing here is a place a secret gets displayed back.
"""
import os
from pathlib import Path
from typing import Optional

from . import catalogue as catalogue_module

DEFAULT_ENV_PATH = Path(__file__).resolve().parent.parent / ".env"


def _key_env(provider: str) -> str:
    spec = catalogue_module.PROVIDERS.get(provider)
    if not spec:
        raise ValueError(f"unknown provider {provider!r}.")
    return spec["key_env"]


def set_key(provider: str, value: str, *, env_path: Optional[Path] = None) -> None:
    """Writes `value` for `provider`'s key into `.env` — replacing an
    existing line for that variable if one exists, appending one if not,
    and leaving every other line untouched. Also updates `os.environ`
    immediately, so a dashboard process already running picks it up on its
    very next request without needing a restart."""
    key_env = _key_env(provider)
    path = Path(env_path or DEFAULT_ENV_PATH)
    lines = path.read_text(encoding="utf-8").splitlines() if path.exists() else []

    prefix = f"{key_env}="
    for i, line in enumerate(lines):
        if line.startswith(prefix):
            lines[i] = f"{key_env}={value}"
            break
    else:
        lines.append(f"{key_env}={value}")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    os.environ[key_env] = value


def status() -> dict:
    """{provider: bool} for every registered provider — whether a key is
    set, never the key itself. Checks the live environment (which `.env` was
    already loaded into at process start, and which `set_key` updates
    immediately) rather than re-reading any particular file, so this always
    reflects what a real call would actually use right now."""
    return {p: bool((os.environ.get(spec["key_env"]) or "").strip())
           for p, spec in catalogue_module.PROVIDERS.items()}
