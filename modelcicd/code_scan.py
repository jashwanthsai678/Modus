"""Finds likely LLM call sites in a connected repo — by reading, not by regex.

WHY THIS CALLS A MODEL AT ALL. A fixed library of per-SDK patterns only ever
covers the languages and call styles someone thought to add — the opposite of
"any language, any project structure". Recognizing "this is a call to an LLM,
here is its model and its prompt" is a reading-comprehension task, the same
kind of judgment `judge.py` already asks a model to make about an ANSWER; this
module asks the same kind of question about a piece of CODE. It reuses
`client.call_json` — the one place this project calls a model — so there is
no second HTTP-calling implementation.

THIS SPENDS MONEY, LIKE `run`. One call per candidate file. Every caller
(`cli.py scan-repo`, the dashboard's scan route) shows a file count first and
asks before running it, exactly like `cli.py run` already does for a bench.

IT NEVER WRITES ANYTHING. A scan only ever proposes candidates — turning one
into a real `use_case.yaml` still goes through the same wizard, confirmed by
a person, that every other AI feature goes through.
"""
import asyncio
from pathlib import Path
from typing import Optional

from . import client as client_module

SCAN_CONCURRENCY = 6
DEFAULT_SCAN_MODEL = "openai/gpt-4o-mini"

_SKIP_DIRS = {".git", "node_modules", "venv", ".venv", "__pycache__", "dist",
             "build", "target", "vendor", ".idea", ".vscode", "bin", "obj"}
_SKIP_SUFFIXES = {
    ".png", ".jpg", ".jpeg", ".gif", ".ico", ".svg", ".webp", ".pdf",
    ".zip", ".tar", ".gz", ".exe", ".dll", ".so", ".dylib", ".woff",
    ".woff2", ".ttf", ".eot", ".mp4", ".mp3", ".lock", ".min.js",
}
# Plain substring matches, language-agnostic on purpose — this is a cheap
# prefilter to bound cost, not the detector itself. A file matching none of
# these is very unlikely to contain an LLM call; one matching several is
# prioritized when there are more candidates than `max_files`.
_KEYWORDS = [
    "openai", "anthropic", "claude", "gpt-", "gemini", "llama", "mistral",
    "groq", "fireworks", "openrouter", "together.ai", "chat.completions",
    "chatcompletion", "generatecontent", "messages.create", "chat_model",
    "llm", "completion(", "responses.create", "vertexai", "bedrock",
]


def candidate_files(repo_path, *, max_files: int = 60, max_file_bytes: int = 40_000
                    ) -> list:
    """Every file under `repo_path` worth sending to the scanner — skipping
    noise directories and binary-ish extensions, keyword-prefiltered, capped
    at `max_files` (kept by keyword-hit count when there are more matches
    than that)."""
    root = Path(repo_path)
    scored = []
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        if any(part in _SKIP_DIRS for part in p.parts):
            continue
        name_lower = p.name.lower()
        if any(name_lower.endswith(suf) for suf in _SKIP_SUFFIXES):
            continue
        try:
            if p.stat().st_size > max_file_bytes * 4:
                continue
            text = p.read_text(encoding="utf-8", errors="ignore").lower()
        except OSError:
            continue
        hits = sum(text.count(kw) for kw in _KEYWORDS)
        if hits:
            scored.append((hits, p))
    scored.sort(key=lambda t: t[0], reverse=True)
    return [p for _, p in scored[:max_files]]


def estimate(files: list) -> dict:
    return {"files": len(files), "calls": len(files)}


def summary(results: list) -> dict:
    """Counts by confidence — lets a caller say "3 high, 2 low" instead of
    just dumping a list, so a low-confidence or empty result reads as its
    own actionable state rather than quietly looking the same as success.
    This never claims the scanner found nothing wrong with the code it
    DIDN'T flag — only how much to trust what it DID flag."""
    high = sum(1 for r in results if r.get("confidence") == "high")
    low = sum(1 for r in results if r.get("confidence") != "high")
    return {"high": high, "low": low, "total": len(results)}


_PROMPT = """You are reading ONE source file to find where it calls an LLM API to
generate text (a chat/completion/generation call to any provider — OpenAI,
Anthropic, Google, Groq, a self-hosted model, anything). This is a code-reading
task, not a guess: only report a call that is actually in this file.

FILE: {path}
```
{content}
```

For each LLM call site found, report:
- model: the literal model id string if one is hardcoded (e.g. "gpt-4o-mini"),
  else null
- prompt: the system prompt / instructions text passed to the call, verbatim
  if it appears in the file, else a short (<200 char) description of what's
  passed
- line: the approximate line number of the call
- confidence: "high" if this is clearly a text-generation LLM call, "low" if
  you're not sure

Return ONLY valid JSON, no markdown fences:
{{"calls": [{{"model": "...", "prompt": "...", "line": 1, "confidence": "high"}}]}}

If there are no LLM call sites in this file, return {{"calls": []}}.
"""


async def scan_file(path: Path, *, scan_model: str, max_file_bytes: int = 40_000) -> list:
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")[:max_file_bytes]
    except OSError:
        return []
    try:
        data = await client_module.call_json(
            _PROMPT.format(path=path.name, content=text),
            model=scan_model, label=f"scan[{path.name}]", required=("calls",),
            temperature=0.0, max_tokens=1200)
    except Exception as exc:                        # noqa: BLE001
        return [{"file": str(path), "error": str(exc)[:200]}]
    out = []
    for c in data.get("calls") or []:
        out.append({"file": str(path), "model": c.get("model"),
                    "prompt": c.get("prompt"), "line": c.get("line"),
                    "confidence": c.get("confidence") or "low"})
    return out


async def scan_repo(repo_path, *, scan_model: str = DEFAULT_SCAN_MODEL,
                    max_files: int = 60) -> list:
    """Scans every candidate file concurrently, returns every call site
    found, high-confidence first. Deduplicates entries that are effectively
    the same call (same file, same model string)."""
    files = candidate_files(repo_path, max_files=max_files)
    gate = asyncio.Semaphore(SCAN_CONCURRENCY)

    async def one(p: Path) -> list:
        async with gate:
            return await scan_file(p, scan_model=scan_model)

    results = await asyncio.gather(*(one(p) for p in files))
    flat = [c for group in results for c in group]

    seen, deduped = set(), []
    for c in flat:
        key = (c.get("file"), c.get("model"), (c.get("prompt") or "")[:80])
        if key in seen:
            continue
        seen.add(key)
        deduped.append(c)
    deduped.sort(key=lambda c: c.get("confidence") != "high")
    return deduped
