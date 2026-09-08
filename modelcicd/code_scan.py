"""Finds likely LLM call sites in a connected repo — by reading, not by regex.

WHY THIS CALLS A MODEL AT ALL. A fixed library of per-SDK patterns only ever
covers the languages and call styles someone thought to add — the opposite of
"any language, any project structure". Recognizing "this is a call to an LLM,
here is its model and its prompt" is a reading-comprehension task, the same
kind of judgment `judge.py` already asks a model to make about an ANSWER; this
module asks the same kind of question about a piece of CODE. It reuses
`client.call_json` — the one place this project calls a model — so there is
no second HTTP-calling implementation.

THIS SPENDS MONEY, LIKE `run`. One call per candidate file, regardless of how
many local imports get bundled into that call for context (see below) — every
caller (`cli.py scan-repo`, the dashboard's scan route) shows a file count
first and asks before running it, exactly like `cli.py run` already does for
a bench.

IT NEVER WRITES ANYTHING. A scan only ever proposes candidates — turning one
into a real `use_case.yaml` still goes through the same wizard, confirmed by
a person, that every other AI feature goes through.

FOLLOWING LOCAL IMPORTS. A file that only builds a prompt or only picks a
model, while the actual call lives in a file it imports (or vice versa), is
invisible to a scanner that reads one file in total isolation — this is the
`agent/router.py` + `agent/prompts.py` split kind of case. `local_imports()`
is a cheap, best-effort, regex-level detector (same "prefilter, not the
detector" spirit as the keyword prefilter below) that finds a file's own
in-repo imports — never third-party packages, which aren't part of the
connected project anyway — and `scan_file` bundles their content into the
SAME single call as extra context, clearly separated from the file actually
being scanned so line numbers stay unambiguous. Still one model call per
candidate file; the call just sees more of the picture.
"""
import asyncio
import re
from pathlib import Path
from typing import Optional

from . import client as client_module

SCAN_CONCURRENCY = 6
DEFAULT_SCAN_MODEL = "nvidia/nemotron-3-ultra-550b-a55b:free"
MAX_IMPORT_FILES = 3

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


def split_results(results: list) -> tuple:
    """One shared place to separate real candidates from failed calls —
    every caller used to do this filtering itself, which is how a scan that
    failed outright ended up looking identical to one that genuinely found
    nothing (the errors were simply discarded before anyone decided what to
    show). Returns (found, errors)."""
    found = [r for r in results if not r.get("error")]
    errors = [r for r in results if r.get("error")]
    return found, errors


# Local-import detection: cheap, regex-level, best-effort — same "prefilter,
# not the detector" spirit as the keyword prefilter. Only ever follows an
# import that resolves to a real file INSIDE the repo; a third-party package
# import is never followed (its source isn't part of the connected project).
_PY_FROM_RE = re.compile(r'^\s*from\s+(?P<dots>\.*)(?P<mod>[\w.]*)\s+import\s+(?P<names>[^\n#]+)', re.M)
_PY_IMPORT_RE = re.compile(r'^\s*import\s+(?P<mod>[\w][\w.]*)', re.M)
_JS_IMPORT_RE = re.compile(r'''(?:require\(\s*|from\s+)['"](?P<mod>\.{1,2}/[^'"]+)['"]''')


def _under_repo(repo_root: Path, candidate: Path) -> Optional[Path]:
    try:
        resolved = candidate.resolve()
        resolved.relative_to(repo_root)
    except (OSError, ValueError):
        return None
    return resolved if resolved.is_file() else None


def _resolve_py_target(dots: str, mod: str, names: str, from_dir: Path, repo_root: Path) -> list:
    hits = []
    if dots:
        base = from_dir
        for _ in range(len(dots) - 1):
            base = base.parent
        targets = [mod] if mod else [n.split(" as ")[0].strip().rstrip(",")
                                     for n in re.split(r'[,\s]+', names.strip())]
    else:
        if not mod:
            return hits
        base = repo_root
        targets = [mod]
    for t in targets:
        t = t.strip()
        if not t or not all(part.isidentifier() for part in t.split(".")):
            continue
        candidate = base.joinpath(*t.split("."))
        hit = (_under_repo(repo_root, candidate.with_suffix(".py"))
              or _under_repo(repo_root, candidate / "__init__.py"))
        if hit:
            hits.append(hit)
    return hits


def _resolve_js_target(mod: str, from_dir: Path, repo_root: Path) -> Optional[Path]:
    base = (from_dir / mod)
    for suf in ("", ".js", ".jsx", ".ts", ".tsx",
               "/index.js", "/index.jsx", "/index.ts", "/index.tsx"):
        hit = _under_repo(repo_root, Path(str(base) + suf))
        if hit:
            return hit
    return None


def local_imports(path: Path, repo_root: Path, *, max_files: int = MAX_IMPORT_FILES) -> list:
    """This file's own local (in-repo) imports, resolved to real file paths —
    best-effort, Python- and JS/TS-style import syntax only, capped at
    `max_files`."""
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return []
    from_dir = path.resolve().parent
    hits = []
    for m in _PY_FROM_RE.finditer(text):
        hits.extend(_resolve_py_target(m.group("dots"), m.group("mod"), m.group("names"),
                                       from_dir, repo_root))
    for m in _PY_IMPORT_RE.finditer(text):
        hits.extend(_resolve_py_target("", m.group("mod"), "", from_dir, repo_root))
    for m in _JS_IMPORT_RE.finditer(text):
        hit = _resolve_js_target(m.group("mod"), from_dir, repo_root)
        if hit:
            hits.append(hit)

    seen, deduped = set(), []
    for h in hits:
        if h == path.resolve() or h in seen:
            continue
        seen.add(h)
        deduped.append(h)
    return deduped[:max_files]


_PROMPT = """You are reading ONE source file to find where it calls an LLM API to
generate text (a chat/completion/generation call to any provider — OpenAI,
Anthropic, Google, Groq, a self-hosted model, anything). This is a code-reading
task, not a guess: only report a call that is actually in this file.

FILE: {path}
```
{content}
```
{imports_block}
For each LLM call site found IN THE FILE ABOVE (not in any imported file shown
below it for context — those are only there to help you follow the prompt or
model through, "line" must always refer to FILE, never to an imported file),
report:
- model: the literal model id string if one is hardcoded (e.g. "gpt-4o-mini"),
  else null
- prompt: the system prompt / instructions text passed to the call, verbatim
  if it appears in the file OR one of the imported files shown below, else a
  short (<200 char) description of what's passed
- inputStructure: what the CODE actually passes as input to this call — read
  the real variables/parameters being sent, not the prompt's wording. If it's
  one plain string, say so plainly (e.g. "a single string: the ticket text").
  If it's built from multiple fields (e.g. an object with several named
  values interpolated into the prompt, or a structured payload), name each
  field you can see in the code. Never invent a field that isn't actually
  read or passed somewhere in the file.
- outputStructure: what shape of answer the code expects back — read how the
  response is parsed or what format it's told to return (e.g. an explicit
  "return exactly {{...}}" instruction, a `response_format`/JSON-mode setting,
  or how the returned value is used afterward). Plain text if nothing in the
  code suggests otherwise.
- line: the approximate line number of the call, in FILE
- confidence: "high" if this is clearly a text-generation LLM call, "low" if
  you're not sure

Return ONLY valid JSON, no markdown fences:
{{"calls": [{{"model": "...", "prompt": "...", "inputStructure": "...",
  "outputStructure": "...", "line": 1, "confidence": "high"}}]}}

If there are no LLM call sites in this file, return {{"calls": []}}.
"""

_IMPORTS_BLOCK = """
For additional context only — these are files {path} imports locally in this
repo. Use them to resolve a prompt or model that FILE builds via a function
call into one of these, but every "line" you report must still point into
FILE, never into one of these:
{sections}
"""


def _imports_block(path: Path, repo_root: Optional[Path], max_file_bytes: int,
                   max_import_files: int) -> str:
    if repo_root is None:
        return ""
    imports = local_imports(path, repo_root, max_files=max_import_files)
    if not imports:
        return ""
    sections = []
    for p in imports:
        try:
            rel = p.relative_to(repo_root)
        except ValueError:
            rel = p
        try:
            text = p.read_text(encoding="utf-8", errors="ignore")[:max_file_bytes]
        except OSError:
            continue
        sections.append(f"IMPORTED FILE: {rel}\n```\n{text}\n```")
    if not sections:
        return ""
    return _IMPORTS_BLOCK.format(path=path.name, sections="\n\n".join(sections))


async def scan_file(path: Path, *, scan_model: str, max_file_bytes: int = 40_000,
                    repo_root: Optional[Path] = None,
                    max_import_files: int = MAX_IMPORT_FILES) -> list:
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")[:max_file_bytes]
    except OSError:
        return []
    imports_block = _imports_block(path, repo_root, max_file_bytes, max_import_files)
    try:
        data = await client_module.call_json(
            _PROMPT.format(path=path.name, content=text, imports_block=imports_block),
            model=scan_model, label=f"scan[{path.name}]", required=("calls",),
            temperature=0.0, max_tokens=1200)
    except Exception as exc:                        # noqa: BLE001
        return [{"file": str(path), "error": str(exc)[:200]}]
    out = []
    for c in data.get("calls") or []:
        out.append({"file": str(path), "model": c.get("model"),
                    "prompt": c.get("prompt"),
                    "inputStructure": c.get("inputStructure"),
                    "outputStructure": c.get("outputStructure"),
                    "line": c.get("line"),
                    "confidence": c.get("confidence") or "low"})
    return out


async def scan_repo(repo_path, *, scan_model: str = DEFAULT_SCAN_MODEL,
                    max_files: int = 60) -> list:
    """Scans every candidate file concurrently, returns every call site
    found, high-confidence first. Deduplicates entries that are effectively
    the same call (same file, same model string)."""
    repo_root = Path(repo_path).resolve()
    files = candidate_files(repo_path, max_files=max_files)
    gate = asyncio.Semaphore(SCAN_CONCURRENCY)

    async def one(p: Path) -> list:
        async with gate:
            return await scan_file(p, scan_model=scan_model, repo_root=repo_root)

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


# ── Test-case generation: draft, never final ─────────────────────────────
#
# LOW-RISK HALF, HIGHER-RISK HALF, BOTH STILL A DRAFT. Generating realistic
# sample INPUTS from a prompt is safe to automate — a model guessing "here
# are plausible support tickets" is a genuinely useful shortcut. Generating
# the REFERENCE answer is the riskier half: a single generated answer
# becoming silent ground truth, never looked at by a person, is exactly the
# failure mode discussed at length before this was built — an AI defining
# what "correct" means with nobody checking it. So this generates both, but
# NOTHING here writes a use_case.yaml directly — every caller is expected to
# show these to a person before `wizard.to_yaml` ever saves them, the same
# "propose, then confirm" shape the scanner itself already uses.

_TEST_CASE_PROMPT = """You are drafting realistic test cases for an AI feature, so its
candidate models can be benchmarked against real-looking inputs.

THE FEATURE'S SYSTEM PROMPT (this is handled separately by the code, already sent
on every call — do not repeat it anywhere in what you write below):
{prompt}

HOW THE CODE WIRES UP A CALL, FOR YOUR BACKGROUND ONLY — this describes the
code's internal request format, NOT the shape of the "input" you should write:
  what it receives: {input_structure}
  what it's expected to return: {output_structure}

"input" MUST BE PLAIN CONTENT ONLY — the raw text (or data) a real end user would
actually type or send, and NOTHING else. Never a JSON messages array, never the
system prompt repeated, never any wrapper object — just the message itself, e.g.
"My order hasn't arrived yet and it's been two weeks." The system prompt and any
message-array wrapping is already handled elsewhere; writing it into "input" would
send it twice.

Write {count} realistic, DIVERSE sample inputs a real user of this feature might
actually send — vary them (different tones, lengths, edge cases) rather than
near-duplicates of each other. For each, also draft what a genuinely good answer
would look like — this is a STARTING POINT for a human to review and edit, not a
guaranteed-correct answer, so write your honest best attempt rather than a token
placeholder.

Return ONLY valid JSON, no markdown fences:
{{"cases": [{{"id": "short_snake_case_id", "input": "...", "reference": "..."}}]}}
"""


async def generate_test_cases(prompt: str, *, input_structure: Optional[str] = None,
                              output_structure: Optional[str] = None,
                              model: str, count: int = 4) -> list:
    """Drafts `count` {id, input, reference} test cases from a feature's
    system prompt. Returns [] on any failure — the caller falls back to an
    empty test-case list a person fills in by hand, the same starting point
    the wizard already offers when nothing was auto-generated."""
    try:
        data = await client_module.call_json(
            _TEST_CASE_PROMPT.format(
                prompt=prompt,
                input_structure=input_structure or "not determined — infer from the prompt",
                output_structure=output_structure or "not determined — infer from the prompt",
                count=count),
            model=model, label="generate-test-cases", required=("cases",),
            temperature=0.7, max_tokens=2000)
    except Exception:                                   # noqa: BLE001
        return []
    out = []
    for i, c in enumerate(data.get("cases") or []):
        if not isinstance(c, dict):
            continue
        cid = _as_text(c.get("id")) or f"case_{i + 1}"
        cin = _as_text(c.get("input"))
        if not cin:
            continue
        out.append({"id": cid, "input": cin, "reference": _as_text(c.get("reference")) or None})
    return out


def _as_text(value) -> str:
    """A model asked for a JSON string field doesn't always give one — a
    free model in particular may return a list of fragments instead of one
    joined string. Coerces whatever came back into plain text rather than
    crashing on it; never invents content that wasn't there."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, (list, tuple)):
        return " ".join(_as_text(v) for v in value if v is not None).strip()
    return str(value).strip()


def suggest_name(file: str) -> str:
    """A starting-point feature name from a scanned candidate's file path —
    always shown for editing, never used to silently name anything. E.g.
    "support_bot/bot.py" -> "support_bot_bot"."""
    parts = Path(file).with_suffix("").parts[-2:]
    raw = "_".join(parts) or "ai_feature"
    safe = "".join(ch if ch.isalnum() else "_" for ch in raw.lower())
    while "__" in safe:
        safe = safe.replace("__", "_")
    return safe.strip("_") or "ai_feature"
