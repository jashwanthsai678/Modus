"""Turns a person's answers into a valid use_case.yaml — the ONE place that
happens, shared by the CLI's interactive prompts and the dashboard's form.

WHY THIS EXISTS SEPARATELY FROM config.py. `config.py` reads and validates a
YAML file that already exists; this module is what PRODUCES that file from
someone answering questions instead of hand-writing YAML. Keeping it as one
shared module — not duplicated logic in `cli.py` and `dashboard.py` — is the
same discipline this project already applies to `state.approve()`: exactly
one implementation, however it gets triggered.

VALIDATES BEFORE WRITING. Every check here mirrors what `config.load` would
reject anyway (a name, a system prompt, at least one test case, a rubric
somewhere) — catching it here means a failed wizard prints one clear message
instead of writing a file that then fails to load.
"""
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import yaml

from .config import DEFAULT_JUDGE_MODEL

DEFAULT_TIERS = ["free", "paid-low", "paid-mid"]


@dataclass
class WizardFields:
    name: str = ""
    description: str = ""
    system_prompt: str = ""
    test_cases: list = field(default_factory=list)   # [{"id","input","reference"}]
    rubric: list = field(default_factory=list)        # [{"id","description","weight"}]
    max_price_in: float = 0.50
    max_price_out: float = 3.00
    min_context: int = 32_000
    require_json: bool = True
    allow_free: bool = True
    tiers: list = field(default_factory=lambda: list(DEFAULT_TIERS))
    judge_model: str = DEFAULT_JUDGE_MODEL
    max_tokens: int = 1200
    notify_email: Optional[str] = None
    min_improvement: float = 0.20
    endpoint_url: Optional[str] = None
    endpoint_method: str = "POST"
    endpoint_input_field: str = "input"
    endpoint_response_field: str = "output"
    endpoint_header: Optional[str] = None   # one "Name: value" line, e.g. "Authorization: Bearer ${VAR}"
    endpoint_timeout_seconds: float = 30.0
    schedule_interval_days: Optional[int] = None
    code_file: Optional[str] = None            # relative to the project's repo_path
    code_current_model: Optional[str] = None   # the exact string hardcoded there right now
    estimated_calls_per_day: Optional[int] = None   # rough usage, for rate-limit context
    input_structure: Optional[str] = None      # what the code sends, from a scan — display only
    output_structure: Optional[str] = None     # what the code expects back, from a scan — display only


def defaults_from_project(defaults: dict) -> WizardFields:
    """A `WizardFields` carrying only a project's SHARED settings (rubric,
    price ceiling, judge, notify threshold) — everything feature-specific
    (name, prompt, test cases, endpoint, schedule, code target) stays at its
    own dataclass default, for a caller to fill in per feature on top of
    this. One place both the single-feature wizard and a bulk-create flow
    apply the same project template from, so they can never disagree about
    what "the project's defaults" means."""
    f = WizardFields()
    f.rubric = [dict(c) for c in (defaults.get("rubric") or [])]
    f.max_price_in = float(defaults.get("maxPriceIn", f.max_price_in))
    f.max_price_out = float(defaults.get("maxPriceOut", f.max_price_out))
    f.min_context = int(defaults.get("minContext", f.min_context))
    f.require_json = bool(defaults.get("requireJson", f.require_json))
    f.allow_free = bool(defaults.get("allowFree", f.allow_free))
    f.tiers = list(defaults.get("tiers") or f.tiers)
    f.judge_model = defaults.get("judgeModel") or f.judge_model
    f.max_tokens = int(defaults.get("maxTokens", f.max_tokens))
    f.min_improvement = float(defaults.get("minImprovement", f.min_improvement))
    return f


def validate(f: WizardFields) -> list:
    errors = []
    if not f.name or not f.name.strip():
        errors.append("give this AI feature a name.")
    if not f.system_prompt or not f.system_prompt.strip():
        errors.append("a system prompt is required — it's what every candidate is judged against.")
    cases = [tc for tc in f.test_cases if (tc.get("id") or "").strip() and (tc.get("input") or "").strip()]
    if not cases:
        errors.append("at least one test case (an id and an input) is required.")
    default_rubric = [c for c in f.rubric if (c.get("id") or "").strip() and (c.get("description") or "").strip()]
    if not default_rubric:
        errors.append("at least one rubric criterion (an id and a description) is required — "
                      "the judge has nothing to score against otherwise.")
    if f.endpoint_url and f.endpoint_header and ":" not in f.endpoint_header:
        errors.append("the endpoint header must look like 'Name: value'.")
    if f.code_file and not (f.code_current_model or "").strip():
        errors.append("a code target needs the exact model string currently hardcoded "
                      "there — the patch step has nothing to search for otherwise.")
    return errors


def _headers(f: WizardFields) -> dict:
    if not f.endpoint_header or ":" not in f.endpoint_header:
        return {}
    name, _, value = f.endpoint_header.partition(":")
    return {name.strip(): value.strip()}


def to_yaml(f: WizardFields) -> str:
    """Renders the exact schema `config.load` expects. Blank/incomplete rows
    in test_cases and rubric are dropped, not written as broken entries."""
    doc = {
        "useCase": f.name.strip(),
        "description": f.description or "",
        "systemPrompt": f.system_prompt,
        "testCases": [
            {"id": tc["id"].strip(), "input": tc["input"],
             **({"reference": tc["reference"]} if tc.get("reference") else {})}
            for tc in f.test_cases
            if (tc.get("id") or "").strip() and (tc.get("input") or "").strip()
        ],
        "rubric": [
            {"id": c["id"].strip(), "description": c["description"],
             "weight": float(c.get("weight") or 1.0)}
            for c in f.rubric
            if (c.get("id") or "").strip() and (c.get("description") or "").strip()
        ],
        "guardrails": {
            "maxPriceIn": f.max_price_in, "maxPriceOut": f.max_price_out,
            "minContext": f.min_context, "requireJson": f.require_json,
            "allowFree": f.allow_free, "tiers": f.tiers or list(DEFAULT_TIERS),
        },
        "judgeModel": f.judge_model, "maxTokens": f.max_tokens,
        "notify": {"email": f.notify_email or None, "minImprovement": f.min_improvement},
    }
    if f.endpoint_url:
        doc["endpoint"] = {
            "url": f.endpoint_url, "method": f.endpoint_method or "POST",
            "inputField": f.endpoint_input_field or "input",
            "responseField": f.endpoint_response_field or "output",
            "headers": _headers(f), "timeoutSeconds": f.endpoint_timeout_seconds,
        }
    if f.schedule_interval_days:
        doc["schedule"] = {"intervalDays": int(f.schedule_interval_days)}
    if f.code_file:
        doc["codeTarget"] = {"file": f.code_file, "currentModel": f.code_current_model}
    if f.estimated_calls_per_day:
        doc["usage"] = {"callsPerDay": int(f.estimated_calls_per_day)}
    if f.input_structure:
        doc["inputStructure"] = f.input_structure
    if f.output_structure:
        doc["outputStructure"] = f.output_structure
    return yaml.safe_dump(doc, sort_keys=False, allow_unicode=True)


# ── CLI collection ───────────────────────────────────────────────────────────

def _ask(prompt_fn: Callable, text: str, default: str = "") -> str:
    suffix = f" [{default}]" if default else ""
    try:
        reply = prompt_fn(f"{text}{suffix}: ").strip()
    except EOFError:
        reply = ""
    return reply or default


def collect_cli(prompt_fn: Callable = input, *, repo_slug: Optional[str] = None,
                prefill: Optional[dict] = None) -> WizardFields:
    """Interactive prompts. Loops for test cases and rubric criteria until an
    empty id is entered. `repo_slug` — a project with a connected repo — adds
    an optional "view a file for reference" step and the code-target
    questions; without it, both are skipped. `prefill` — one candidate from
    `code_scan.scan_repo` — offers its detected prompt/model/file as a
    starting point, always shown and confirmable, never accepted silently."""
    f = WizardFields()
    f.name = _ask(prompt_fn, "AI feature name (e.g. support_bot_reply)")
    f.description = _ask(prompt_fn, "one-line description")

    if repo_slug and not prefill:
        from . import project as project_module
        print("\nThe connected repo is available for reference. View a file "
             "before writing the system prompt?")
        while True:
            rel = _ask(prompt_fn, "  file path relative to the repo (blank to continue)")
            if not rel:
                break
            try:
                print(f"\n--- {rel} ---")
                print(project_module.read_repo_file(repo_slug, rel))
                print(f"--- end {rel} ---\n")
            except (ValueError, FileNotFoundError) as exc:
                print(f"  {exc}")

    used_prefill_prompt = False
    if prefill and prefill.get("prompt"):
        print(f"\nScan found this in {prefill.get('file')} "
             f"(line {prefill.get('line')}, confidence {prefill.get('confidence')}):")
        print(f"\n--- detected prompt ---\n{prefill['prompt']}\n--- end ---\n")
        use_it = _ask(prompt_fn, "use this as the system prompt? [Y/n]", "y")
        if use_it.lower() in ("y", "yes"):
            f.system_prompt = prefill["prompt"]
            used_prefill_prompt = True
    if prefill:
        f.input_structure = prefill.get("inputStructure") or None
        f.output_structure = prefill.get("outputStructure") or None

    if not used_prefill_prompt:
        print("System prompt — paste it, then an empty line to finish:")
        lines = []
        while True:
            try:
                line = prompt_fn("")
            except EOFError:
                break
            if not line:
                break
            lines.append(line)
        f.system_prompt = "\n".join(lines)

    print("\nTest cases — enter an id, then its input. Empty id to stop.")
    while True:
        tc_id = _ask(prompt_fn, "  test case id (blank to stop)")
        if not tc_id:
            break
        tc_input = _ask(prompt_fn, "  input")
        tc_ref = _ask(prompt_fn, "  reference answer (optional, blank for none)")
        f.test_cases.append({"id": tc_id, "input": tc_input, "reference": tc_ref or None})

    print("\nRubric — what the judge scores, 1-5. Empty id to stop.")
    while True:
        c_id = _ask(prompt_fn, "  criterion id (blank to stop)")
        if not c_id:
            break
        c_desc = _ask(prompt_fn, "  description")
        c_weight = _ask(prompt_fn, "  weight", "1.0")
        f.rubric.append({"id": c_id, "description": c_desc, "weight": float(c_weight or 1.0)})

    f.max_price_in = float(_ask(prompt_fn, "max price in ($/1M tokens)", str(f.max_price_in)))
    f.max_price_out = float(_ask(prompt_fn, "max price out ($/1M tokens)", str(f.max_price_out)))
    f.min_context = int(_ask(prompt_fn, "minimum context length", str(f.min_context)))
    f.judge_model = _ask(prompt_fn, "judge model (must not be a candidate)", f.judge_model)
    f.notify_email = _ask(prompt_fn, "notify email (optional)") or None
    endpoint_url = _ask(prompt_fn, "live endpoint URL for a baseline comparison (optional)")
    f.endpoint_url = endpoint_url or None
    interval = _ask(prompt_fn, "re-run automatically every N days (optional)")
    f.schedule_interval_days = int(interval) if interval else None
    calls = _ask(prompt_fn, "approximate calls per day this feature will get "
                            "(optional — used to flag rate-limit risk, never to filter candidates)")
    f.estimated_calls_per_day = int(calls) if calls else None

    if repo_slug:
        default_file, default_model = "", ""
        if prefill and prefill.get("file"):
            default_model = prefill.get("model") or ""
            try:
                from . import project as project_module
                repo_root = Path(project_module.load(repo_slug).repo_path or "")
                default_file = str(Path(prefill["file"]).relative_to(repo_root))
            except (ValueError, FileNotFoundError, TypeError):
                default_file = prefill["file"]
        code_file = _ask(prompt_fn, "which file (relative to the repo) hardcodes this "
                                    "feature's model, if any? (optional, blank to skip)",
                         default_file)
        f.code_file = code_file or None
        if f.code_file:
            f.code_current_model = _ask(
                prompt_fn, "exact model string hardcoded there right now",
                default_model) or None
    return f


# ── GUI (Flask form) collection ──────────────────────────────────────────────

def from_form(get_list: Callable, get: Callable) -> WizardFields:
    """`get_list(key) -> list[str]` and `get(key, default) -> str`, matching
    `request.form.getlist` / `request.form.get` — kept as plain callables so
    this has no Flask dependency and can be unit-tested directly."""
    f = WizardFields()
    f.name = get("name", "").strip()
    f.description = get("description", "").strip()
    f.system_prompt = get("system_prompt", "")

    ids = get_list("testcase_id")
    inputs = get_list("testcase_input")
    refs = get_list("testcase_reference")
    for i, tc_id in enumerate(ids):
        tc_input = inputs[i] if i < len(inputs) else ""
        tc_ref = refs[i] if i < len(refs) else ""
        if tc_id.strip() and tc_input.strip():
            f.test_cases.append({"id": tc_id.strip(), "input": tc_input,
                                 "reference": tc_ref.strip() or None})

    c_ids = get_list("rubric_id")
    c_descs = get_list("rubric_description")
    c_weights = get_list("rubric_weight")
    for i, c_id in enumerate(c_ids):
        c_desc = c_descs[i] if i < len(c_descs) else ""
        c_weight = c_weights[i] if i < len(c_weights) else ""
        if c_id.strip() and c_desc.strip():
            f.rubric.append({"id": c_id.strip(), "description": c_desc,
                             "weight": float(c_weight) if c_weight else 1.0})

    f.max_price_in = float(get("max_price_in", "") or f.max_price_in)
    f.max_price_out = float(get("max_price_out", "") or f.max_price_out)
    f.min_context = int(get("min_context", "") or f.min_context)
    f.require_json = get("require_json", "") == "on"
    f.allow_free = get("allow_free", "") == "on"
    f.tiers = get_list("tiers") or list(DEFAULT_TIERS)
    f.judge_model = get("judge_model", "").strip() or f.judge_model
    f.max_tokens = int(get("max_tokens", "") or f.max_tokens)
    f.notify_email = get("notify_email", "").strip() or None
    f.min_improvement = float(get("min_improvement", "") or f.min_improvement)
    f.endpoint_url = get("endpoint_url", "").strip() or None
    f.endpoint_method = get("endpoint_method", "").strip() or f.endpoint_method
    f.endpoint_input_field = get("endpoint_input_field", "").strip() or f.endpoint_input_field
    f.endpoint_response_field = get("endpoint_response_field", "").strip() or f.endpoint_response_field
    f.endpoint_header = get("endpoint_header", "").strip() or None
    f.endpoint_timeout_seconds = float(get("endpoint_timeout_seconds", "") or f.endpoint_timeout_seconds)
    interval = get("schedule_interval_days", "").strip()
    f.schedule_interval_days = int(interval) if interval else None
    f.code_file = get("code_file", "").strip() or None
    f.code_current_model = get("code_current_model", "").strip() or None
    calls = get("estimated_calls_per_day", "").strip()
    f.estimated_calls_per_day = int(calls) if calls else None
    f.input_structure = get("input_structure", "").strip() or None
    f.output_structure = get("output_structure", "").strip() or None
    return f
