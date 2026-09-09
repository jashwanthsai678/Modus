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
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import yaml

from . import assertions as assertions_module
from .config import DEFAULT_JUDGE_MODEL

DEFAULT_TIERS = ["free", "paid-low", "paid-mid"]


@dataclass
class WizardFields:
    name: str = ""
    description: str = ""
    system_prompt: str = ""
    test_cases: list = field(default_factory=list)   # [{"id","input","reference"}]
    rubric: list = field(default_factory=list)        # [{"id","description","weight"}]
    # Deterministic checks shared by every test case — the YAML shape, not
    # parsed `Assertion` objects, since this module's whole job is producing
    # the file `config.load` then validates. [{"type","value","path",
    # "weight","required","caseSensitive"}]
    assertions: list = field(default_factory=list)
    max_price_in: float = 0.50
    max_price_out: float = 3.00
    min_context: int = 32_000
    require_json: bool = True
    allow_free: bool = True
    tiers: list = field(default_factory=lambda: list(DEFAULT_TIERS))
    judge_model: str = DEFAULT_JUDGE_MODEL
    generation_model: Optional[str] = None   # drafts a test case's reference answer; None = uses judge_model
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
    f.assertions = [dict(a) for a in (defaults.get("assertions") or [])]
    f.max_price_in = float(defaults.get("maxPriceIn", f.max_price_in))
    f.max_price_out = float(defaults.get("maxPriceOut", f.max_price_out))
    f.min_context = int(defaults.get("minContext", f.min_context))
    f.require_json = bool(defaults.get("requireJson", f.require_json))
    f.allow_free = bool(defaults.get("allowFree", f.allow_free))
    f.tiers = list(defaults.get("tiers") or f.tiers)
    f.judge_model = defaults.get("judgeModel") or f.judge_model
    f.generation_model = defaults.get("generationModel") or None
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
    # Run through the real parser rather than re-checking by hand, so the
    # wizard and `config.load` can never disagree about what's valid — the
    # thing that would let a form write a file that then fails to load.
    errors += validate_assertions(_clean_assertions(f.assertions))
    return errors


def _headers(f: WizardFields) -> dict:
    if not f.endpoint_header or ":" not in f.endpoint_header:
        return {}
    name, _, value = f.endpoint_header.partition(":")
    return {name.strip(): value.strip()}


def validate_assertions(rows: list) -> list:
    """Errors in already-cleaned assertion rows, via the REAL parser — the
    same one `config.load` uses — so a form can never save something the
    loader would then refuse."""
    try:
        assertions_module.parse(rows)
    except assertions_module.BadAssertion as exc:
        return [str(exc)]
    return []


def assertions_from_form(getlist: Callable) -> list:
    """Reads assertion rows out of a submitted form — used by BOTH the
    feature wizard and the project-defaults page, so the two can't disagree
    about what a row means. Blank rows are dropped by `_clean_assertions`."""
    types = getlist("assert_type")
    values = getlist("assert_value")
    paths = getlist("assert_path")
    weights = getlist("assert_weight")
    # A CHECKBOX ONLY SUBMITS WHEN TICKED, so its list can't be positionally
    # zipped with the others — an unticked row would silently shift every
    # later row's "required" flag onto the wrong assertion. Named per-index
    # instead, and read back the same way.
    required = set(getlist("assert_required"))
    sensitive = set(getlist("assert_case_sensitive"))
    rows = []
    for i, kind in enumerate(types):
        rows.append({
            "type": kind,
            "value": values[i] if i < len(values) else "",
            "path": paths[i] if i < len(paths) else "",
            "weight": weights[i] if i < len(weights) else "1.0",
            "required": str(i) in required,
            "caseSensitive": str(i) in sensitive,
        })
    return _clean_assertions(rows)


def _clean_assertions(raw: list) -> list:
    """Drops blank rows (a form always submits more slots than were filled)
    and normalizes the rest to the YAML shape."""
    out = []
    for a in raw or []:
        kind = str(a.get("type") or "").strip()
        value = a.get("value")
        if not kind:
            continue
        if kind.endswith("has-keys"):
            keys = value if isinstance(value, list) else [
                k.strip() for k in str(value or "").split(",") if k.strip()]
            if not keys:
                continue
            value = keys
        else:
            value = str(value or "").strip()
            if not value:
                continue
        entry = {"type": kind, "value": value}
        if (a.get("path") or "").strip():
            entry["path"] = a["path"].strip()
        weight = float(a.get("weight") or 1.0)
        if weight != 1.0:
            entry["weight"] = weight
        if a.get("required"):
            entry["required"] = True
        if a.get("caseSensitive"):
            entry["caseSensitive"] = True
        out.append(entry)
    return out


def _test_case_doc(tc: dict) -> dict:
    """One test case's YAML entry.

    CARRIES ITS PER-TEST-CASE OVERRIDES THROUGH. `rubric` and `assertions`
    on a single test case aren't editable in any form — they're an advanced,
    hand-written thing — but they MUST survive a round trip, because the
    edit form reads a file into `WizardFields` and writes it back out. Drop
    them here and editing a feature's price ceiling through the UI would
    silently delete a rubric override someone wrote by hand. That is the
    exact class of quiet data loss this project keeps having to design
    against, and an edit form is where it would finally bite."""
    entry = {"id": tc["id"].strip(), "input": tc["input"]}
    if tc.get("reference"):
        entry["reference"] = tc["reference"]
    if tc.get("rubric"):
        entry["rubric"] = [
            {"id": c["id"], "description": c["description"],
             "weight": float(c.get("weight") or 1.0)}
            for c in tc["rubric"]]
    if tc.get("assertions"):
        entry["assertions"] = _clean_assertions(tc["assertions"])
    return entry


def to_yaml(f: WizardFields) -> str:
    """Renders the exact schema `config.load` expects. Blank/incomplete rows
    in test_cases and rubric are dropped, not written as broken entries."""
    doc = {
        "useCase": f.name.strip(),
        "description": f.description or "",
        "systemPrompt": f.system_prompt,
        "testCases": [
            _test_case_doc(tc) for tc in f.test_cases
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
    # Written only when there ARE any, so nothing changes in a file for a
    # feature that configured none — and validated on the way out, so the
    # wizard can never produce a YAML that `config.load` will then refuse.
    checks = _clean_assertions(f.assertions)
    if checks:
        doc["assertions"] = checks
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


# ── Reading an existing feature back, for editing ───────────────────────────

def from_yaml(path) -> WizardFields:
    """The inverse of `to_yaml`: a saved use_case.yaml back into editable
    fields, so the same form that creates a feature can edit one.

    READS THE RAW YAML, NOT `config.load`. That loader is deliberately
    lossy in a way that would be destructive here: it resolves defaults
    (giving every test case the use-case rubric when it has no override of
    its own, and merging shared assertions into each test case's list), so
    round-tripping through it would rewrite every test case with an
    explicit copy of the shared rubric — turning inherited values into
    pinned ones, and quietly breaking the inheritance the file was written
    to use. The raw document is the only faithful source for "what did the
    author actually write".

    `config.load` is still the validator — `dashboard`'s edit route runs it
    on the result before writing anything — this function just doesn't use
    it as a reader.
    """
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    f = WizardFields()
    f.name = raw.get("useCase", "")
    f.description = raw.get("description", "") or ""
    f.system_prompt = raw.get("systemPrompt", "")

    for tc in raw.get("testCases") or []:
        entry = {"id": tc.get("id", ""), "input": tc.get("input", ""),
                 "reference": tc.get("reference")}
        # Preserved verbatim, never surfaced in the form — see `_test_case_doc`.
        if tc.get("rubric"):
            entry["rubric"] = tc["rubric"]
        if tc.get("assertions"):
            entry["assertions"] = tc["assertions"]
        f.test_cases.append(entry)

    f.rubric = [dict(c) for c in (raw.get("rubric") or [])]
    f.assertions = [dict(a) for a in (raw.get("assertions") or [])]

    g = raw.get("guardrails") or {}
    f.max_price_in = float(g.get("maxPriceIn", f.max_price_in))
    f.max_price_out = float(g.get("maxPriceOut", f.max_price_out))
    f.min_context = int(g.get("minContext", f.min_context))
    f.require_json = bool(g.get("requireJson", f.require_json))
    f.allow_free = bool(g.get("allowFree", f.allow_free))
    f.tiers = list(g.get("tiers") or f.tiers)

    f.judge_model = raw.get("judgeModel") or f.judge_model
    f.max_tokens = int(raw.get("maxTokens", f.max_tokens))

    n = raw.get("notify") or {}
    f.notify_email = n.get("email")
    f.min_improvement = float(n.get("minImprovement", f.min_improvement))

    e = raw.get("endpoint") or {}
    if e.get("url"):
        f.endpoint_url = e["url"]
        f.endpoint_method = e.get("method", "POST")
        f.endpoint_input_field = e.get("inputField", "input")
        f.endpoint_response_field = e.get("responseField", "output")
        f.endpoint_timeout_seconds = float(e.get("timeoutSeconds", 30.0))
        # One "Name: value" line, matching what `_headers` parses back out.
        headers = e.get("headers") or {}
        if headers:
            name, value = next(iter(headers.items()))
            f.endpoint_header = f"{name}: {value}"

    interval = (raw.get("schedule") or {}).get("intervalDays")
    f.schedule_interval_days = int(interval) if interval else None

    ct = raw.get("codeTarget") or {}
    if ct.get("file"):
        f.code_file = ct["file"]
        f.code_current_model = ct.get("currentModel")

    calls = (raw.get("usage") or {}).get("callsPerDay")
    f.estimated_calls_per_day = int(calls) if calls else None
    f.input_structure = raw.get("inputStructure")
    f.output_structure = raw.get("outputStructure")
    return f


def carry_over_overrides(new: WizardFields, existing: WizardFields) -> None:
    """Copies per-test-case rubric/assertion overrides from the on-disk
    version onto freshly submitted fields, matched by test case id.

    WHY THIS IS NEEDED AT ALL. `from_form` rebuilds `test_cases` from the
    form's inputs, and the form has no inputs for a single test case's own
    rubric or assertions — they're an advanced, hand-written thing. Without
    this, editing a feature's price ceiling through the UI would submit test
    cases stripped of those overrides and write them away. Matched by id
    rather than position, so reordering or inserting a test case doesn't
    move someone's override onto the wrong one; an override whose test case
    was deleted goes with it, which is correct.
    """
    by_id = {tc.get("id"): tc for tc in existing.test_cases if tc.get("id")}
    for tc in new.test_cases:
        old = by_id.get(tc.get("id"))
        if not old:
            continue
        for key in ("rubric", "assertions"):
            if old.get(key) and not tc.get(key):
                tc[key] = old[key]


def write_yaml(path, f: WizardFields) -> Path:
    """Renders and writes one use_case.yaml ATOMICALLY — temp file then
    `os.replace` — the same discipline `state.py` uses. A crash or a
    concurrent reader can never observe a half-written config, which for an
    EDIT (as opposed to a create) would mean a corrupted file where a
    working feature used to be."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(f"{p.suffix}.{os.getpid()}.tmp")
    tmp.write_text(to_yaml(f), encoding="utf-8")
    os.replace(tmp, p)
    return p


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

    print("\nDeterministic checks (optional) — things decidable in code, run free "
          "before any judge call. Empty type to stop.")
    print(f"  types: {', '.join(assertions_module.TYPES)} (each also as not-<type>)")
    while True:
        kind = _ask(prompt_fn, "  type (blank to stop)")
        if not kind:
            break
        value = _ask(prompt_fn, "  value (for has-keys: comma-separated key names)")
        path = _ask(prompt_fn, "  path into the JSON response (blank = the whole answer)")
        weight = _ask(prompt_fn, "  weight", "1.0")
        required = _ask(prompt_fn, "  required? failing it disqualifies the answer (y/N)").lower()
        sensitive = _ask(prompt_fn, "  case-sensitive? (y/N)").lower()
        f.assertions.append({
            "type": kind, "value": value, "path": path or None,
            "weight": float(weight or 1.0),
            "required": required in ("y", "yes"),
            "caseSensitive": sensitive in ("y", "yes")})

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

    f.assertions = assertions_from_form(get_list)

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
