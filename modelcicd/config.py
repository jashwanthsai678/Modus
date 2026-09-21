"""One use case, fully specified — the thing everything else in this project reads.

A "use case" is one place in someone's application where an LLM is called: a
support bot's reply generator, a summarizer, a codegen assistant. MoCICD
benchmarks candidates AGAINST that specific job, using THAT job's own test cases
and THAT job's own quality bar — never a generic leaderboard, because a model
that tops MMLU can still write unusable output for a task nobody else has.

ONE YAML FILE IS THE WHOLE ONBOARDING. Everything the loop needs — the system
prompt, the test cases, the rubric, the price ceilings, who gets emailed — lives
in one file a person can read top to bottom. There is no database and no web
form in front of it: the config IS the product's onboarding, and keeping it a
flat file is what makes this usable as an open-source tool rather than a service
someone has to trust with a signup flow.

WHAT THIS FILE DELIBERATELY DOES NOT HOLD. API keys. They come from the
environment (`OPENROUTER_API_KEY`, `SMTP_*`), never from a file that gets
committed to a repo — a config checked into version control with a key in it is
a leaked key, and open-source users will absolutely commit their config.
"""
import hashlib
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml

from . import assertions as assertions_module

# The judge every use case falls back to when its YAML doesn't name one.
# ONE definition, referenced by `wizard.py` too — it was previously spelled
# out as a literal in three separate places, which is exactly how a default
# drifts apart between the CLI, the dashboard, and the loader.
#
# A FREE MODEL BY DEFAULT, ON PURPOSE. A judge runs once per candidate per
# test case, so it's the single biggest per-run cost after the candidates
# themselves — an expensive default is a bill someone didn't ask for on
# their first run. This one is verified to return valid JSON through
# `client.call_json`'s json_object path, which is the actual requirement.
# Override per use case with `judgeModel:` in its YAML.
DEFAULT_JUDGE_MODEL = "nvidia/nemotron-3-super-120b-a12b:free"

# ── The rubric ────────────────────────────────────────────────────────────

@dataclass
class Criterion:
    """One thing the judge scores, 1-5, with a weight.

    THE RUBRIC IS BRING-YOUR-OWN, AND THIS IS THE CENTRAL DESIGN CALL. A single
    generic rubric ("is this a good response?") cannot judge a support bot and a
    codegen assistant with the same words — "correctness" means something
    different to each. So the rubric is authored per use case, by whoever owns
    the use case, in their own language. The judge is handed these descriptions
    verbatim; it never invents its own criteria.
    """
    id: str
    description: str
    weight: float = 1.0


@dataclass
class TestCase:
    """One input the model must handle, and what a good answer looks like.

    `reference` is optional and deliberately so: some tasks have a gold answer
    (translate this sentence) and some do not (write an engaging opener) — and a
    judge told to compare against a reference that does not exist will invent
    one to satisfy the instruction, which is worse than being told there is
    none.
    """
    id: str
    input: str
    reference: Optional[str] = None
    # Per-test-case overrides of the use case's rubric — most test cases share
    # the same rubric, so this is empty far more often than not.
    rubric: list = field(default_factory=list)
    # Deterministic checks on this answer (see `assertions.py`).
    #
    # THESE ADD TO THE USE CASE'S, THEY DO NOT REPLACE THEM — deliberately
    # the OPPOSITE of how `rubric` above inherits, and the difference is
    # not an inconsistency. Two rubrics can't coexist: they're competing
    # scales for the same judgement, so a test case's own rubric has to
    # replace the default. Two assertions are independent facts about one
    # answer, and combine fine. Concretely: a use case declaring "every
    # answer must have a `team` key" and a test case adding "and this one
    # must say billing" means both, which is plainly what someone writing
    # that meant. Under replace-semantics the shape check would silently
    # vanish from exactly the test cases whose answers were pinned down
    # most precisely — a required check disappearing without a word, which
    # is the class of bug this project exists to not have.
    assertions: list = field(default_factory=list)


# ── Guardrails — what may be spent on ────────────────────────────────────────

@dataclass
class Guardrails:
    """What a candidate must clear before it is ever generated from.

    UNKNOWN IS NOT FREE. A model whose price a platform does not publish is
    EXCLUDED by the cost rule, never assumed to be $0 — the one class of bug in
    a cost guardrail that actually spends someone's money.
    """
    max_price_in: float = 0.50      # $ per 1M input tokens
    max_price_out: float = 3.00     # $ per 1M output tokens
    min_context: int = 32_000
    require_json: bool = True
    allow_free: bool = True
    modality: str = "text"
    tiers: list = field(default_factory=lambda: ["free", "paid-low", "paid-mid"])


# ── Notification + promotion ─────────────────────────────────────────────────

@dataclass
class Notify:
    """Who hears about a new candidate, and how big the win has to be to bother them."""
    email: Optional[str] = None
    # A candidate must beat the currently-approved model by at least this much
    # (on the 1-5 scale) before anyone is emailed. Below it, two models are
    # statistically the same at one sample per test case, and paging a human
    # over noise is how they learn to ignore the emails.
    min_improvement: float = 0.20


# ── Live endpoint — an optional baseline to compare candidates against ──────

@dataclass
class Endpoint:
    """A use case's own live implementation (hosted or localhost — just a URL).

    CAPTURES A BASELINE, NEVER ROUTES A CANDIDATE. The bench calls this once
    per test case, with the same input every candidate model gets, and judges
    the answer through the same rubric — so the leaderboard shows "what you
    have now" next to every candidate. It is never called on behalf of a
    candidate model; OpenRouter is still the only thing candidates run through.
    """
    url: str
    method: str = "POST"
    input_field: str = "input"
    response_field: str = "output"
    headers: dict = field(default_factory=dict)
    timeout_seconds: float = 30.0


# ── Code target — where an approved model gets written back to, on request ──

@dataclass
class CodeTarget:
    """The file, inside a project's connected repo, that hardcodes this
    feature's model — and the model string modelcicd believes is written
    there right now.

    `current_model` IS NOT A ONE-TIME SEED. `code_patch.update_tracked_model`
    rewrites it after every successful patch, so this field is always this
    use case's own record of "what's currently in the code" — the exact
    string the next patch will search for and replace. Nothing here EVER
    changes what `resolve()` returns; that is still `state.approve()` alone.
    """
    file: str                          # relative to the project's repo_path
    current_model: Optional[str] = None


@dataclass
class UseCase:
    """Everything the loop needs for one place an LLM is called."""
    name: str
    description: str
    system_prompt: str
    test_cases: list                 # list[TestCase]
    rubric: list                     # list[Criterion] — the default, per test case may override
    guardrails: Guardrails
    notify: Notify
    # list[Assertion] — the default, per test case may override. See `assertions.py`.
    assertions: list = field(default_factory=list)
    judge_model: str = DEFAULT_JUDGE_MODEL
    max_tokens: int = 1200
    endpoint: Optional[Endpoint] = None       # live baseline to compare candidates against
    schedule_interval_days: Optional[int] = None  # re-run automatically on this cadence
    code_target: Optional[CodeTarget] = None  # where to (optionally) patch an approved model back to
    estimated_calls_per_day: Optional[int] = None  # rough usage, paired with any observed rate limiting
    input_structure: Optional[str] = None   # what the code actually sends, from a scan — display only
    output_structure: Optional[str] = None  # what the code expects back, from a scan — display only
    path: Optional[Path] = None      # where this was loaded from, for error messages


def _criteria(raw: list) -> list:
    return [Criterion(id=c["id"], description=c["description"],
                      weight=float(c.get("weight", 1.0))) for c in (raw or [])]


def load(path) -> UseCase:
    """Read and validate one use case. Fails LOUDLY and SPECIFICALLY.

    A config error caught here, with the field name and why, costs nothing. The
    same error caught three stages later — the judge silently scoring against an
    empty rubric — costs a run's worth of model calls to produce a leaderboard
    that means nothing.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(
            f"no use-case file at {p}. Start from examples/prep_material/ and "
            f"copy it, or run `python -m modelcicd.cli init` to write a template.")
    raw = yaml.safe_load(p.read_text(encoding="utf-8")) or {}

    missing = [k for k in ("useCase", "systemPrompt", "testCases") if not raw.get(k)]
    if missing:
        raise ValueError(
            f"{p.name} is missing required field(s): {', '.join(missing)}. "
            f"A use case needs at least a name, a system prompt, and one test case.")

    default_rubric = _criteria(raw.get("rubric"))
    if not default_rubric and not any(tc.get("rubric") for tc in raw["testCases"]):
        raise ValueError(
            f"{p.name} defines no rubric — at the top level or on any test case. "
            f"The judge has nothing to score against, which is not a permissive "
            f"default, it is an unanswerable question.")

    # Parsed HERE so a typo'd type or an unparseable regex is a config error
    # with a file name on it, not a surprise thirty candidates into a bench.
    try:
        default_assertions = assertions_module.parse(raw.get("assertions"))
    except assertions_module.BadAssertion as exc:
        raise ValueError(f"{p.name}: {exc}") from exc

    test_cases = []
    for tc in raw["testCases"]:
        if not tc.get("id") or not tc.get("input"):
            raise ValueError(
                f"{p.name}: every test case needs an 'id' and an 'input'. "
                f"Got: {tc}")
        try:
            own_assertions = assertions_module.parse(tc.get("assertions"))
        except assertions_module.BadAssertion as exc:
            raise ValueError(f"{p.name}, test case {tc['id']!r}: {exc}") from exc
        test_cases.append(TestCase(
            id=tc["id"], input=tc["input"], reference=tc.get("reference"),
            rubric=_criteria(tc.get("rubric")) or default_rubric,
            # ADDED to the use case's, not replacing them — see TestCase.
            assertions=list(default_assertions) + own_assertions))

    g = raw.get("guardrails") or {}
    guardrails = Guardrails(
        max_price_in=float(g.get("maxPriceIn", Guardrails.max_price_in)),
        max_price_out=float(g.get("maxPriceOut", Guardrails.max_price_out)),
        min_context=int(g.get("minContext", Guardrails.min_context)),
        require_json=bool(g.get("requireJson", Guardrails.require_json)),
        allow_free=bool(g.get("allowFree", Guardrails.allow_free)),
        modality=g.get("modality", Guardrails.modality),
        # `Guardrails().tiers`, NOT `Guardrails.tiers`. `tiers` is the one
        # field here declared with `field(default_factory=...)` — a mutable
        # default — which means it exists on an INSTANCE but not on the
        # class, so the class-attribute spelling every neighbouring line
        # uses raises AttributeError for this one. It only ever fired on a
        # YAML that omits `guardrails.tiers` entirely, which the template
        # always includes, so a hand-written minimal file was the one thing
        # that couldn't load.
        tiers=g.get("tiers") or list(Guardrails().tiers))

    n = raw.get("notify") or {}
    notify = Notify(email=n.get("email") or os.environ.get("MODELCICD_NOTIFY_EMAIL"),
                    min_improvement=float(n.get("minImprovement",
                                                Notify.min_improvement)))

    judge = raw.get("judgeModel") or DEFAULT_JUDGE_MODEL

    endpoint = None
    e = raw.get("endpoint") or {}
    if e.get("url"):
        endpoint = Endpoint(
            url=e["url"], method=(e.get("method") or "POST").upper(),
            input_field=e.get("inputField", "input"),
            response_field=e.get("responseField", "output"),
            headers=dict(e.get("headers") or {}),
            timeout_seconds=float(e.get("timeoutSeconds", 30.0)))

    sched = raw.get("schedule") or {}
    interval = sched.get("intervalDays")

    code_target = None
    ct = raw.get("codeTarget") or {}
    if ct.get("file"):
        code_target = CodeTarget(file=ct["file"], current_model=ct.get("currentModel"))

    usage = raw.get("usage") or {}
    calls_per_day = usage.get("callsPerDay")

    return UseCase(
        name=raw["useCase"], description=raw.get("description", ""),
        system_prompt=raw["systemPrompt"], test_cases=test_cases,
        rubric=default_rubric, guardrails=guardrails, notify=notify,
        assertions=default_assertions,
        judge_model=judge, max_tokens=int(raw.get("maxTokens", 1200)),
        endpoint=endpoint,
        schedule_interval_days=int(interval) if interval else None,
        code_target=code_target,
        estimated_calls_per_day=int(calls_per_day) if calls_per_day else None,
        input_structure=raw.get("inputStructure"), output_structure=raw.get("outputStructure"),
        path=p)


# ── What was measured — so two scores can't be silently compared ──────────
#
# THE PROBLEM THIS SOLVES, OBSERVED IN THIS PROJECT'S OWN DATA. `agent_router`
# scored 2.25, then 4.45. Nothing about the models changed between those two
# runs; deterministic checks were added to the use case, so the second number
# was produced by a different measuring stick. Both points sit on the same
# trend line, and a trend line is exactly what a person reads as "this got
# better". Without a fingerprint the history silently lies, and it lies more
# every time someone improves their rubric — which is a thing this tool
# actively encourages them to do.
#
# COMPONENTS, NOT ONE OPAQUE HASH. Knowing "something changed" sends someone
# diffing YAML by hand. Knowing "the rubric changed but the test cases
# didn't" is immediately actionable, and it's the difference between a
# warning people read and a warning people learn to click past.

# BUMP THIS WHENEVER THE HASHING SCHEME CHANGES, and note what happens when
# you do: every stored fingerprint becomes UNCOMPARABLE, not "changed". This
# was found the hard way within an hour of shipping the first version —
# recomposing which component covers what (see `measurement`) silently made
# every existing run look like the rubric, checks and test cases had all
# changed at once. A false "not comparable" on every feature is how a warning
# stops being read, so a version mismatch now makes NO CLAIM, exactly like an
# absent fingerprint.
MEASUREMENT_VERSION = 2


def _hash(payload) -> str:
    """A short, stable digest. Twelve hex characters — long enough that a
    collision isn't a practical concern for one use case's ~50-run history,
    short enough to print on a leaderboard next to a score."""
    blob = json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:12]


def measurement(uc: UseCase) -> dict:
    """A fingerprint of everything that decides what a score MEANS.

    IN HERE: the system prompt (what the model was asked), the test cases
    (what it was asked about), the rubric and the deterministic checks (how
    the answer was scored), and the judge model (who scored it — a different
    judge is a different scale, even against an identical rubric).

    DELIBERATELY NOT IN HERE: the guardrails, the tiers, the price ceilings,
    the notify threshold, the schedule. Those decide WHICH CANDIDATES get
    measured, not how. Widening a price ceiling adds rows to a leaderboard;
    it doesn't make the existing rows mean something different, so folding
    it in would cry wolf on every routine edit and teach people to ignore
    the warning that matters.
    """
    # ONE CONCEPT PER COMPONENT, which took a second attempt to get right.
    # Folding each test case's rubric into the `testCases` component seemed
    # natural — that's where an override is authored — but `config.load`
    # gives every test case the use-case rubric when it has no override of
    # its own. So editing the shared rubric moved BOTH components and the
    # report read "the test cases and the rubric changed" for an edit that
    # touched one line of rubric. True, and useless.
    #
    # So: `testCases` is WHAT WAS ASKED (id, input, reference). `rubric` is
    # HOW IT WAS JUDGED, shared and per-test-case together. `checks` is
    # WHAT WAS VERIFIED, likewise. Each edit now moves exactly one.
    def criteria(items):
        return [[c.id, c.description, c.weight] for c in items]

    def checks_of(items):
        return [[a.type, a.value, a.path, a.weight, a.required, a.negate,
                 a.case_sensitive] for a in items]

    rubric = _hash([criteria(uc.rubric),
                    [[tc.id, criteria(tc.rubric)] for tc in uc.test_cases]])
    checks = _hash([checks_of(uc.assertions),
                    [[tc.id, checks_of(tc.assertions)] for tc in uc.test_cases]])
    cases = _hash([[tc.id, tc.input, tc.reference] for tc in uc.test_cases])
    prompt = _hash(uc.system_prompt)
    parts = {"rubric": rubric, "checks": checks, "testCases": cases,
             "prompt": prompt, "judge": _hash(uc.judge_model)}
    return {**parts, "version": MEASUREMENT_VERSION,
            "combined": _hash({**parts, "version": MEASUREMENT_VERSION}),
            "testCaseCount": len(uc.test_cases)}


# Which component labels read as what, when a run is compared to the one
# before it. Kept next to `measurement` so a new component can't be added
# without a name to report it under.
MEASUREMENT_LABELS = {
    "prompt": "the system prompt",
    "testCases": "the test cases",
    "rubric": "the rubric",
    "checks": "the deterministic checks",
    "judge": "the judge model",
}


def comparable(current: Optional[dict], previous: Optional[dict]) -> Optional[bool]:
    """Are two runs' scores on the same scale? True / False / None.

    THREE-VALUED ON PURPOSE, AND THE ONE PLACE THAT DECIDES IT. `None` means
    unknown — either side missing a fingerprint, or the two computed by
    different scheme versions. Callers must not collapse `None` into either
    answer: a drawn line asserts "the measurement didn't change" and a gap
    asserts "it did", so an unknown gets a third rendering (dashed) and no
    warning text at all.

    Every consumer reads this rather than comparing hashes itself, so the
    trend line and the run-page warning cannot end up disagreeing about
    whether two runs are comparable."""
    if not current or not previous:
        return None
    if current.get("version") != previous.get("version"):
        return None
    return current.get("combined") == previous.get("combined")


def measurement_changes(current: Optional[dict], previous: Optional[dict]) -> list:
    """Which parts of the measuring stick moved between two runs, in words.

    Returns [] when they match AND when comparability is unknown — saying
    nothing is honest there, where either "unchanged" or a list of changes
    would be invented. Also naturally ignores `testCaseCount`, which
    `testCases` already covers."""
    if comparable(current, previous) is not False:
        return []
    return [label for key, label in MEASUREMENT_LABELS.items()
            if current.get(key) and previous.get(key) and current[key] != previous[key]]


def describe(uc: UseCase) -> str:
    return "\n".join([
        f"use case      {uc.name}",
        f"  {uc.description}" if uc.description else "",
        f"  test cases    {len(uc.test_cases)}",
        f"  checks        {len(uc.assertions)} deterministic "
        f"({sum(1 for a in uc.assertions if a.required)} required)"
        if uc.assertions else "",
        f"  judge         {uc.judge_model}",
        f"  price ceiling in <= ${uc.guardrails.max_price_in:.2f}/M, "
        f"out <= ${uc.guardrails.max_price_out:.2f}/M",
        f"  min context   {uc.guardrails.min_context:,}",
        f"  tiers         {', '.join(uc.guardrails.tiers)}",
        f"  notify        {uc.notify.email or '(no email configured)'} "
        f"— on a win of at least {uc.notify.min_improvement}",
        f"  endpoint      {uc.endpoint.url}" if uc.endpoint else "",
        f"  schedule      every {uc.schedule_interval_days} day(s)"
        if uc.schedule_interval_days else "",
        f"  code target   {uc.code_target.file}" if uc.code_target else "",
        f"  usage est.    ~{uc.estimated_calls_per_day:,} calls/day"
        if uc.estimated_calls_per_day else "",
    ]).replace("\n\n", "\n")
