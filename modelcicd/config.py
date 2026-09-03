"""One use case, fully specified — the thing everything else in this project reads.

A "use case" is one place in someone's application where an LLM is called: a
support bot's reply generator, a summarizer, a codegen assistant. Model CICD
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
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml

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
    judge_model: str = "openai/gpt-4o"
    max_tokens: int = 1200
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

    test_cases = []
    for tc in raw["testCases"]:
        if not tc.get("id") or not tc.get("input"):
            raise ValueError(
                f"{p.name}: every test case needs an 'id' and an 'input'. "
                f"Got: {tc}")
        test_cases.append(TestCase(
            id=tc["id"], input=tc["input"], reference=tc.get("reference"),
            rubric=_criteria(tc.get("rubric")) or default_rubric))

    g = raw.get("guardrails") or {}
    guardrails = Guardrails(
        max_price_in=float(g.get("maxPriceIn", Guardrails.max_price_in)),
        max_price_out=float(g.get("maxPriceOut", Guardrails.max_price_out)),
        min_context=int(g.get("minContext", Guardrails.min_context)),
        require_json=bool(g.get("requireJson", Guardrails.require_json)),
        allow_free=bool(g.get("allowFree", Guardrails.allow_free)),
        modality=g.get("modality", Guardrails.modality),
        tiers=g.get("tiers") or Guardrails.tiers)

    n = raw.get("notify") or {}
    notify = Notify(email=n.get("email") or os.environ.get("MODELCICD_NOTIFY_EMAIL"),
                    min_improvement=float(n.get("minImprovement",
                                                Notify.min_improvement)))

    judge = raw.get("judgeModel") or "openai/gpt-4o"
    return UseCase(
        name=raw["useCase"], description=raw.get("description", ""),
        system_prompt=raw["systemPrompt"], test_cases=test_cases,
        rubric=default_rubric, guardrails=guardrails, notify=notify,
        judge_model=judge, max_tokens=int(raw.get("maxTokens", 1200)), path=p)


def describe(uc: UseCase) -> str:
    return "\n".join([
        f"use case      {uc.name}",
        f"  {uc.description}" if uc.description else "",
        f"  test cases    {len(uc.test_cases)}",
        f"  judge         {uc.judge_model}",
        f"  price ceiling in <= ${uc.guardrails.max_price_in:.2f}/M, "
        f"out <= ${uc.guardrails.max_price_out:.2f}/M",
        f"  min context   {uc.guardrails.min_context:,}",
        f"  tiers         {', '.join(uc.guardrails.tiers)}",
        f"  notify        {uc.notify.email or '(no email configured)'} "
        f"— on a win of at least {uc.notify.min_improvement}",
    ]).replace("\n\n", "\n")
