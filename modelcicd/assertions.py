"""Deterministic checks on a candidate's answer — the half of "is this good?"
that needs no model to decide.

WHY THIS EXISTS ALONGSIDE THE JUDGE, NOT INSTEAD OF IT. A rubric judged by an
LLM is the right tool for "is this reply helpful and on-brand". It is the
wrong tool for "does the response contain an `intent` key", because that
question has an answer, and asking a model to guess at it introduces noise
into something that could have been certain. So: anything decidable in code
is decided in code, and the judge is left to score the part that genuinely
needs reading.

FOUR TYPES, DELIBERATELY. `contains`, `equals`, `regex`, `has-keys`. This is
not a metric library and is not trying to become one — no embedding
similarity, no BLEU, no model-graded assertions (that's what the rubric is).
Four checks cover the failures that actually sink an integration: a required
field went missing, a forbidden phrase came back, a format changed shape.

A MISSING FIELD FAILS, IT DOES NOT SKIP. `path` points into the response
object; if nothing is there, the assertion FAILS. This is the same rule the
price guardrails follow — an unknown is never resolved in the candidate's
favor, because that's the version of the bug that quietly promotes something
broken.

CASE-INSENSITIVE BY DEFAULT, and that is a considered choice rather than
laziness. These run against prose written by a language model, where
"Refund" versus "refund" is a difference nobody authoring the check meant to
care about — a case-sensitive default produces false alarms far more often
than it catches anything. Set `caseSensitive: true` on an assertion where the
casing genuinely is the requirement.

`required: true` IS A GATE, AND IT SAVES A JUDGE CALL. An answer that failed
a hard requirement cannot win no matter how well it reads, so `judge.py`
skips the judge call for it entirely and floors its score. That's real money
not spent on scoring the prose of an answer that's already disqualified — and
the leaderboard says exactly which check failed in place of the judge's
reasoning, so the row is never just mysteriously bad.
"""
import json
import re
from dataclasses import dataclass, field
from typing import Optional

TYPES = ("contains", "equals", "regex", "has-keys")

# The floor of the rubric's own 1-5 scale, and the top of it. A boolean
# check has to land somewhere on the same scale to be blended with judged
# criteria at all; a failed check is a 1 ("fails"), exactly as the rubric
# defines it, not a 0 — nothing else in this project scores below 1.
FAIL_SCORE = 1.0
PASS_SCORE = 5.0


class BadAssertion(ValueError):
    """An assertion that cannot mean anything — caught at config load, never
    mid-run. A typo'd type or an unparseable regex discovered thirty candidates
    into a bench has already cost the money it was supposed to save."""


@dataclass
class Assertion:
    """One deterministic check on one answer.

    `negate` comes from a `not-` prefix on the type (`not-contains`), which
    is the spelling promptfoo users will already have in their fingers.
    """
    type: str
    value: object = None
    path: Optional[str] = None          # dotted path into the response; None = whole answer
    weight: float = 1.0
    required: bool = False              # failing this disqualifies the answer
    negate: bool = False
    case_sensitive: bool = False
    _pattern: object = field(default=None, repr=False, compare=False)

    @property
    def label(self) -> str:
        """How this reads on a leaderboard row — the check, in words."""
        prefix = "not " if self.negate else ""
        where = f"{self.path} " if self.path else ""
        if self.type == "has-keys":
            keys = ", ".join(str(k) for k in (self.value or []))
            return f"{prefix}has keys: {keys}"
        return f"{where}{prefix}{self.type} {self.value!r}"


def parse(raw: list) -> list:
    """Validates and builds assertions from YAML. Raises `BadAssertion` with
    the offending entry named — this runs at config load, so the cost of a
    typo is a clear message instead of a wasted run."""
    parsed = []
    for entry in raw or []:
        if not isinstance(entry, dict):
            raise BadAssertion(f"an assertion must be a mapping, got {entry!r}.")
        declared = str(entry.get("type") or "").strip()
        negate = declared.startswith("not-")
        kind = declared[4:] if negate else declared
        if kind not in TYPES:
            raise BadAssertion(
                f"unknown assertion type {declared!r}. Known types: "
                f"{', '.join(TYPES)} (each also as not-<type>).")

        value = entry.get("value")
        if kind == "has-keys":
            if isinstance(value, str):
                value = [value]
            if not isinstance(value, list) or not value:
                raise BadAssertion(
                    "has-keys needs a 'value' listing the key(s) the response "
                    f"must contain, got {value!r}.")
            value = [str(k) for k in value]
        elif value is None or str(value) == "":
            raise BadAssertion(f"{declared} needs a 'value' to check against.")
        else:
            value = str(value)

        pattern = None
        if kind == "regex":
            flags = 0 if entry.get("caseSensitive") else re.IGNORECASE
            try:
                # COMPILED HERE, NOT AT RUN TIME. A bad pattern is a config
                # error, and this is the only place it can still be reported
                # as one.
                pattern = re.compile(value, flags)
            except re.error as exc:
                raise BadAssertion(f"{declared} has an invalid pattern {value!r}: {exc}")

        weight = float(entry.get("weight", 1.0))
        if weight < 0:
            raise BadAssertion(f"{declared} has a negative weight ({weight}).")

        parsed.append(Assertion(
            type=kind, value=value, path=entry.get("path") or None,
            weight=weight, required=bool(entry.get("required")),
            negate=negate, case_sensitive=bool(entry.get("caseSensitive")),
            _pattern=pattern))
    return parsed


def to_yaml_dicts(assertions: list) -> list:
    """Back to the YAML shape, for `wizard.to_yaml`. Only non-default fields
    are emitted, so a hand-written file stays as short as it was."""
    out = []
    for a in assertions or []:
        entry = {"type": f"not-{a.type}" if a.negate else a.type, "value": a.value}
        if a.path:
            entry["path"] = a.path
        if a.weight != 1.0:
            entry["weight"] = a.weight
        if a.required:
            entry["required"] = True
        if a.case_sensitive:
            entry["caseSensitive"] = True
        out.append(entry)
    return out


# ── reading the answer ──────────────────────────────────────────────────────

_MISSING = object()


def _dig(output, path: str):
    """A dotted lookup into the response. Returns `_MISSING` — never raises,
    and never returns a falsy stand-in that would read as "present but
    empty"; the caller has to distinguish those to fail correctly."""
    current = output
    for part in path.split("."):
        if isinstance(current, dict) and part in current:
            current = current[part]
        elif isinstance(current, list) and part.isdigit() and int(part) < len(current):
            current = current[int(part)]
        else:
            return _MISSING
    return current


def _as_text(value) -> str:
    """Whatever came back, as something a substring check can run on.

    `client.call_json` returns a PARSED DICT, not a string — JSON mode is
    requested on every candidate call — so a `contains` check with no `path`
    is really asking about the whole response object. Serializing it (rather
    than str()-ing it) means the text being searched is the same JSON a
    person reads on the run page, not Python's dict repr with its single
    quotes."""
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, ensure_ascii=False)
    except (TypeError, ValueError):
        return str(value)


def _check(a: Assertion, output) -> tuple:
    """(passed, detail) for one assertion, before negation is applied."""
    if a.type == "has-keys":
        if not isinstance(output, dict):
            return False, f"the response is a {type(output).__name__}, not an object"
        missing = [k for k in a.value if k not in output]
        if missing:
            return False, f"missing: {', '.join(missing)}"
        return True, "all present"

    target = output if a.path is None else _dig(output, a.path)
    if target is _MISSING:
        # UNKNOWN IS NEVER FAVORABLE. A path that isn't there is a failure,
        # not a check that didn't apply.
        return False, f"no {a.path!r} in the response"

    text = _as_text(target)
    if a.type == "regex":
        return bool(a._pattern.search(text)), f"pattern {'matched' if a._pattern.search(text) else 'did not match'}"

    needle = str(a.value)
    haystack = text if a.case_sensitive else text.lower()
    if not a.case_sensitive:
        needle = needle.lower()

    if a.type == "contains":
        return (needle in haystack), ("found" if needle in haystack else "not found")
    # equals — stripped, because trailing whitespace from a model is never
    # the difference someone authoring an equality check meant to catch.
    return (haystack.strip() == needle.strip()), f"got {text[:120]!r}"


def evaluate(output, assertions: list) -> dict:
    """Runs every assertion against one answer.

    Returns {results, passed, failed, total, weight, score, gateFailed,
    gateLabels} where `score` is the weighted 1-5 contribution of the
    assertions alone and `weight` is their total weight — `judge.py` blends
    those into the rubric's own weighted mean. With no assertions this
    returns zero weight and `judge.py`'s arithmetic is untouched, which is
    what keeps every existing use case scoring EXACTLY as it did before this
    module existed."""
    results = []
    for a in assertions or []:
        raw_passed, detail = _check(a, output)
        passed = (not raw_passed) if a.negate else raw_passed
        results.append({"type": f"not-{a.type}" if a.negate else a.type,
                        "label": a.label, "passed": passed, "detail": detail,
                        "weight": a.weight, "required": a.required})

    total_weight = sum(r["weight"] for r in results)
    earned = sum((PASS_SCORE if r["passed"] else FAIL_SCORE) * r["weight"] for r in results)
    gate_failures = [r["label"] for r in results if r["required"] and not r["passed"]]
    return {
        "results": results,
        "passed": sum(1 for r in results if r["passed"]),
        "failed": sum(1 for r in results if not r["passed"]),
        "total": len(results),
        "weight": total_weight,
        "score": round(earned / total_weight, 3) if total_weight else None,
        "gateFailed": bool(gate_failures),
        "gateLabels": gate_failures,
    }
