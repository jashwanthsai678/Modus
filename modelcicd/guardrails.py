"""What may be spent money on, decided before any is.

FREE AND FIRST. Most of a several-hundred-model catalogue cannot do the job at
all — no JSON mode, not enough context, priced past what the use case's
guardrails allow. Filtering costs nothing; discovering the same thing by
generating from each one is one paid call per model that never needed to
happen.

EVERY REJECTION IS NAMED. "Why is this model not in the shortlist" has to be
answerable without reading the code, so each excluded model keeps the rule that
excluded it and the value that failed — and the tally counts by RULE, not by the
price value that happened to fail it, or a reader gets forty rows all reading
"too expensive" at forty different numbers.
"""
from typing import Optional

from .config import Guardrails


def _cheapest(model: dict, field: str) -> Optional[float]:
    values = [h.get(field) for h in (model.get("hosts") or [])
              if h.get(field) is not None]
    return min(values) if values else None


def check(model: dict, g: Guardrails) -> list:
    """Every reason this model is excluded, as {"rule", "detail"}. Empty passes."""
    fails = []
    price_in = _cheapest(model, "price_in")
    price_out = _cheapest(model, "price_out")

    if price_in is None or price_out is None:
        fails.append({"rule": "price unknown",
                      "detail": "no usable price on any host — excluded rather "
                                "than assumed free"})
    else:
        if price_in > g.max_price_in:
            fails.append({"rule": "input price too high",
                          "detail": f"${price_in:.2f}/M vs ${g.max_price_in:.2f}"})
        if price_out > g.max_price_out:
            fails.append({"rule": "output price too high",
                          "detail": f"${price_out:.2f}/M vs ${g.max_price_out:.2f}"})
        if not g.allow_free and price_in == 0 and price_out == 0:
            fails.append({"rule": "free tier excluded", "detail": "this use case excludes it"})

    context = model.get("max_context")
    if context is None:
        fails.append({"rule": "context unknown", "detail": "no host declares one"})
    elif context < g.min_context:
        fails.append({"rule": "context too small",
                      "detail": f"{context:,} vs {g.min_context:,} needed"})

    if g.require_json and not model.get("json_mode"):
        fails.append({"rule": "no JSON mode",
                      "detail": "the sandbox scores structured output; a model "
                                "without JSON mode cannot be judged, only guessed at"})

    if g.modality == "text":
        host = (model.get("hosts") or [{}])[0]
        takes = [m.lower() for m in host.get("input_modalities") or []]
        gives = [m.lower() for m in host.get("output_modalities") or []]
        if takes and "text" not in takes:
            fails.append({"rule": "wrong modality", "detail": f"input {takes} excludes text"})
        # EXCLUSIVE, not merely inclusive. A model declaring text+audio output
        # (a voice/music model that also emits transcripts) satisfies "includes
        # text" while being unable to write the thing being judged.
        extra = [m for m in gives if m != "text"]
        if gives and extra:
            fails.append({"rule": "wrong modality",
                          "detail": f"outputs {gives} — not a text-only writer"})

    for host in model.get("hosts") or []:
        vendor = str(host.get("id") or "").split("/", 1)[0].lower()
        if vendor == host.get("provider"):
            fails.append({"rule": "routed meta-model",
                          "detail": f"{host['id']} routes to whichever backend "
                                    f"answers — not a repeatable candidate"})
            break

    return fails


def apply(catalogue: dict, g: Guardrails) -> dict:
    """Split the catalogue into what may be spent on and what may not."""
    passed, rejected, by_rule = [], [], {}
    for model in (catalogue.get("models") or {}).values():
        fails = check(model, g)
        if fails:
            rejected.append({**model, "rejectedBecause": fails})
            by_rule[fails[0]["rule"]] = by_rule.get(fails[0]["rule"], 0) + 1
        else:
            passed.append(model)
    passed.sort(key=lambda m: (m.get("cheapest_out") if m.get("cheapest_out")
                               is not None else 1e9, m["key"]))
    return {"passed": passed, "rejected": rejected,
            "counts": {"considered": len(catalogue.get("models") or {}),
                       "passed": len(passed), "rejected": len(rejected),
                       "byRule": dict(sorted(by_rule.items(), key=lambda kv: -kv[1]))}}


def tier(model: dict) -> str:
    """Which price band this model competes in. Ranking is always PER TIER — a
    single global list always concludes the expensive one won, which answers
    nobody's actual question of 'what's the best I can afford'."""
    out = model.get("cheapest_out")
    if out is None:
        return "unknown"
    if out == 0:
        return "free"
    if out <= 0.50:
        return "paid-low"
    if out <= 3.00:
        return "paid-mid"
    return "paid-high"


def within_tiers(models: list, tiers: list) -> list:
    if not tiers:
        return models
    return [m for m in models if tier(m) in tiers]


def summarise(result: dict) -> str:
    c = result["counts"]
    lines = [f"considered {c['considered']} model(s) — "
             f"PASSED {c['passed']}   REJECTED {c['rejected']}", ""]
    for rule, n in c["byRule"].items():
        lines.append(f"   {n:>4}  {rule}")
    return "\n".join(lines)
