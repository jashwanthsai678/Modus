"""The leaderboard, per tier, with its own limits stated on it.

RANKED WITHIN TIERS. A global list always concludes the expensive model won,
which answers nobody's actual question of "what's the best I can afford".

TIES, NOT WINNERS. Every score here is ONE sample per test case. Two candidates
within NOISE of each other are a tie at that sample size, not a ranking — naming
a single winner from one run each is exactly the overclaim this tool is cheap
enough to be tempted into.
"""
from typing import Optional

from . import catalogue as catalogue_module
from . import guardrails as guardrails_module

NOISE = 0.25
TIER_ORDER = ("free", "paid-low", "paid-mid", "paid-high", "unknown")


def build(bench_result: dict, catalogue: Optional[dict] = None,
         approved_model: Optional[str] = None) -> dict:
    models = (catalogue or {}).get("models") or {}
    rows = []
    for entry in bench_result.get("results") or []:
        model_id = entry["model"]
        meta = models.get(catalogue_module.key(model_id)) or {}
        rows.append({
            "model": model_id,
            "tier": guardrails_module.tier(meta) if meta else "unknown",
            "price_out": meta.get("cheapest_out"),
            "context": meta.get("max_context"),
            "score": entry.get("mean"),
            "wouldShipRate": entry.get("wouldShipRate"),
            "counts": entry.get("counts"),
            # The first real error, surfaced onto the leaderboard row itself.
            # Without it a failed candidate reads as "could not be scored
            # ({'generation_failed': 1})" and a human has to go find the run's
            # JSON file to learn WHY — which is the one piece of information
            # that decides whether the failure is worth investigating (a
            # provider needing its own key) or ignoring (a flaky timeout).
            "error": next((tc.get("error") for tc in entry.get("testCases") or []
                          if tc.get("error")), None),
            "isApproved": approved_model is not None
                         and catalogue_module.key(model_id) == catalogue_module.key(approved_model),
        })

    tiers: dict = {}
    for row in rows:
        tiers.setdefault(row["tier"], []).append(row)
    for band in tiers.values():
        band.sort(key=lambda r: (r["score"] is None, -(r["score"] or 0)))

    shortlists = {}
    for name, band in tiers.items():
        scored = [r for r in band if r["score"] is not None]
        if not scored:
            shortlists[name] = []
            continue
        best = scored[0]["score"]
        shortlists[name] = [r["model"] for r in scored if best - r["score"] <= NOISE]

    return {
        "ranAt": bench_result.get("ranAt"), "useCase": bench_result.get("useCase"),
        "judge": bench_result.get("judge"),
        "tiers": {n: tiers[n] for n in TIER_ORDER if n in tiers},
        "shortlists": shortlists, "noise": NOISE,
    }


def best_overall(board: dict) -> Optional[dict]:
    """The single best-scoring candidate across every tier, for the
    approved-vs-candidate comparison the notifier needs. Ties within a tier are
    a tie; across tiers the higher score still has to be picked as SOMETHING to
    email about, and price is shown alongside it so a human can weigh the two."""
    best = None
    for band in board.get("tiers", {}).values():
        for row in band:
            if row["score"] is None:
                continue
            if best is None or row["score"] > best["score"]:
                best = row
    return best


def report(board: dict) -> str:
    lines = [f"MODEL CICD — {board.get('useCase')}",
            f"  judge   {board.get('judge')}",
            f"  ran     {str(board.get('ranAt'))[:19]}", ""]
    for name, band in board.get("tiers", {}).items():
        lines.append(f"-- {name.upper()} " + "-" * (56 - len(name)))
        for i, r in enumerate(band, start=1):
            if r["score"] is None:
                reason = (r.get("error") or "").split("] ", 1)[-1][:90]
                lines.append(f"   {'-':>3}  {r['model']:<44} "
                             f"could not be scored — {reason or r['counts']}")
                continue
            flag = "*" if r["model"] in (board["shortlists"].get(name) or []) else " "
            approved = "  <- currently approved" if r["isApproved"] else ""
            price = f"${r['price_out']:.2f}/M" if r.get("price_out") is not None else "  -  "
            lines.append(f"  {flag}{i:>3}  {r['model']:<44} {r['score']:.2f}  "
                         f"{price:>9}  ship={r.get('wouldShipRate', 0):.0%}{approved}")
        lines.append("")
    lines += [f"  * = within {board.get('noise')} of this tier's best — a TIE "
             f"at one sample per test case, not a ranking.",
             "  ONE ANSWER PER TEST CASE. Confident at the extremes, "
             "suggestive in the middle."]
    return "\n".join(lines)
