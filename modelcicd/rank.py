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


def _rate_limit_note(rate_limited_count: int, estimated_calls_per_day: Optional[int]
                     ) -> Optional[str]:
    if not rate_limited_count:
        return None
    if estimated_calls_per_day:
        return (f"rate limited during this bench ({rate_limited_count} call(s)) — "
               f"a light test already hit it; your ~{estimated_calls_per_day:,}/day "
               f"estimate is unlikely to be sustainable here")
    return f"rate limited during this bench ({rate_limited_count} call(s))"


def build(bench_result: dict, catalogue: Optional[dict] = None,
         approved_model: Optional[str] = None, *,
         estimated_calls_per_day: Optional[int] = None) -> dict:
    models = (catalogue or {}).get("models") or {}
    rows = []
    for entry in bench_result.get("results") or []:
        model_id = entry["model"]
        meta = models.get(catalogue_module.key(model_id)) or {}
        rate_limited_count = (entry.get("counts") or {}).get("rate_limited", 0)
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
            # A REAL, OBSERVED signal, never a guessed per-model limit: this
            # candidate returned 429 during the bench itself. Paired with the
            # use case's own usage estimate, if it gave one, so a human sees
            # both the evidence and the context together.
            "rateLimited": rate_limited_count > 0,
            "rateLimitNote": _rate_limit_note(rate_limited_count, estimated_calls_per_day),
            # Filled in later by `attach_judge_spread`, only for candidates
            # that made a tie-band shortlist — how much the JUDGE's own
            # score moved across repeated looks at the SAME answer. None
            # means "not re-checked" (outside the tie zone), not "zero
            # noise" — don't read a bare None as a clean bill of health.
            "judgeSpread": None,
            # Filled in later by `attach_candidate_spread`, only when a
            # caller opted into `bench.resample_candidates_for_spread` —
            # how much the CANDIDATE's own answer varied across repeated,
            # freshly-generated attempts at the same test case. A different
            # source of noise than judgeSpread: that one re-scores the same
            # answer, this one re-asks the question. None means "not
            # measured", not "zero noise".
            "candidateSpread": None,
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


def attach_judge_spread(board: dict, spread_by_model: dict) -> None:
    """Writes `judgeSpread` onto whichever rows are in `spread_by_model` —
    mutates `board` in place, called after `build()` once a caller has
    re-judged the tie zone (see `bench.rejudge_for_spread`). A model not in
    `spread_by_model` keeps `judgeSpread: None` — never re-checked, not
    assumed noise-free."""
    for band in board.get("tiers", {}).values():
        for row in band:
            if row["model"] in spread_by_model:
                row["judgeSpread"] = spread_by_model[row["model"]]


def attach_candidate_spread(board: dict, spread_by_model: dict) -> None:
    """Writes `candidateSpread` onto whichever rows are in `spread_by_model`
    — mutates `board` in place, called after `build()` once a caller has
    opted into `bench.resample_candidates_for_spread` for the tie zone. A
    model not in `spread_by_model` keeps `candidateSpread: None` — never
    re-sampled, not assumed noise-free."""
    for band in board.get("tiers", {}).values():
        for row in band:
            if row["model"] in spread_by_model:
                row["candidateSpread"] = spread_by_model[row["model"]]


def combined_score(score: Optional[float], price_out: Optional[float],
                   max_price_out: float, price_sensitivity: float) -> Optional[float]:
    """Blends a 1-5 quality score with price into one number, weighted by
    `price_sensitivity` (0-100 — 0 means "price doesn't count", 100 means
    "only price counts"). AT 0, THIS RETURNS `score` UNCHANGED — the
    default reproduces today's quality-only ranking exactly, so nothing
    about an existing leaderboard's order changes unless someone actually
    moves this dial.

    Price is normalized against `max_price_out` — the feature's OWN price
    ceiling, what you already said you're willing to pay at most — never
    against whatever happened to be the cheapest candidate that ran. A
    candidate right at the ceiling scores 0 on price, free scores 5,
    linear between. UNKNOWN PRICE IS NEVER FAVORABLE, the same rule as
    every other guardrail in this project: a candidate with no price data
    gets the worst possible price score, never a guessed good one."""
    if score is None:
        return None
    w = max(0.0, min(100.0, price_sensitivity)) / 100.0
    if w == 0.0:
        return round(score, 3)
    if price_out is None:
        price_score = 0.0
    elif max_price_out <= 0:
        price_score = 5.0 if price_out <= 0 else 0.0
    else:
        price_score = 5.0 * max(0.0, 1.0 - min(price_out, max_price_out) / max_price_out)
    return round(score * (1 - w) + price_score * w, 3)


def reorder_by_preference(board: dict, *, price_sensitivity: float, max_price_out: float) -> dict:
    """A NEW `{tier: [row, ...]}` dict, re-sorted by `combined_score`
    instead of raw quality — never mutates `board`, and never touches what
    `record_run` used to decide what's pending (that stays quality-only,
    always — this is a display-time lens for a human comparing options, not
    something that changes what gets automatically proposed). At
    `price_sensitivity=0` this is `board["tiers"]` itself, unchanged."""
    if not price_sensitivity:
        return board.get("tiers") or {}
    out = {}
    for tier, rows in (board.get("tiers") or {}).items():
        annotated = []
        for r in rows:
            c = combined_score(r.get("score"), r.get("price_out"), max_price_out, price_sensitivity)
            annotated.append({**r, "combinedScore": c})
        annotated.sort(key=lambda r: (r["combinedScore"] is None, -(r["combinedScore"] or 0)))
        out[tier] = annotated
    return out


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
                if r.get("rateLimited"):
                    reason = r.get("rateLimitNote")
                else:
                    reason = (r.get("error") or "").split("] ", 1)[-1][:90]
                lines.append(f"   {'-':>3}  {r['model']:<44} "
                             f"could not be scored — {reason or r['counts']}")
                continue
            flag = "*" if r["model"] in (board["shortlists"].get(name) or []) else " "
            approved = "  <- currently approved" if r["isApproved"] else ""
            price = f"${r['price_out']:.2f}/M" if r.get("price_out") is not None else "  -  "
            spread = f"  (judge spread ±{r['judgeSpread']:.2f})" if r.get("judgeSpread") is not None else ""
            cand_spread = (f"  (candidate spread ±{r['candidateSpread']:.2f})"
                          if r.get("candidateSpread") is not None else "")
            lines.append(f"  {flag}{i:>3}  {r['model']:<44} {r['score']:.2f}  "
                         f"{price:>9}  ship={r.get('wouldShipRate', 0):.0%}{approved}"
                         f"{spread}{cand_spread}")
        lines.append("")
    lines += [f"  * = within {board.get('noise')} of this tier's best — a TIE "
             f"at one sample per test case, not a ranking.",
             "  ONE ANSWER PER TEST CASE. Confident at the extremes, "
             "suggestive in the middle."]
    return "\n".join(lines)
