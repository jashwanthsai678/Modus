"""Guards for the free, deterministic parts — where the errors that cost money
or produce a wrong promotion actually live. No model call, no network.

    python -m modelcicd.tests.test_modelcicd
"""
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from modelcicd import catalogue, config, guardrails, judge, rank, state  # noqa: E402

FAILURES: list = []


def check(name: str, condition: bool, detail: str = "") -> None:
    if condition:
        print(f"  ok    {name}")
    else:
        print(f"  FAIL  {name}" + (f" — {detail}" if detail else ""))
        FAILURES.append(name)


def model(key="m", *, price_in=0.10, price_out=0.20, context=200_000,
          json_mode=True, provider="openrouter", model_id=None,
          inputs=("text",), outputs=("text",)) -> dict:
    host = {"provider": provider, "id": model_id or f"vendor/{key}",
            "price_in": price_in, "price_out": price_out, "context": context,
            "json_mode": json_mode, "input_modalities": list(inputs),
            "output_modalities": list(outputs)}
    return {"key": key, "hosts": [host], "cheapest_out": price_out,
            "max_context": context, "json_mode": json_mode,
            "providers": [provider], "name": key}


# ── catalogue: the two bugs that spend money ─────────────────────────────────

def test_negative_price_is_a_sentinel_not_a_bargain() -> None:
    check("-1 becomes unknown", catalogue._price("-1") is None)
    check("a real price survives", catalogue._price("0.0000005") == 0.5)


def test_cross_platform_duplicates_merge() -> None:
    check("llama merges across providers",
          catalogue.key("meta-llama/llama-3.1-70b-instruct")
          == catalogue.key("groq/llama-3.1-70b-versatile"))


def test_distinct_models_never_merge() -> None:
    for a, b in (("openai/gpt-4o", "openai/gpt-4o-mini"),
                 ("deepseek/deepseek-chat", "deepseek/deepseek-r1")):
        check(f"{a.split('/')[-1]} != {b.split('/')[-1]}", catalogue.key(a) != catalogue.key(b))


# ── guardrails ────────────────────────────────────────────────────────────────

def _g(**over):
    g = config.Guardrails()
    for k, v in over.items():
        setattr(g, k, v)
    return g


def test_a_model_within_ceilings_passes() -> None:
    check("passes", guardrails.check(model(price_in=0.50, price_out=3.00), _g()) == [])


def test_over_ceiling_is_named_by_rule() -> None:
    fails = guardrails.check(model(price_in=9.0), _g())
    check("named", any(f["rule"] == "input price too high" for f in fails), f"{fails}")


def test_unknown_price_excluded_not_assumed_free() -> None:
    fails = guardrails.check(model(price_in=None, price_out=None), _g())
    check("excluded", any(f["rule"] == "price unknown" for f in fails))


def test_a_music_model_is_excluded_before_it_costs_a_call() -> None:
    """`google/lyria-3-pro-preview` declares text+image -> text+audio: reads a
    prompt fine, and its output technically "includes" text. Only exclusivity
    catches it, or it costs a paid generation call to discover otherwise."""
    fails = guardrails.check(model(inputs=("text", "image"), outputs=("text", "audio")), _g())
    check("audio output rejected", any(f["rule"] == "wrong modality" for f in fails))
    check("text-only accepted",
          not guardrails.check(model(inputs=("text",), outputs=("text",)), _g()))


def test_a_router_is_not_a_model() -> None:
    fails = guardrails.check(
        model(provider="openrouter", model_id="openrouter/free", price_in=0, price_out=0), _g())
    check("router rejected", any(f["rule"] == "routed meta-model" for f in fails))


def test_tiers_split_where_the_budget_conversation_does() -> None:
    check("free", guardrails.tier(model(price_out=0.0)) == "free")
    check("paid-low", guardrails.tier(model(price_out=0.50)) == "paid-low")
    check("paid-mid", guardrails.tier(model(price_out=3.00)) == "paid-mid")
    check("paid-high", guardrails.tier(model(price_out=9.00)) == "paid-high")


# ── the judge must not mark its own paper ────────────────────────────────────

def test_judge_in_candidate_pool_is_refused() -> None:
    try:
        judge.check_judge_not_candidate("openai/gpt-4o", ["openai/gpt-4o", "google/gemma-3-4b-it"])
        check("refused", False)
    except judge.JudgeIsCandidate:
        check("refused", True)


def test_judge_outside_pool_is_allowed() -> None:
    try:
        judge.check_judge_not_candidate("openai/gpt-4o", ["mistralai/mistral-nemo"])
        check("allowed", True)
    except judge.JudgeIsCandidate as exc:
        check("allowed", False, str(exc))


def test_clash_detected_across_platforms() -> None:
    try:
        judge.check_judge_not_candidate("meta-llama/llama-3.1-70b-instruct",
                                        ["groq/llama-3.1-70b-versatile"])
        check("cross-platform clash caught", False)
    except judge.JudgeIsCandidate:
        check("cross-platform clash caught", True)


# ── config: bring-your-own rubric must be validated, not silently empty ─────

def test_a_use_case_with_no_rubric_anywhere_is_refused() -> None:
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "bad.yaml"
        p.write_text(
            "useCase: x\nsystemPrompt: hi\ntestCases:\n  - id: t1\n    input: hi\n",
            encoding="utf-8")
        try:
            config.load(p)
            check("refused", False)
        except ValueError as exc:
            check("refused", "rubric" in str(exc))


def test_a_real_use_case_loads() -> None:
    uc = config.load(ROOT / "examples" / "prep_material" / "use_case.yaml")
    check("test cases loaded", len(uc.test_cases) == 2)
    check("rubric loaded", len(uc.rubric) == 4)
    check("each test case gets the default rubric",
          all(tc.rubric == uc.rubric for tc in uc.test_cases))


# ── rank ──────────────────────────────────────────────────────────────────────

def _bench(*rows) -> dict:
    return {"ranAt": "t", "useCase": "u", "judge": "openai/gpt-4o", "results": list(rows)}


def _row(model_id, mean, ship=1.0, status="ok") -> dict:
    return {"model": model_id, "mean": mean, "wouldShipRate": ship,
            "counts": {"ok": 1}, "testCases": []}


def test_models_within_noise_are_a_tie() -> None:
    board = rank.build(_bench(_row("a", 4.4), _row("b", 4.3), _row("c", 3.0)), None)
    short = board["shortlists"]["unknown"]
    check("near-tie shortlisted together", set(short) == {"a", "b"}, f"{short}")
    check("clear loser excluded", "c" not in short)


def test_a_failed_candidate_stays_with_its_reason() -> None:
    board = rank.build(_bench({"model": "x", "mean": None, "wouldShipRate": None,
                              "counts": {"ok": 0, "generation_failed": 1},
                              "testCases": [{"error": "[sandbox] 403 Forbidden"}]}),
                       None)
    row = board["tiers"]["unknown"][0]
    check("present", row["model"] == "x")
    check("no score", row["score"] is None)
    check("carries its error", "403" in (row.get("error") or ""))


def test_report_is_ascii() -> None:
    text = rank.report(rank.build(_bench(_row("a", 4.0)), None))
    try:
        text.encode("cp1252")
        check("encodes on cp1252", True)
    except UnicodeEncodeError as exc:
        check("encodes on cp1252", False, str(exc))


# ── state / resolver: approval is the only thing that promotes ─────────────

def test_resolve_fails_loud_with_no_fallback() -> None:
    import modelcicd.resolver as resolver
    with tempfile.TemporaryDirectory() as d:
        try:
            resolver.resolve("nothing_approved_yet", root=Path(d))
            check("raises", False)
        except LookupError:
            check("raises", True)


def test_resolve_uses_fallback_when_nothing_approved() -> None:
    import modelcicd.resolver as resolver
    with tempfile.TemporaryDirectory() as d:
        got = resolver.resolve("nothing_approved_yet", fallback="gpt-4o-mini", root=Path(d))
        check("fallback used", got == "gpt-4o-mini")


def test_a_pending_candidate_does_not_change_what_resolve_returns() -> None:
    """A bench finding a better model is a PROPOSAL. Only approve() may promote."""
    import modelcicd.resolver as resolver
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        state.approve.__module__  # noqa: B018 (import touch)
        st = state.load("uc", root=root)
        st["approvedModel"] = "old-model"
        state._save(st, root=root)
        board = {"useCase": "uc", "tiers": {"paid-low": [
            {"model": "new-model", "tier": "paid-low", "score": 4.9,
             "price_out": 0.1, "isApproved": False, "wouldShipRate": 1.0,
             "counts": {"ok": 1}}]}, "shortlists": {}}
        state.record_run("uc", board, root=root)
        check("resolve still returns the OLD model",
              resolver.resolve("uc", root=root) == "old-model")
        state.approve("uc", root=root)
        check("approve() promotes the pending one",
              resolver.resolve("uc", root=root) == "new-model")


def test_min_improvement_threshold_suppresses_noise() -> None:
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        st = state.load("uc2", root=root)
        st["approvedModel"] = "old-model"
        st["approvedScore"] = 4.0
        state._save(st, root=root)
        board = {"useCase": "uc2", "tiers": {"paid-low": [
            {"model": "barely-better", "tier": "paid-low", "score": 4.05,
             "price_out": 0.1, "isApproved": False, "wouldShipRate": 1.0,
             "counts": {"ok": 1}}]}, "shortlists": {}}
        new_state = state.record_run("uc2", board, root=root)
        check("a 0.05 win does not page anyone", new_state.get("pending") is None,
              f"{new_state.get('pending')}")


def main() -> int:
    print("modelcicd\n")
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
    print()
    if FAILURES:
        print(f"{len(FAILURES)} failure(s): {', '.join(FAILURES)}")
        return 1
    print("all guards pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
