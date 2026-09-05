"""Guards for the free, deterministic parts — where the errors that cost money
or produce a wrong promotion actually live. No model call, no network.

    python -m modelcicd.tests.test_modelcicd
"""
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from modelcicd import (bench, catalogue, client, code_patch, code_scan, config,  # noqa: E402
                       endpoint_client, guardrails, judge, project, rank, runner,
                       sandbox, scheduler, secrets, state, wizard)

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


def test_groq_rows_normalize_without_fabricating_price() -> None:
    rows = catalogue._rows_openai_compatible(
        "groq", {"data": [{"id": "llama-3.1-70b-versatile", "context_window": 131072}]})
    check("one row", len(rows) == 1, f"{rows}")
    check("context read", rows[0]["context"] == 131072)
    check("price unknown, not zero", rows[0]["price_in"] is None and rows[0]["price_out"] is None)
    check("json_mode unknown defaults false", rows[0]["json_mode"] is False)


def test_fireworks_rows_skip_entries_without_an_id() -> None:
    rows = catalogue._rows_openai_compatible("fireworks", {"data": [{"context_length": 4096}]})
    check("no id, no row", rows == [], f"{rows}")


def test_provider_map_traces_each_host_to_its_platform() -> None:
    cat = {"models": {
        "llama": {"hosts": [{"id": "vendor/llama-a", "provider": "groq"},
                            {"id": "vendor/llama-b", "provider": "openrouter"}]},
        "mixtral": {"hosts": [{"id": "vendor/mixtral-a", "provider": "fireworks"}]},
    }}
    pm = catalogue.provider_map(cat)
    check("groq host traced", pm["vendor/llama-a"] == "groq", f"{pm}")
    check("openrouter host traced", pm["vendor/llama-b"] == "openrouter", f"{pm}")
    check("fireworks host traced", pm["vendor/mixtral-a"] == "fireworks", f"{pm}")


def test_provider_map_handles_an_empty_catalogue() -> None:
    check("empty in, empty out", catalogue.provider_map({}) == {})
    check("None in, empty out", catalogue.provider_map(None) == {})


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


def _rate_limited_entry(model_id="x") -> dict:
    return {"model": model_id, "mean": None, "wouldShipRate": None,
            "counts": {"ok": 0, "rate_limited": 1},
            "testCases": [{"status": "rate_limited", "error": "rate limited (429)"}]}


def test_rate_limited_row_is_flagged_without_a_usage_estimate() -> None:
    board = rank.build(_bench(_rate_limited_entry()), None)
    row = board["tiers"]["unknown"][0]
    check("flagged", row["rateLimited"] is True)
    check("note present, no estimate mentioned",
         row["rateLimitNote"] is not None and "estimate" not in row["rateLimitNote"])


def test_rate_limited_row_mentions_the_usage_estimate_when_given() -> None:
    board = rank.build(_bench(_rate_limited_entry()), None, estimated_calls_per_day=2000)
    row = board["tiers"]["unknown"][0]
    check("estimate mentioned", "2,000" in row["rateLimitNote"], row["rateLimitNote"])


def test_a_normal_failure_is_not_flagged_as_rate_limited() -> None:
    board = rank.build(_bench({"model": "x", "mean": None, "wouldShipRate": None,
                              "counts": {"ok": 0, "generation_failed": 1},
                              "testCases": [{"error": "[sandbox] 403 Forbidden"}]}),
                       None)
    row = board["tiers"]["unknown"][0]
    check("not flagged", row["rateLimited"] is False)
    check("no note", row["rateLimitNote"] is None)


# ── client: a 429 is a distinct, real signal — never a guessed limit ────────

def test_rate_limit_detected_on_429() -> None:
    import httpx
    resp = httpx.Response(429, request=httpx.Request("POST", "http://example.com"))
    try:
        client._raise_if_rate_limited(resp)
        check("raised", False)
    except client.RateLimitedError:
        check("raised", True)


def test_rate_limit_not_raised_on_ok_or_server_error() -> None:
    import httpx
    for code in (200, 500):
        resp = httpx.Response(code, request=httpx.Request("POST", "http://example.com"))
        try:
            client._raise_if_rate_limited(resp)
            check(f"no raise on {code}", True)
        except client.RateLimitedError:
            check(f"no raise on {code}", False)


def test_sandbox_maps_rate_limited_error_to_its_own_status() -> None:
    import asyncio

    async def fake_call_json(*args, **kwargs):
        raise client.RateLimitedError("rate limited (429)")

    uc = config.load(ROOT / "examples" / "prep_material" / "use_case.yaml")
    original = sandbox.client_module.call_json
    sandbox.client_module.call_json = fake_call_json
    try:
        result = asyncio.run(sandbox.run_one("vendor/model", uc, uc.test_cases[0]))
    finally:
        sandbox.client_module.call_json = original
    check("status is rate_limited", result["status"] == "rate_limited", f"{result}")


def test_sandbox_calls_through_the_candidates_own_platform() -> None:
    """A Groq candidate must actually be called against Groq's endpoint with
    Groq's key env var — not silently sent to OpenRouter regardless of which
    platform it was discovered on."""
    import asyncio

    seen = {}

    async def fake_call_json(*args, **kwargs):
        seen["base_url"] = kwargs.get("base_url")
        seen["api_key_env"] = kwargs.get("api_key_env")
        return {"reply": "ok"}

    uc = config.load(ROOT / "examples" / "prep_material" / "use_case.yaml")
    original = sandbox.client_module.call_json
    sandbox.client_module.call_json = fake_call_json
    try:
        asyncio.run(sandbox.run_one("vendor/model", uc, uc.test_cases[0], provider="groq"))
    finally:
        sandbox.client_module.call_json = original
    check("used groq's chat endpoint", seen.get("base_url") == catalogue.PROVIDERS["groq"]["chat_url"],
         f"{seen}")
    check("used groq's key env var", seen.get("api_key_env") == "GROQ_API_KEY", f"{seen}")


def test_sandbox_defaults_to_openrouter_for_an_unrecognized_provider() -> None:
    import asyncio

    seen = {}

    async def fake_call_json(*args, **kwargs):
        seen["base_url"] = kwargs.get("base_url")
        return {"reply": "ok"}

    uc = config.load(ROOT / "examples" / "prep_material" / "use_case.yaml")
    original = sandbox.client_module.call_json
    sandbox.client_module.call_json = fake_call_json
    try:
        asyncio.run(sandbox.run_one("vendor/model", uc, uc.test_cases[0], provider="nonsense"))
    finally:
        sandbox.client_module.call_json = original
    check("falls back to openrouter", seen.get("base_url") == catalogue.PROVIDERS["openrouter"]["chat_url"],
         f"{seen}")


# ── client: an unset key names the RIGHT env var, whichever platform it is ──

def test_call_json_missing_key_names_the_given_env_var() -> None:
    import asyncio
    import os

    var = "MODELCICD_TEST_MISSING_KEY"
    os.environ.pop(var, None)
    try:
        asyncio.run(client.call_json("hi", model="x", label="t", api_key_env=var))
        check("raised", False)
    except RuntimeError as exc:
        check("names the right env var", var in str(exc), str(exc))


# ── secrets: the one place this project writes an API key ───────────────────

def test_set_key_preserves_other_lines_and_updates_environ() -> None:
    import os

    with tempfile.TemporaryDirectory() as d:
        env_path = Path(d) / ".env"
        env_path.write_text("SOME_OTHER_VAR=keep-me\nOPENROUTER_API_KEY=old\n", encoding="utf-8")
        secrets.set_key("groq", "new-groq-key", env_path=env_path)
        text = env_path.read_text(encoding="utf-8")
        check("other line preserved", "SOME_OTHER_VAR=keep-me" in text, text)
        check("existing key untouched", "OPENROUTER_API_KEY=old" in text, text)
        check("new key appended", "GROQ_API_KEY=new-groq-key" in text, text)
        check("environ updated immediately", os.environ.get("GROQ_API_KEY") == "new-groq-key")


def test_set_key_replaces_an_existing_line_rather_than_duplicating() -> None:
    with tempfile.TemporaryDirectory() as d:
        env_path = Path(d) / ".env"
        env_path.write_text("GROQ_API_KEY=old-value\n", encoding="utf-8")
        secrets.set_key("groq", "new-value", env_path=env_path)
        text = env_path.read_text(encoding="utf-8")
        check("only one line for the key", text.count("GROQ_API_KEY=") == 1, text)
        check("value updated", "GROQ_API_KEY=new-value" in text, text)


def test_set_key_rejects_an_unknown_provider() -> None:
    with tempfile.TemporaryDirectory() as d:
        try:
            secrets.set_key("not-a-real-provider", "x", env_path=Path(d) / ".env")
            check("refused", False)
        except ValueError:
            check("refused", True)


def test_status_reflects_the_live_environment() -> None:
    import os

    var = "FIREWORKS_API_KEY"
    had = os.environ.pop(var, None)
    try:
        check("not set", secrets.status()["fireworks"] is False)
        os.environ[var] = "some-key"
        check("now set", secrets.status()["fireworks"] is True)
    finally:
        if had is not None:
            os.environ[var] = had
        else:
            os.environ.pop(var, None)


# ── usage estimate: round-trips, never used to filter or rank ───────────────

def test_usage_estimate_round_trips_through_wizard_and_config() -> None:
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "demo_feature.yaml"
        fields = _wizard_fields(estimated_calls_per_day=2000)
        p.write_text(wizard.to_yaml(fields), encoding="utf-8")
        uc = config.load(p)
        check("estimate loaded", uc.estimated_calls_per_day == 2000)


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


# ── state: dismissing is not approving, and the pending queue is real ───────

def test_reject_clears_pending_without_touching_approved() -> None:
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        st = state.load("uc3", root=root)
        st["approvedModel"] = "old-model"
        st["pending"] = {"model": "candidate-x", "score": 4.5, "foundAt": "t"}
        st["notifiedModel"] = "candidate-x"
        state._save(st, root=root)

        new_state = state.reject("uc3", root=root)
        check("pending cleared", new_state.get("pending") is None)
        check("notifiedModel cleared", new_state.get("notifiedModel") is None)
        check("approved untouched", new_state.get("approvedModel") == "old-model")
        check("recorded, not forgotten",
             new_state.get("rejected") and new_state["rejected"][-1]["model"] == "candidate-x")


def test_reject_with_nothing_pending_is_refused() -> None:
    with tempfile.TemporaryDirectory() as d:
        try:
            state.reject("uc4", root=Path(d))
            check("refused", False)
        except ValueError:
            check("refused", True)


# ── state: concurrent writers to the same use case never lose an update ─────

def test_lock_file_is_cleaned_up_after_normal_use() -> None:
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        state.set_min_improvement("uc-lock", 0.3, root=root)
        lock_path = state._path("uc-lock", root).with_suffix(".json.lock")
        check("no leftover .lock file", not lock_path.exists(), f"{lock_path}")


def test_stuck_lock_raises_timeout_instead_of_corrupting_state() -> None:
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        target = state._path("uc-stuck", root)
        target.parent.mkdir(parents=True, exist_ok=True)
        lock = state._FileLock(target)
        lock._path.parent.mkdir(parents=True, exist_ok=True)
        lock._path.touch()  # simulate another process already holding it
        original_timeout = state._LOCK_TIMEOUT_S
        state._LOCK_TIMEOUT_S = 0.1
        try:
            state.set_min_improvement("uc-stuck", 0.3, root=root)
            check("raised", False)
        except TimeoutError:
            check("raised", True)
        finally:
            state._LOCK_TIMEOUT_S = original_timeout
            lock._path.unlink()
        check("state file untouched by the failed writer", not target.exists())


def test_concurrent_record_run_never_loses_an_update() -> None:
    """The exact bug a missing lock would allow: N threads all recording a
    run for the SAME use case at once, each appending its own distinct
    history entry. Without the lock, a read-modify-write race can silently
    drop entries when two writers save based on the same stale read — this
    proves none are lost."""
    import threading

    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        n = 8
        barrier = threading.Barrier(n)

        def record(i: int) -> None:
            barrier.wait()  # maximize actual overlap, not just "runs eventually"
            board = {"useCase": "uc-race", "tiers": {"paid-low": [
                {"model": f"model-{i}", "tier": "paid-low", "score": 3.0,
                 "price_out": 0.1, "isApproved": False, "wouldShipRate": 1.0,
                 "counts": {"ok": 1}}]}, "shortlists": {}}
            state.record_run("uc-race", board, root=root)

        threads = [threading.Thread(target=record, args=(i,)) for i in range(n)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        final = state.load("uc-race", root=root)
        check(f"all {n} concurrent writes landed in history",
             len(final.get("history") or []) == n, f"{len(final.get('history') or [])}")


def test_all_pending_finds_items_across_projects_and_unscoped() -> None:
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        projects_root = root / "projects"
        unscoped_root = root / "state"

        proj = project.create("Demo App", notify_email="demo@example.com", root=projects_root)
        st = state.load("scoped_uc", project.state_dir(proj.slug, projects_root))
        st["pending"] = {"model": "a", "score": 4.0, "foundAt": "t"}
        state._save(st, project.state_dir(proj.slug, projects_root))

        st2 = state.load("unscoped_uc", unscoped_root)
        st2["pending"] = {"model": "b", "score": 4.1, "foundAt": "t"}
        state._save(st2, unscoped_root)

        found = state.all_pending(projects_root=projects_root, unscoped_root=unscoped_root)
        names = {(f["project"], f["useCase"]) for f in found}
        check("scoped found", ("demo-app", "scoped_uc") in names, f"{names}")
        check("unscoped found", (None, "unscoped_uc") in names, f"{names}")


def _fake_bench_result(model="vendor/winner", score=4.5) -> dict:
    return {"schema": 1, "ranAt": "2026-01-01T00:00:00+00:00", "useCase": "prep_material_writer",
            "judge": "openai/gpt-4o", "testCases": 1, "candidates": 1,
            "results": [{"model": model, "mean": score, "wouldShipRate": 1.0,
                        "counts": {"ok": 1, "generation_failed": 0, "judge_failed": 0,
                                  "rate_limited": 0},
                        "testCases": [{"testCase": "t1", "status": "ok",
                                      "output": "hi", "weighted": score}]}]}


def test_runner_does_not_renotify_for_an_unchanged_pending_candidate() -> None:
    import asyncio

    async def fake_bench_run(candidates, uc, *, judge_model=None, provider_by_model=None):
        return _fake_bench_result()

    async def fake_rejudge(*a, **k):
        return {}

    sent = []

    def fake_send_pending(use_case, state_dict, text, to=None):
        sent.append(use_case)
        return True

    original_run = runner.bench_module.run
    original_rejudge = runner.bench_module.rejudge_for_spread
    original_notify = runner.notify_module.send_pending
    runner.bench_module.run = fake_bench_run
    runner.bench_module.rejudge_for_spread = fake_rejudge
    runner.notify_module.send_pending = fake_send_pending
    try:
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            uc = config.load(ROOT / "examples" / "prep_material" / "use_case.yaml")
            asyncio.run(runner.execute(uc, {}, ["vendor/winner"], state_root=root, out_dir=root))
            asyncio.run(runner.execute(uc, {}, ["vendor/winner"], state_root=root, out_dir=root))
    finally:
        runner.bench_module.run = original_run
        runner.bench_module.rejudge_for_spread = original_rejudge
        runner.notify_module.send_pending = original_notify
    check("notified exactly once for the same repeated candidate", sent == ["prep_material_writer"],
         f"{sent}")


# ── judge: repeated scoring surfaces JUDGE noise, not just candidate noise ──

def test_judge_score_repeated_reports_the_spread() -> None:
    import asyncio

    uc = config.load(ROOT / "examples" / "prep_material" / "use_case.yaml")
    tc = uc.test_cases[0]
    criteria_ids = [c.id for c in (tc.rubric or uc.rubric)]
    responses = iter([
        {"scores": {cid: 5 for cid in criteria_ids}, "reasons": {}, "wouldShip": True},
        {"scores": {cid: 3 for cid in criteria_ids}, "reasons": {}, "wouldShip": True},
        {"scores": {cid: 4 for cid in criteria_ids}, "reasons": {}, "wouldShip": True},
    ])

    async def fake_call_json(*a, **k):
        return next(responses)

    original = judge.client_module.call_json
    judge.client_module.call_json = fake_call_json
    try:
        result = asyncio.run(judge.score_repeated("vendor/x", "openai/gpt-4o", uc, tc,
                                                   "some output", repeats=3))
    finally:
        judge.client_module.call_json = original
    check("spread is max-min across repeats", result["spread"] == 2.0, f"{result}")
    check("weighted is the median", result["weighted"] == 4.0, f"{result}")


def test_bench_resample_candidates_for_spread_measures_candidate_noise() -> None:
    """A DIFFERENT source of noise than judge-spread: this re-GENERATES the
    candidate's answer, not just re-scores an existing one. Single test
    case on purpose, so the spread (max-min across the 2 repeats) is
    order-invariant regardless of how asyncio interleaves the concurrent
    calls — the point under test is the mechanism, not a race-prone exact
    ordering."""
    import asyncio
    import itertools

    uc = config.load(ROOT / "examples" / "prep_material" / "use_case.yaml")
    uc.test_cases = uc.test_cases[:1]
    criteria_ids = [c.id for c in (uc.test_cases[0].rubric or uc.rubric)]
    judge_scores = itertools.cycle([5, 3])

    async def fake_call_json(prompt, *, model, label, **kwargs):
        if label.startswith("sandbox["):
            return {"reply": "hi"}
        score = next(judge_scores)
        return {"scores": {cid: score for cid in criteria_ids}, "reasons": {}, "wouldShip": True}

    original = client.call_json
    client.call_json = fake_call_json
    try:
        result = asyncio.run(bench.resample_candidates_for_spread(
            uc, ["vendor/x"], "openai/gpt-4o", repeats=2))
    finally:
        client.call_json = original
    check("candidate spread measured from repeated generation",
         result.get("vendor/x") == 2.0, f"{result}")


def test_rank_attach_candidate_spread_only_touches_named_rows() -> None:
    board = rank.build(_bench(_row("a", 4.4), _row("b", 4.3)), None)
    rank.attach_candidate_spread(board, {"a": 0.4})
    row_a = next(r for r in board["tiers"]["unknown"] if r["model"] == "a")
    row_b = next(r for r in board["tiers"]["unknown"] if r["model"] == "b")
    check("named row updated", row_a["candidateSpread"] == 0.4)
    check("unnamed row stays None", row_b["candidateSpread"] is None)


def test_rank_attach_judge_spread_only_touches_named_rows() -> None:
    board = rank.build(_bench(_row("a", 4.4), _row("b", 4.3)), None)
    rank.attach_judge_spread(board, {"a": 0.3})
    row_a = next(r for r in board["tiers"]["unknown"] if r["model"] == "a")
    row_b = next(r for r in board["tiers"]["unknown"] if r["model"] == "b")
    check("named row updated", row_a["judgeSpread"] == 0.3)
    check("unnamed row stays None", row_b["judgeSpread"] is None)


# ── code_scan: confidence is a real, actionable signal ───────────────────────

def test_scan_summary_counts_by_confidence() -> None:
    results = [{"confidence": "high"}, {"confidence": "high"}, {"confidence": "low"},
              {"confidence": None}]
    s = code_scan.summary(results)
    check("high count", s["high"] == 2, f"{s}")
    check("low count (incl. missing)", s["low"] == 2, f"{s}")
    check("total", s["total"] == 4)


# ── project: a lightweight registration, round-tripped ──────────────────────

def test_project_create_load_list_round_trip() -> None:
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        proj = project.create("Demo App", description="a demo",
                              notify_email="demo@example.com", root=root)
        check("slugified", proj.slug == "demo-app", proj.slug)
        loaded = project.load("demo-app", root=root)
        check("name round-trips", loaded.name == "Demo App")
        check("description round-trips", loaded.description == "a demo")
        check("listed", [p.slug for p in project.list_all(root=root)] == ["demo-app"])
        check("dirs created",
              project.use_cases_dir("demo-app", root=root).exists()
              and project.state_dir("demo-app", root=root).exists())


def test_project_duplicate_slug_is_refused() -> None:
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        project.create("Demo App", notify_email="demo@example.com", root=root)
        try:
            project.create("Demo App", notify_email="demo@example.com", root=root)
            check("refused", False)
        except ValueError:
            check("refused", True)


def test_project_requires_a_notify_email() -> None:
    with tempfile.TemporaryDirectory() as d:
        try:
            project.create("Demo App", root=Path(d))
            check("refused", False)
        except ValueError as exc:
            check("refused", True)
            check("names why", "notify" in str(exc).lower(), str(exc))


def test_project_rejects_a_repo_path_that_does_not_exist() -> None:
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        try:
            project.create("Demo App", notify_email="demo@example.com",
                          repo_path=str(root / "nowhere"), root=root)
            check("refused", False)
        except ValueError:
            check("refused", True)


def test_project_rejects_both_repo_path_and_repo_url() -> None:
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        try:
            project.create("X", notify_email="demo@example.com", repo_path=".",
                          repo_url="https://example.com/x.git", root=root)
            check("refused", False)
        except ValueError:
            check("refused", True)


def test_project_providers_default_to_openrouter() -> None:
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        proj = project.create("Demo App", notify_email="demo@example.com", root=root)
        check("default provider", proj.providers == ["openrouter"], f"{proj.providers}")
        loaded = project.load("demo-app", root=root)
        check("providers round-trip", loaded.providers == ["openrouter"])


def test_read_repo_file_refuses_a_path_outside_the_repo() -> None:
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        repo = root / "repo"
        repo.mkdir()
        (repo / "app.py").write_text("model = 'gpt-4o-mini'\n", encoding="utf-8")
        project.create("Demo App", notify_email="demo@example.com", repo_path=str(repo), root=root)
        check("reads a real file",
              "gpt-4o-mini" in project.read_repo_file("demo-app", "app.py", root=root))
        try:
            project.read_repo_file("demo-app", "../outside.py", root=root)
            check("escape refused", False)
        except ValueError:
            check("escape refused", True)


# ── code_patch: the narrowest possible edit to someone else's real code ─────

def test_code_patch_preview_and_apply_round_trip() -> None:
    with tempfile.TemporaryDirectory() as d:
        repo = Path(d)
        target = repo / "app.py"
        target.write_text("model = 'gpt-4o-mini'\nother = 'gpt-4o-mini-extra'\n", encoding="utf-8")
        prev = code_patch.preview(repo, "app.py", "gpt-4o-mini", "gpt-4.1-mini")
        check("counts exact occurrences", prev["occurrences"] == 2, f"{prev}")
        count = code_patch.apply(repo, "app.py", "gpt-4o-mini", "gpt-4.1-mini")
        check("apply returns the count", count == 2)
        check("file actually changed",
              target.read_text(encoding="utf-8").count("gpt-4.1-mini") == 2)


def test_code_patch_preview_refuses_when_old_string_is_absent() -> None:
    with tempfile.TemporaryDirectory() as d:
        repo = Path(d)
        (repo / "app.py").write_text("model = 'claude-3'\n", encoding="utf-8")
        try:
            code_patch.preview(repo, "app.py", "gpt-4o-mini", "gpt-4.1-mini")
            check("refused", False)
        except ValueError:
            check("refused", True)


def test_code_patch_apply_refuses_paths_outside_the_repo() -> None:
    with tempfile.TemporaryDirectory() as d:
        repo = Path(d)
        try:
            code_patch.preview(repo, "../elsewhere.py", "a", "b")
            check("escape refused", False)
        except ValueError:
            check("escape refused", True)


# ── code_scan: a cheap, language-agnostic prefilter before any model call ───

def test_candidate_files_keeps_keyword_matches_and_skips_noise() -> None:
    with tempfile.TemporaryDirectory() as d:
        repo = Path(d)
        (repo / "node_modules").mkdir()
        (repo / "node_modules" / "skip_me.py").write_text(
            "import openai\nopenai.ChatCompletion.create(model='x')\n", encoding="utf-8")
        (repo / "bot.py").write_text(
            "import openai\nopenai.ChatCompletion.create(model='gpt-4o-mini')\n", encoding="utf-8")
        (repo / "unrelated.py").write_text("def add(a, b):\n    return a + b\n", encoding="utf-8")
        files = code_scan.candidate_files(repo)
        names = [str(f) for f in files]
        check("keeps the real match", any("bot.py" in n and "node_modules" not in n for n in names))
        check("skips node_modules", not any("node_modules" in n for n in names))
        check("skips a file with no LLM keywords", not any("unrelated.py" in n for n in names))


def test_scan_repo_aggregates_and_dedupes() -> None:
    import asyncio

    with tempfile.TemporaryDirectory() as d:
        repo = Path(d)
        (repo / "bot.py").write_text("openai.ChatCompletion.create(model='gpt-4o-mini')\n",
                                     encoding="utf-8")

        async def fake_call_json(prompt, *, model, label, **kwargs):
            return {"calls": [{"model": "gpt-4o-mini", "prompt": "hi", "line": 1,
                               "confidence": "high"}]}

        original = code_scan.client_module.call_json
        code_scan.client_module.call_json = fake_call_json
        try:
            results = asyncio.run(code_scan.scan_repo(repo))
        finally:
            code_scan.client_module.call_json = original
        check("found one candidate", len(results) == 1, f"{results}")
        check("carries the file", results[0]["file"].endswith("bot.py"))


def test_split_results_separates_errors_from_found() -> None:
    results = [{"file": "a.py", "model": "x"}, {"file": "b.py", "error": "boom"}]
    found, errors = code_scan.split_results(results)
    check("one found", len(found) == 1 and found[0]["file"] == "a.py")
    check("one error", len(errors) == 1 and errors[0]["file"] == "b.py")


def test_local_imports_follows_relative_python_import_only_in_repo() -> None:
    with tempfile.TemporaryDirectory() as d:
        repo = Path(d)
        pkg = repo / "agent"
        pkg.mkdir()
        (pkg / "prompts.py").write_text(
            "TICKET_ROUTER_INSTRUCTIONS = 'be a routing assistant'\n", encoding="utf-8")
        (pkg / "router.py").write_text(
            "import os\nfrom groq import Groq\nfrom . import prompts\n"
            "def route(): return prompts.TICKET_ROUTER_INSTRUCTIONS\n", encoding="utf-8")
        hits = code_scan.local_imports(pkg / "router.py", repo.resolve())
        names = [h.name for h in hits]
        check("follows the sibling module", "prompts.py" in names, f"{names}")
        check("never follows a third-party package (groq has no local file)",
              "groq" not in names and len(hits) == 1, f"{names}")


def test_local_imports_from_dotted_form_also_resolves() -> None:
    with tempfile.TemporaryDirectory() as d:
        repo = Path(d)
        pkg = repo / "agent"
        pkg.mkdir()
        (pkg / "prompts.py").write_text("X = 1\n", encoding="utf-8")
        (pkg / "router.py").write_text(
            "from .prompts import X\n", encoding="utf-8")
        hits = code_scan.local_imports(pkg / "router.py", repo.resolve())
        check("resolves 'from .prompts import X' too",
              any(h.name == "prompts.py" for h in hits), f"{hits}")


def test_scan_file_bundles_local_import_content_into_the_same_call() -> None:
    import asyncio

    with tempfile.TemporaryDirectory() as d:
        repo = Path(d)
        pkg = repo / "agent"
        pkg.mkdir()
        (pkg / "prompts.py").write_text(
            "SECRET_MARKER_PROMPT = 'be a routing assistant'\n", encoding="utf-8")
        (pkg / "router.py").write_text(
            "from . import prompts\ndef route(): pass\n", encoding="utf-8")

        seen_prompt = {}

        async def fake_call_json(prompt, *, model, label, **kwargs):
            seen_prompt["text"] = prompt
            return {"calls": []}

        original = code_scan.client_module.call_json
        code_scan.client_module.call_json = fake_call_json
        try:
            asyncio.run(code_scan.scan_file(pkg / "router.py", scan_model="x",
                                            repo_root=repo.resolve()))
        finally:
            code_scan.client_module.call_json = original
        check("the imported file's content reached the same call",
              "SECRET_MARKER_PROMPT" in seen_prompt.get("text", ""))
        check("labeled as imported, not as the file being scanned",
              "IMPORTED FILE:" in seen_prompt.get("text", ""))


# ── runner: candidate selection respects a project's chosen providers ───────

def test_select_candidates_filters_by_provider() -> None:
    cat = {"models": {}}
    uc = config.load(ROOT / "examples" / "prep_material" / "use_case.yaml")
    passed_models = [
        {"key": "a", "hosts": [{"id": "vendor/a", "price_out": 0.1}],
         "providers": ["openrouter"]},
        {"key": "b", "hosts": [{"id": "vendor/b", "price_out": 0.1}],
         "providers": ["groq"]},
    ]
    import types
    fake_guardrails = types.SimpleNamespace(
        apply=lambda cat, g: {"passed": passed_models, "failed": []},
        within_tiers=lambda models, tiers: models)
    original = runner.guardrails_module
    runner.guardrails_module = fake_guardrails
    try:
        ids = runner.select_candidates(uc, cat, providers=["groq"])
    finally:
        runner.guardrails_module = original
    check("only the groq model survives", ids == ["vendor/b"], f"{ids}")


def test_wizard_code_target_needs_current_model() -> None:
    errors = wizard.validate(_wizard_fields(code_file="app.py", code_current_model=None))
    check("current model required", any("code" in e for e in errors), f"{errors}")
    check("passes once given",
         wizard.validate(_wizard_fields(code_file="app.py", code_current_model="gpt-4o-mini")) == [])


# ── wizard: the only path from "someone's answers" to a valid use_case.yaml ─

def _wizard_fields(**over) -> wizard.WizardFields:
    f = wizard.WizardFields(
        name="demo_feature", system_prompt="You are helpful.",
        test_cases=[{"id": "t1", "input": "hi", "reference": None}],
        rubric=[{"id": "tone", "description": "is it nice", "weight": 1.0}])
    for k, v in over.items():
        setattr(f, k, v)
    return f


def test_wizard_yaml_round_trips_through_config_load() -> None:
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "demo_feature.yaml"
        p.write_text(wizard.to_yaml(_wizard_fields()), encoding="utf-8")
        uc = config.load(p)
        check("name", uc.name == "demo_feature")
        check("test case", len(uc.test_cases) == 1 and uc.test_cases[0].id == "t1")
        check("rubric", len(uc.rubric) == 1 and uc.rubric[0].id == "tone")


def test_wizard_yaml_carries_endpoint_and_schedule() -> None:
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "demo_feature.yaml"
        fields = _wizard_fields(endpoint_url="http://localhost:8000/reply",
                                schedule_interval_days=7)
        p.write_text(wizard.to_yaml(fields), encoding="utf-8")
        uc = config.load(p)
        check("endpoint loaded", uc.endpoint is not None and uc.endpoint.url == fields.endpoint_url)
        check("schedule loaded", uc.schedule_interval_days == 7)


def test_wizard_rejects_missing_name() -> None:
    errors = wizard.validate(_wizard_fields(name=""))
    check("name required", any("name" in e for e in errors), f"{errors}")


def test_wizard_rejects_no_test_cases() -> None:
    errors = wizard.validate(_wizard_fields(test_cases=[]))
    check("test case required", any("test case" in e for e in errors), f"{errors}")


def test_wizard_rejects_no_rubric() -> None:
    errors = wizard.validate(_wizard_fields(rubric=[]))
    check("rubric required", any("rubric" in e for e in errors), f"{errors}")


def test_wizard_valid_fields_pass() -> None:
    check("no errors", wizard.validate(_wizard_fields()) == [])


# ── scheduler: only what has a schedule, and only when it's actually due ────

def test_due_is_false_with_no_interval() -> None:
    check("never due", scheduler.due({}, None) is False)


def test_due_is_true_when_never_run() -> None:
    check("due", scheduler.due({}, 7) is True)


def test_due_respects_the_interval() -> None:
    from datetime import datetime, timedelta, timezone
    now = datetime.now(timezone.utc)
    recent = {"history": [{"ranAt": (now - timedelta(days=1)).isoformat()}]}
    stale = {"history": [{"ranAt": (now - timedelta(days=10)).isoformat()}]}
    check("too soon", scheduler.due(recent, 7, now=now) is False)
    check("overdue", scheduler.due(stale, 7, now=now) is True)


# ── endpoint_client: pure helpers, no network ────────────────────────────────

def test_expand_env_substitutes_from_environment() -> None:
    import os
    os.environ["MODELCICD_TEST_TOKEN"] = "secret123"
    check("expanded", endpoint_client._expand_env("Bearer ${MODELCICD_TEST_TOKEN}")
         == "Bearer secret123")
    del os.environ["MODELCICD_TEST_TOKEN"]


def test_dig_reads_a_nested_field() -> None:
    check("nested field", endpoint_client._dig({"data": {"reply": "hi"}}, "data.reply") == "hi")
    try:
        endpoint_client._dig({"data": {}}, "data.reply")
        check("missing field raises", False)
    except KeyError:
        check("missing field raises", True)


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
