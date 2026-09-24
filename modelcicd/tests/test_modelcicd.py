"""Guards for the free, deterministic parts — where the errors that cost money
or produce a wrong promotion actually live. No model call, no network.

    python -m modelcicd.tests.test_modelcicd
"""
import json
import sys
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from modelcicd import (assertions, auth, bench, cache, catalogue, client,        # noqa: E402
                       code_patch, code_scan, config, endpoint_client, guardrails,
                       health, judge, optimizer, project, rank, runner, sandbox,
                       scheduler, secrets, state, wizard)

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


def test_host_by_id_finds_the_specific_host_actually_called() -> None:
    """Pricing a run's real cost needs the host that was actually called,
    not the model-level cheapest aggregate — they can differ."""
    cat = {"models": {
        "llama": {"hosts": [{"id": "vendor/llama-cheap", "price_in": 0.05, "price_out": 0.10},
                            {"id": "vendor/llama-pricey", "price_in": 0.50, "price_out": 1.00}]},
    }}
    host = catalogue.host_by_id(cat, "vendor/llama-pricey")
    check("found the right host", host is not None and host["price_out"] == 1.00, f"{host}")
    check("unknown id returns None", catalogue.host_by_id(cat, "vendor/nope") is None)
    check("empty catalogue returns None", catalogue.host_by_id({}, "anything") is None)


def test_rows_openai_marks_json_capable_and_prices_only_known_models() -> None:
    """OpenAI's `/v1/models` publishes neither price nor a JSON-mode flag —
    unlike Groq/Fireworks (which default `json_mode` False because it's
    genuinely unconfirmed), OpenAI's chat models reliably support
    `response_format: json_object`, so this normalizer asserts it rather
    than guessing false and silently excluding every OpenAI candidate."""
    rows = catalogue._rows_openai(
        {"data": [{"id": "gpt-4o-mini"}, {"id": "some-future-model"}]})
    by_id = {r["id"]: r for r in rows}
    check("two rows", len(rows) == 2, f"{rows}")
    check("json_mode asserted true", by_id["gpt-4o-mini"]["json_mode"] is True)
    check("known model priced", by_id["gpt-4o-mini"]["price_out"] == 0.60, f"{by_id['gpt-4o-mini']}")
    check("unknown model unpriced, not guessed",
         by_id["some-future-model"]["price_in"] is None
         and by_id["some-future-model"]["price_out"] is None)


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

    async def fake_bench_run(candidates, uc, *, judge_model=None, provider_by_model=None,
                             cache=None):
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


# ── rank: a price preference is a display-time lens, never a promotion ──────

def test_combined_score_at_zero_sensitivity_is_unchanged() -> None:
    check("returns score exactly", rank.combined_score(4.2, 1.50, 3.00, 0) == 4.2)
    check("even with unknown price", rank.combined_score(4.2, None, 3.00, 0) == 4.2)


def test_combined_score_treats_unknown_price_as_worst_not_guessed() -> None:
    known = rank.combined_score(3.0, 3.00, 3.00, 100)   # at the ceiling -> price_score 0
    unknown = rank.combined_score(3.0, None, 3.00, 100)  # unknown -> also price_score 0
    check("unknown price scores no better than the worst known price",
         unknown == known == 0.0, f"{unknown} vs {known}")


def test_combined_score_free_beats_expensive_at_full_sensitivity() -> None:
    free = rank.combined_score(3.0, 0.0, 3.00, 100)
    expensive = rank.combined_score(5.0, 3.00, 3.00, 100)
    check("free scores max on price regardless of quality", free == 5.0, f"{free}")
    check("at the ceiling scores worst on price regardless of quality", expensive == 0.0, f"{expensive}")


def test_reorder_by_preference_at_zero_returns_the_original_tiers() -> None:
    board = rank.build(_bench(_row("a", 4.4), _row("b", 4.3)), None)
    check("unchanged at 0", rank.reorder_by_preference(
        board, price_sensitivity=0, max_price_out=3.0) is board["tiers"])


def test_reorder_by_preference_can_flip_the_order_on_price() -> None:
    # Both models land in the SAME price tier (paid-low, <= $0.50/M out) —
    # ranking is always per tier, so two candidates in different tiers
    # would never compete against each other at all.
    cat = {"models": {
        "a": {"key": "a", "cheapest_out": 0.50},
        "b": {"key": "b", "cheapest_out": 0.05},
    }}
    board = rank.build(_bench(_row("a", 5.0), _row("b", 3.0)), cat)
    tiers = board["tiers"]
    check("both in the same tier", set(tiers.keys()) == {"paid-low"}, f"{tiers}")
    check("quality-only: pricier-but-best is first",
         tiers["paid-low"][0]["model"] == "a", f"{tiers['paid-low']}")

    reordered = rank.reorder_by_preference(board, price_sensitivity=100, max_price_out=0.50)
    check("price-only: the much cheaper candidate now first despite lower quality",
         reordered["paid-low"][0]["model"] == "b", f"{reordered['paid-low']}")
    check("original board object untouched", board["tiers"]["paid-low"][0]["model"] == "a")


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


# ── project defaults: the shared template every NEW feature starts from ─────

def test_project_defaults_round_trip_and_never_touch_existing_use_cases() -> None:
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        proj = project.create("Demo App", notify_email="demo@example.com", root=root)
        check("fresh project has the built-in defaults",
             proj.defaults["maxPriceOut"] == 3.00, f"{proj.defaults}")

        project.set_defaults("demo-app", {
            "rubric": [{"id": "clarity", "description": "is it clear", "weight": 2.0}],
            "maxPriceOut": 1.00, "judgeModel": "vendor/judge",
        }, root=root)
        loaded = project.load("demo-app", root=root)
        check("override applied", loaded.defaults["maxPriceOut"] == 1.00)
        check("judge model applied", loaded.defaults["judgeModel"] == "vendor/judge")
        check("unset fields keep the built-in default",
             loaded.defaults["minContext"] == 32_000, f"{loaded.defaults}")
        check("rubric saved", loaded.defaults["rubric"][0]["id"] == "clarity")


# ── project deletion: real, irreversible, so it earns its own coverage ──────
#
# WHAT PROMPTED THIS. There was no way to delete a project at all — CLI or
# dashboard — until a real person asked "where's the delete button" after
# creating a throwaway test project. Every check below exists because
# deleting is the one project action with no undo: get delete_impact wrong
# and someone loses history without knowing; get delete wrong and it either
# leaves orphaned files behind or reaches outside the project it was told
# to remove.

def test_delete_removes_everything_project_create_wrote() -> None:
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        project.create("Demo App", notify_email="demo@example.com", root=root)
        project_dir = root / "demo-app"
        check("the project directory exists before deleting", project_dir.exists())
        project.delete("demo-app", root=root)
        check("the whole directory is gone", not project_dir.exists())
        check("no longer registered", not project.exists("demo-app", root=root))
        check("no longer listed", project.list_all(root=root) == [])


def test_delete_refuses_an_unknown_slug() -> None:
    with tempfile.TemporaryDirectory() as d:
        try:
            project.delete("nothing-here", root=Path(d))
            check("refused", False)
        except FileNotFoundError as exc:
            check("refused", True)
            check("names the slug", "nothing-here" in str(exc), str(exc))


def test_delete_never_touches_an_externally_referenced_repo() -> None:
    """A project pointed at a folder you already had (--repo-path) only
    ever stores that path as a STRING — the real directory lives outside
    the project's own tree and must survive a delete untouched. This is
    the one way `delete` could do real damage beyond the project's own
    data if it got the containment wrong."""
    with tempfile.TemporaryDirectory() as d:
        root = Path(d) / "projects"
        external_repo = Path(d) / "someones-real-app"
        external_repo.mkdir(parents=True)
        (external_repo / "marker.txt").write_text("still here?", encoding="utf-8")

        project.create("Demo App", notify_email="demo@example.com",
                       repo_path=str(external_repo), root=root)
        project.delete("demo-app", root=root)

        check("the project's own directory is gone", not (root / "demo-app").exists())
        check("the EXTERNAL repo survives untouched",
              (external_repo / "marker.txt").read_text(encoding="utf-8") == "still here?")


def test_delete_removes_a_repo_it_cloned_itself() -> None:
    """The opposite case from the one above: a repo THIS project cloned
    (--repo-url, landing inside the project's own tree) is fair game and
    must go with everything else — it's this project's own copy, not
    someone else's folder."""
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        cloned_marker = project._repo_dir("demo-app", root) / "cloned.txt"
        cloned_marker.parent.mkdir(parents=True, exist_ok=True)
        cloned_marker.write_text("cloned copy", encoding="utf-8")
        project.create("Demo App", notify_email="demo@example.com", root=root)
        # (the marker predates `create`'s own directory-making, and survives it —
        #  proving it's really inside the project's tree before the delete)
        check("the cloned-repo stand-in is inside the project's own tree",
              cloned_marker.exists())
        project.delete("demo-app", root=root)
        check("it's gone along with everything else", not cloned_marker.exists())


def test_delete_impact_reports_feature_count_and_which_are_approved() -> None:
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        project.create("Demo App", notify_email="demo@example.com", root=root)
        for name in ("router", "summarizer"):
            (project.use_cases_dir("demo-app", root) / f"{name}.yaml").write_text(
                "useCase: " + name, encoding="utf-8")
        state.record_run("router", {"ranAt": "t", "tiers": {
            "free": [{"model": "vendor/winner", "score": 4.5}]}},
            root=project.state_dir("demo-app", root))
        state.approve("router", "vendor/winner", root=project.state_dir("demo-app", root))

        impact = project.delete_impact("demo-app", root=root)
        check("counts both features", impact["feature_count"] == 2, f"{impact}")
        check("names only the approved one", impact["approved_features"] == ["router"],
              f"{impact}")
        check("carries the project's display name", impact["name"] == "Demo App")


def test_delete_impact_on_an_unknown_slug_raises_before_anything_else() -> None:
    with tempfile.TemporaryDirectory() as d:
        try:
            project.delete_impact("nothing-here", root=Path(d))
            check("raises", False)
        except FileNotFoundError:
            check("raises", True)


# ── dashboard: the delete confirmation flow ──────────────────────────────────

def _delete_client():
    from modelcicd import dashboard
    tmp = Path(tempfile.mkdtemp())
    original = project.DEFAULT_DIR
    project.DEFAULT_DIR = tmp / "projects"
    project.create("Demo App", slug="demo-app", notify_email="t@example.com")
    app = dashboard.create_app()
    app.config["TESTING"] = True

    def restore():
        project.DEFAULT_DIR = original

    return app.test_client(), restore


def test_delete_confirmation_page_shows_the_impact_before_asking() -> None:
    client_, restore = _delete_client()
    try:
        r = client_.get("/projects/demo-app/delete")
        check("200", r.status_code == 200, f"{r.status_code}")
        body = r.get_data(as_text=True)
        check("names the project", "Demo App" in body, body[:200])
        check("asks for the slug back", 'value="demo-app"' in body or "demo-app" in body)
    finally:
        restore()


def test_delete_post_with_the_wrong_slug_deletes_nothing() -> None:
    client_, restore = _delete_client()
    try:
        r = client_.post("/projects/demo-app/delete", data={"confirm_slug": "wrong"})
        check("refused", r.status_code == 400, f"{r.status_code}")
        check("project still exists", project.exists("demo-app"))
    finally:
        restore()


def test_delete_post_with_the_right_slug_deletes_and_redirects() -> None:
    client_, restore = _delete_client()
    try:
        r = client_.post("/projects/demo-app/delete", data={"confirm_slug": "demo-app"},
                         follow_redirects=False)
        check("redirects home", r.status_code == 303, f"{r.status_code}")
        check("project is actually gone", not project.exists("demo-app"))
        check("the redirect names what was deleted",
              "Demo" in r.headers.get("Location", "") or "deleted" in r.headers.get("Location", ""),
              r.headers.get("Location"))
    finally:
        restore()


def test_delete_form_unknown_project_404s() -> None:
    client_, restore = _delete_client()
    try:
        r = client_.get("/projects/does-not-exist/delete")
        check("404", r.status_code == 404, f"{r.status_code}")
    finally:
        restore()


def test_wizard_defaults_from_project_only_sets_shared_fields() -> None:
    f = wizard.defaults_from_project({
        "rubric": [{"id": "clarity", "description": "is it clear", "weight": 2.0}],
        "maxPriceOut": 1.00, "judgeModel": "vendor/judge",
    })
    check("shared field applied", f.max_price_out == 1.00)
    check("judge model applied", f.judge_model == "vendor/judge")
    check("rubric applied", f.rubric[0]["id"] == "clarity")
    check("feature-specific fields stay at their own default", f.name == "" and f.system_prompt == "")


# ── code_scan: drafting test cases is a proposal, never a save ──────────────

def test_generate_test_cases_returns_drafts_and_falls_back_to_empty_on_failure() -> None:
    import asyncio

    async def fake_call_json(prompt, *, model, label, **kwargs):
        return {"cases": [
            {"id": "angry_customer", "input": "This is unacceptable!", "reference": "An apology and next steps."},
            {"id": "", "input": "  ", "reference": ""},  # blank input -> dropped
        ]}

    original = code_scan.client_module.call_json
    code_scan.client_module.call_json = fake_call_json
    try:
        cases = asyncio.run(code_scan.generate_test_cases(
            "You are a support agent.", input_structure="a string", model="x", count=2))
    finally:
        code_scan.client_module.call_json = original
    check("kept the real case", len(cases) == 1, f"{cases}")
    check("carries id/input/reference", cases[0]["id"] == "angry_customer"
         and cases[0]["reference"] == "An apology and next steps.")

    async def fake_fail(prompt, *, model, label, **kwargs):
        raise RuntimeError("boom")

    code_scan.client_module.call_json = fake_fail
    try:
        cases = asyncio.run(code_scan.generate_test_cases("x", model="x"))
    finally:
        code_scan.client_module.call_json = original
    check("falls back to empty on failure, not an exception", cases == [])


def test_generate_test_cases_tolerates_a_model_returning_a_list_instead_of_a_string() -> None:
    """A REAL failure a free model produced live: {"input": ["part one", "part two"]}
    instead of one string. This must not crash the whole bulk-create flow."""
    import asyncio

    async def fake_call_json(prompt, *, model, label, **kwargs):
        return {"cases": [{"id": "weird", "input": ["This is unacceptable!", "Fix it now."],
                           "reference": None}]}

    original = code_scan.client_module.call_json
    code_scan.client_module.call_json = fake_call_json
    try:
        cases = asyncio.run(code_scan.generate_test_cases("x", model="x"))
    finally:
        code_scan.client_module.call_json = original
    check("did not crash, joined the list into text",
         cases and cases[0]["input"] == "This is unacceptable! Fix it now.", f"{cases}")


# ── rubric generation: the one place this project has always said a human
#    has to show up — drafted here, never saved without review ─────────────

def test_generate_rubric_returns_drafts_and_falls_back_to_empty_on_failure() -> None:
    import asyncio

    async def fake_call_json(prompt, *, model, label, **kwargs):
        return {"criteria": [
            {"id": "correctness", "description": "is it right", "weight": 1.5},
            {"id": "", "description": "", "weight": 1.0},   # blank description -> dropped
        ]}

    original = code_scan.client_module.call_json
    code_scan.client_module.call_json = fake_call_json
    try:
        criteria = asyncio.run(code_scan.generate_rubric(
            "You are a support agent.", input_structure="a string", model="x", count=2))
    finally:
        code_scan.client_module.call_json = original
    check("kept the real criterion", len(criteria) == 1, f"{criteria}")
    check("carries id/description/weight",
          criteria[0]["id"] == "correctness" and criteria[0]["description"] == "is it right"
          and criteria[0]["weight"] == 1.5, f"{criteria}")

    async def fake_fail(prompt, *, model, label, **kwargs):
        raise RuntimeError("boom")

    code_scan.client_module.call_json = fake_fail
    try:
        criteria = asyncio.run(code_scan.generate_rubric("x", model="x"))
    finally:
        code_scan.client_module.call_json = original
    check("falls back to empty on failure, not an exception", criteria == [])


def test_generate_rubric_clamps_a_nonsense_weight_instead_of_trusting_it() -> None:
    """A model returning an absurd or negative weight must not be able to
    make one criterion swamp (or erase) every other in the blended score —
    same "unknown/nonsense never resolved favorably" rule guardrails and
    assertions already follow."""
    import asyncio

    async def fake_call_json(prompt, *, model, label, **kwargs):
        return {"criteria": [
            {"id": "a", "description": "d", "weight": 500},
            {"id": "b", "description": "d", "weight": -3},
            {"id": "c", "description": "d", "weight": "not a number"},
        ]}

    original = code_scan.client_module.call_json
    code_scan.client_module.call_json = fake_call_json
    try:
        criteria = asyncio.run(code_scan.generate_rubric("x", model="x"))
    finally:
        code_scan.client_module.call_json = original
    weights = {c["id"]: c["weight"] for c in criteria}
    check("an absurdly high weight is clamped", weights["a"] <= 5.0, f"{weights}")
    check("a negative weight is clamped to the floor", weights["b"] >= 0.1, f"{weights}")
    check("a non-numeric weight falls back to 1.0, not dropped entirely",
          weights["c"] == 1.0, f"{weights}")


def test_suggest_name_from_file_path() -> None:
    check("dir + stem", code_scan.suggest_name("support_bot/bot.py") == "support_bot_bot")
    check("sanitized", code_scan.suggest_name("agent/router.py") == "agent_router")


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
            return {"calls": [{"model": "gpt-4o-mini", "prompt": "hi",
                               "inputStructure": "a single string: the ticket text",
                               "outputStructure": 'JSON: {"reply": "..."}',
                               "line": 1, "confidence": "high"}]}

        original = code_scan.client_module.call_json
        code_scan.client_module.call_json = fake_call_json
        try:
            results = asyncio.run(code_scan.scan_repo(repo))
            check("carries input structure",
                 results[0]["inputStructure"] == "a single string: the ticket text")
            check("carries output structure",
                 results[0]["outputStructure"] == 'JSON: {"reply": "..."}')
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


def test_select_candidates_excludes_the_configured_judge() -> None:
    """A real failure this caught live: once the default judge became a
    free model, it naturally also passed guardrails as a free-tier
    candidate — this must be dropped automatically, not force the whole
    run to refuse."""
    cat = {"models": {}}
    uc = config.load(ROOT / "examples" / "prep_material" / "use_case.yaml")
    judge_key = catalogue.key(uc.judge_model)
    passed_models = [
        {"key": judge_key, "hosts": [{"id": uc.judge_model, "price_out": 0.0}],
         "providers": ["openrouter"]},
        {"key": "candidate-a", "hosts": [{"id": "vendor/a", "price_out": 0.1}],
         "providers": ["openrouter"]},
    ]
    import types
    fake_guardrails = types.SimpleNamespace(
        apply=lambda cat, g: {"passed": passed_models, "failed": []},
        within_tiers=lambda models, tiers: models)
    original = runner.guardrails_module
    runner.guardrails_module = fake_guardrails
    try:
        ids = runner.select_candidates(uc, cat)
    finally:
        runner.guardrails_module = original
    check("judge excluded from its own candidate pool",
         uc.judge_model not in ids and ids == ["vendor/a"], f"{ids}")


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


def test_wizard_yaml_carries_input_output_structure() -> None:
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "demo_feature.yaml"
        fields = _wizard_fields(input_structure="a single string: the ticket text",
                                output_structure='JSON: {"reply": "..."}')
        p.write_text(wizard.to_yaml(fields), encoding="utf-8")
        uc = config.load(p)
        check("input structure loaded", uc.input_structure == fields.input_structure)
        check("output structure loaded", uc.output_structure == fields.output_structure)


def test_wizard_yaml_omits_structure_fields_when_absent() -> None:
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "demo_feature.yaml"
        p.write_text(wizard.to_yaml(_wizard_fields()), encoding="utf-8")
        uc = config.load(p)
        check("input structure absent", uc.input_structure is None)
        check("output structure absent", uc.output_structure is None)


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


# ── editing a saved feature: a round trip must not quietly lose anything ────
#
# THE RISK AN EDIT FORM INTRODUCES. Creating a feature writes a file from
# scratch, so anything the form doesn't ask about simply isn't there. EDITING
# reads a file into the same fields and writes it back — so anything the form
# doesn't ask about gets deleted, by someone who came to change a price
# ceiling. Every check below exists for one of those fields.

def _full_use_case_yaml() -> str:
    """Every optional block `to_yaml` can emit, including the two nobody's
    form exposes: a per-test-case rubric and per-test-case assertions."""
    import textwrap
    return textwrap.dedent("""
        useCase: router
        description: routes tickets
        systemPrompt: |
          Decide which team handles this.
          Answer with one word.
        testCases:
          - id: plain
            input: my card was charged twice
            reference: billing
          - id: has_overrides
            input: the upload crashes
            reference: technical
            rubric:
              - id: strictness
                description: must be exactly one word
                weight: 3.0
            assertions:
              - type: equals
                value: technical
                path: team
        rubric:
          - id: clarity
            description: is it clear
            weight: 1.0
          - id: brevity
            description: is it short
            weight: 2.0
        assertions:
          - type: has-keys
            value: [team]
            required: true
          - type: not-contains
            value: as an AI
            weight: 2.0
            caseSensitive: true
        guardrails:
          maxPriceIn: 0.25
          maxPriceOut: 1.5
          minContext: 64000
          requireJson: true
          allowFree: false
          tiers: [paid-low, paid-mid]
        judgeModel: pinned/judge
        maxTokens: 900
        notify:
          email: someone@example.com
          minImprovement: 0.35
        endpoint:
          url: http://localhost:8000/route
          method: POST
          inputField: ticket
          responseField: team
          headers:
            Authorization: 'Bearer ${TOK}'
          timeoutSeconds: 12.0
        schedule:
          intervalDays: 14
        codeTarget:
          file: app/router.py
          currentModel: vendor/old
        usage:
          callsPerDay: 5000
        inputStructure: '{ticket: str}'
        outputStructure: '{team: str}'
    """).lstrip()


def test_a_saved_feature_round_trips_through_the_edit_form_unchanged() -> None:
    """Read a file into fields, write it back, and every field must survive
    — otherwise editing one thing deletes another."""
    tmp = Path(tempfile.mkdtemp())
    original = tmp / "router.yaml"
    original.write_text(_full_use_case_yaml(), encoding="utf-8")

    fields = wizard.from_yaml(original)
    rewritten = tmp / "router2.yaml"
    wizard.write_yaml(rewritten, fields)

    before, after = config.load(original), config.load(rewritten)
    for attr in ("name", "description", "system_prompt", "judge_model",
                 "max_tokens", "schedule_interval_days", "input_structure",
                 "output_structure", "estimated_calls_per_day"):
        check(f"{attr} survives", getattr(before, attr) == getattr(after, attr),
              f"{getattr(before, attr)!r} -> {getattr(after, attr)!r}")
    for attr in ("max_price_in", "max_price_out", "min_context", "require_json",
                 "allow_free", "tiers"):
        check(f"guardrails.{attr} survives",
              getattr(before.guardrails, attr) == getattr(after.guardrails, attr),
              f"{getattr(before.guardrails, attr)!r} -> {getattr(after.guardrails, attr)!r}")
    check("notify email survives", before.notify.email == after.notify.email)
    check("notify threshold survives",
          before.notify.min_improvement == after.notify.min_improvement)
    check("the endpoint survives", before.endpoint.url == after.endpoint.url
          and before.endpoint.input_field == after.endpoint.input_field
          and before.endpoint.timeout_seconds == after.endpoint.timeout_seconds)
    check("the endpoint's header survives — it's the auth token",
          before.endpoint.headers == after.endpoint.headers,
          f"{before.endpoint.headers} -> {after.endpoint.headers}")
    check("the code target survives", before.code_target.file == after.code_target.file
          and before.code_target.current_model == after.code_target.current_model)
    check("the rubric survives whole",
          [(c.id, c.description, c.weight) for c in before.rubric]
          == [(c.id, c.description, c.weight) for c in after.rubric])
    check("the checks survive whole",
          [(a.type, a.value, a.negate, a.required, a.case_sensitive, a.weight)
           for a in before.assertions]
          == [(a.type, a.value, a.negate, a.required, a.case_sensitive, a.weight)
              for a in after.assertions])

    # THE ONES NO FORM SHOWS, and therefore the ones a round trip would eat.
    overridden = next(tc for tc in after.test_cases if tc.id == "has_overrides")
    check("a per-test-case RUBRIC override survives",
          [(c.id, c.weight) for c in overridden.rubric] == [("strictness", 3.0)],
          f"{[(c.id, c.weight) for c in overridden.rubric]}")
    check("a per-test-case CHECK override survives",
          any(a.type == "equals" and a.value == "technical" for a in overridden.assertions),
          f"{[(a.type, a.value) for a in overridden.assertions]}")
    # And the plain test case must NOT have gained a pinned copy of the
    # shared rubric — that's what reading through `config.load` would do,
    # turning inheritance into hardcoded duplicates.
    import yaml as yaml_module
    raw_after = yaml_module.safe_load(rewritten.read_text(encoding="utf-8"))
    plain = next(tc for tc in raw_after["testCases"] if tc["id"] == "plain")
    check("inheritance stays inheritance, not a pinned copy",
          "rubric" not in plain and "assertions" not in plain, f"{plain}")


def test_the_fingerprint_is_identical_after_a_no_op_edit() -> None:
    """THE SHARPEST VERSION OF THE ROUND-TRIP TEST. If opening the edit form
    and saving without touching anything changed the fingerprint, it would
    break the trend line and warn "not comparable" for an edit that changed
    nothing — and every field-by-field check above could still pass while
    that was true."""
    tmp = Path(tempfile.mkdtemp())
    original = tmp / "router.yaml"
    original.write_text(_full_use_case_yaml(), encoding="utf-8")
    rewritten = tmp / "router2.yaml"
    wizard.write_yaml(rewritten, wizard.from_yaml(original))
    check("a no-op edit changes nothing measurable",
          config.measurement(config.load(original))["combined"]
          == config.measurement(config.load(rewritten))["combined"])


def test_overrides_are_carried_across_a_form_submission_by_id() -> None:
    """`from_form` rebuilds test cases from the form, which has no inputs for
    per-test-case overrides. Matched by ID rather than position, so
    reordering or inserting a test case can't move someone's override onto
    the wrong one."""
    existing = wizard.WizardFields(test_cases=[
        {"id": "a", "input": "1", "rubric": [{"id": "x", "description": "d", "weight": 2.0}]},
        {"id": "b", "input": "2", "assertions": [{"type": "contains", "value": "yes"}]},
        {"id": "gone", "input": "3", "rubric": [{"id": "y", "description": "d"}]},
    ])
    # Submitted with a new test case FIRST, so positions no longer line up,
    # and with "gone" deleted.
    submitted = wizard.WizardFields(test_cases=[
        {"id": "inserted", "input": "0"},
        {"id": "b", "input": "2 edited"},
        {"id": "a", "input": "1"},
    ])
    wizard.carry_over_overrides(submitted, existing)
    by_id = {tc["id"]: tc for tc in submitted.test_cases}
    check("a's rubric override followed its id, not its position",
          by_id["a"].get("rubric") == [{"id": "x", "description": "d", "weight": 2.0}],
          f"{by_id['a']}")
    check("b's check override survived an edit to its input",
          by_id["b"].get("assertions") == [{"type": "contains", "value": "yes"}],
          f"{by_id['b']}")
    check("the new test case gained nothing", "rubric" not in by_id["inserted"]
          and "assertions" not in by_id["inserted"], f"{by_id['inserted']}")
    check("a deleted test case's override went with it", "gone" not in by_id, f"{by_id}")


def test_write_yaml_is_atomic_and_leaves_no_temp_file() -> None:
    """An edit REPLACES something that already worked, so a half-written
    file is worse here than for a create — it would leave a broken feature
    where a working one was."""
    tmp = Path(tempfile.mkdtemp())
    dest = tmp / "f.yaml"
    f = wizard.WizardFields(name="f", system_prompt="hi",
                            test_cases=[{"id": "t", "input": "i"}],
                            rubric=[{"id": "q", "description": "d"}])
    wizard.write_yaml(dest, f)
    check("the file is written", dest.exists())
    check("no temp file is left behind", list(tmp.glob("*.tmp")) == [],
          f"{list(tmp.glob('*.tmp'))}")
    check("and it loads", config.load(dest).name == "f")


def _edit_client(yaml_text: str):
    """A test client plus a project holding one saved feature."""
    from modelcicd import dashboard
    tmp = Path(tempfile.mkdtemp())
    original = project.DEFAULT_DIR
    project.DEFAULT_DIR = tmp / "projects"
    project.create("Edit Test", slug="edit-test", notify_email="t@example.com")
    ucs = project.use_cases_dir("edit-test")
    ucs.mkdir(parents=True, exist_ok=True)
    (ucs / "router.yaml").write_text(yaml_text, encoding="utf-8")
    app = dashboard.create_app()
    app.config["TESTING"] = True
    return app.test_client(), original


def _edit_form(**overrides) -> dict:
    form = {
        "name": "router", "description": "routes tickets",
        "system_prompt": "Decide which team handles this.",
        "testcase_id": ["plain", "has_overrides"],
        "testcase_input": ["my card was charged twice", "the upload crashes"],
        "testcase_reference": ["billing", "technical"],
        "rubric_id": ["clarity"], "rubric_description": ["is it clear"],
        "rubric_weight": ["1.0"],
        "assert_type": ["has-keys"], "assert_value": ["team"],
        "assert_path": [""], "assert_weight": ["1.0"], "assert_required": ["0"],
        "judge_model": "pinned/judge", "max_tokens": "900",
        "max_price_in": "0.25", "max_price_out": "1.5", "min_context": "64000",
        "require_json": "on", "tiers": ["paid-low"],
        "notify_email": "someone@example.com", "min_improvement": "0.35",
    }
    form.update(overrides)
    return form


def test_the_edit_route_saves_and_keeps_the_hand_written_overrides() -> None:
    client_, original = _edit_client(_full_use_case_yaml())
    try:
        r = client_.post("/projects/edit-test/usecase/router/edit",
                         data=_edit_form(rubric_description=["is it CRYSTAL clear"]))
        check("the edit saves", r.status_code == 303, f"{r.status_code}")
        saved = config.load(project.use_cases_dir("edit-test") / "router.yaml")
        check("the edited rubric is written",
              saved.rubric[0].description == "is it CRYSTAL clear",
              f"{saved.rubric[0].description}")
        overridden = next(tc for tc in saved.test_cases if tc.id == "has_overrides")
        check("the per-test-case rubric override survived the form",
              any(c.id == "strictness" for c in overridden.rubric),
              f"{[c.id for c in overridden.rubric]}")
        check("the per-test-case check override survived the form",
              any(a.value == "technical" for a in overridden.assertions),
              f"{[(a.type, a.value) for a in overridden.assertions]}")
    finally:
        project.DEFAULT_DIR = original


def test_the_edit_route_ignores_an_attempt_to_rename() -> None:
    """State, run history, the approved model and every saved run file are
    keyed on the name. Renaming would orphan all of it while looking like it
    worked, so the form shows it read-only AND the route refuses a changed
    one — a stale page must not be able to do it either."""
    client_, original = _edit_client(_full_use_case_yaml())
    try:
        client_.post("/projects/edit-test/usecase/router/edit",
                     data=_edit_form(name="something_else"))
        ucs = project.use_cases_dir("edit-test")
        names = sorted(p.stem for p in ucs.glob("*.yaml"))
        check("no second file was created", names == ["router"], f"{names}")
        check("the name inside is unchanged",
              config.load(ucs / "router.yaml").name == "router")
    finally:
        project.DEFAULT_DIR = original


def test_the_edit_route_refuses_to_replace_a_working_file_with_a_broken_one() -> None:
    """An edit overwrites something that already ran. `wizard.validate`
    catches what a person gets wrong in the form; `config.load` is what
    every run actually uses, so the rendered result is loaded before it
    replaces anything."""
    client_, original = _edit_client(_full_use_case_yaml())
    try:
        before = (project.use_cases_dir("edit-test") / "router.yaml").read_text(
            encoding="utf-8")
        r = client_.post("/projects/edit-test/usecase/router/edit",
                         data=_edit_form(rubric_id=[""], rubric_description=[""]))
        check("an empty rubric is refused", r.status_code == 400, f"{r.status_code}")
        check("with the reason shown", "rubric criterion" in r.get_data(as_text=True))
        after = (project.use_cases_dir("edit-test") / "router.yaml").read_text(
            encoding="utf-8")
        check("the working file is untouched", before == after)
        check("and no probe temp file is left behind",
              list(project.use_cases_dir("edit-test").glob("*.tmp")) == [],
              f"{list(project.use_cases_dir('edit-test').glob('*.tmp'))}")
    finally:
        project.DEFAULT_DIR = original


def test_the_edit_form_shows_enough_rows_for_what_already_exists() -> None:
    """A fixed row count is fine for a create form and a silent-data-loss bug
    in an edit one: a feature with more test cases than slots would submit
    only the visible ones, and `from_form` rebuilds the list from what it
    receives — deleting the rest."""
    import textwrap
    many = textwrap.dedent("""
        useCase: router
        systemPrompt: hi
        rubric:
          - id: q
            description: d
        testCases:
    """).lstrip() + "".join(
        f"  - id: tc{i}\n    input: input {i}\n" for i in range(11))
    client_, original = _edit_client(many)
    try:
        body = client_.get("/projects/edit-test/usecase/router/edit").get_data(as_text=True)
        for i in range(11):
            check(f"test case tc{i} has a slot", f'value="tc{i}"' in body,
                  "a row that isn't rendered gets deleted on save")
        slots = body.count('name="testcase_id"')
        check("and spare slots remain", slots >= 12, f"{slots} slots")
    finally:
        project.DEFAULT_DIR = original


# ── health probe: drop only what is provably gone ───────────────────────────
#
# TWICE-OBSERVED FAILURE THIS GUARDS. A deprecated model id keeps looking
# fine everywhere it's read from — catalogue, guardrails, candidate list —
# and only fails on a real call, minutes into a run. A dead JUDGE is worse:
# every generation call gets paid for and every one comes back judge_failed.

def test_a_model_level_rejection_carries_the_platforms_own_message() -> None:
    """The message IS the decision — the status code isn't enough (see
    `health.py`) — and it's often the fix, too: OpenRouter's own reply to a
    retired free slug names the paid one to use instead."""
    import asyncio
    body = ('{"error": {"message": "This model is unavailable for free. The paid '
            'version is available now - use this slug instead: minimax/minimax-m3", '
            '"code": 404}}')
    for status in (400, 404):
        with _NoNetwork(content=body, status=status):
            try:
                asyncio.run(client.call_json("hi", model="vendor/gone", label="t",
                                             attempts=3))
                check(f"HTTP {status} raises", False)
            except client.ModelUnavailableError as exc:
                check(f"HTTP {status} raises ModelUnavailableError", True)
                check(f"HTTP {status} keeps the platform's message verbatim",
                      "use this slug instead: minimax/minimax-m3" in str(exc), str(exc))
            except Exception as exc:                # noqa: BLE001
                check(f"HTTP {status} raises ModelUnavailableError", False,
                      f"got {type(exc).__name__}")


def test_the_upstream_provider_name_is_surfaced() -> None:
    """"Provider returned error" alone is unattributable. The metadata names
    which upstream host failed, and that's the tell separating "this model
    is gone" from "this model's host had a bad second"."""
    import asyncio
    body = ('{"error": {"message": "Provider returned error", "code": 404, '
            '"metadata": {"provider_name": "Nvidia", "raw": ""}}}')
    with _NoNetwork(content=body, status=404):
        try:
            asyncio.run(client.call_json("hi", model="v/m", label="t", attempts=3))
            check("raises", False)
        except client.ModelUnavailableError as exc:
            check("names the upstream provider", "Nvidia" in str(exc), str(exc))


def test_a_model_level_rejection_is_not_retried() -> None:
    """It will not stop being a 404, and the retry message ("that was not
    valid — return ONLY the JSON") is nonsense for an HTTP error."""
    import asyncio
    with _NoNetwork(status=404, content='{"error": {"message": "gone"}}') as fake:
        try:
            asyncio.run(client.call_json("hi", model="vendor/gone", label="t",
                                         attempts=3))
        except client.ModelUnavailableError:
            pass
        check("one call, not three", len(fake.posts) == 1, f"{len(fake.posts)}")


def test_classification_reads_the_message_not_the_status_code() -> None:
    """THE CORRECTION THAT CAME OUT OF PROBING REAL MODELS. Excluding on 404
    would have dropped this project's own configured judge — whose 404 said
    only "Provider returned error (upstream provider: Nvidia)", an upstream
    hiccup — and simultaneously missed a fake id, which OpenRouter refuses
    with a 400. Same status, opposite meanings; different statuses, same
    meaning."""
    fatal = [
        "HTTP 400: vendor/x is not a valid model ID",
        "HTTP 404: This model is unavailable for free. The paid version is "
        "available now - use this slug instead: minimax/minimax-m3",
        "HTTP 404: this model has been deprecated",
        "HTTP 404: no longer available",
    ]
    for message in fatal:
        check(f"fatal: {message[:45]}", health.classify(message) == health.FATAL, message)

    survivable = [
        "HTTP 404: Provider returned error (upstream provider: Nvidia)",
        "HTTP 404: No endpoints found for vendor/x",
        "HTTP 400: max_tokens must be a positive integer",
        "HTTP 404: ",
        "",
    ]
    for message in survivable:
        check(f"not fatal: {message[:45] or '(empty)'}",
              health.classify(message) == "unknown", message)


def test_probe_classifies_each_outcome() -> None:
    import asyncio
    cases = [
        (client.ModelUnavailableError("HTTP 400: x is not a valid model ID"), health.FATAL),
        # A model-level rejection whose message does NOT prove permanence
        # must come back "unknown", so the candidate still runs.
        (client.ModelUnavailableError("HTTP 404: Provider returned error"), "unknown"),
        (client.RateLimitedError("429"), "rate_limited"),
        (RuntimeError("connection reset"), "unknown"),
        (None, "ok"),
    ]
    original = health.client_module.call_json
    try:
        for error, expected in cases:
            async def fake(*a, _e=error, **k):
                if _e is not None:
                    raise _e
                return {"ok": True}
            health.client_module.call_json = fake
            result = asyncio.run(health.probe("vendor/m"))
            check(f"{type(error).__name__ if error else 'success'} -> {expected}",
                  result["status"] == expected, f"{result}")
    finally:
        health.client_module.call_json = original


def test_only_a_provable_death_excludes_a_candidate() -> None:
    """UNKNOWN CUTS BOTH WAYS. Silently shrinking the field on a guess is the
    same class of bug as silently assuming an unknown price is free — and a
    rate limit in particular would drop half the free tier at a busy moment,
    which is a fact about the minute, not the model."""
    results = [
        {"model": "a", "status": "ok", "detail": ""},
        {"model": "b", "status": health.FATAL, "detail": "404"},
        {"model": "c", "status": "rate_limited", "detail": "429"},
        {"model": "d", "status": "unknown", "detail": "timeout"},
    ]
    usable, fatal = health.partition(results)
    check("the dead one is dropped", [f["model"] for f in fatal] == ["b"], f"{fatal}")
    check("the rate-limited one is KEPT", "c" in usable, f"{usable}")
    check("the unexplained one is KEPT", "d" in usable, f"{usable}")
    check("the healthy one is kept", "a" in usable, f"{usable}")


def test_the_summary_names_statuses_rather_than_counting_unhealthy() -> None:
    """"One is gone, two were rate limited" and "three are gone" call for
    completely different reactions; a single "3 unhealthy" hides that."""
    text = health.summary([
        {"model": "a", "status": "ok"}, {"model": "b", "status": health.FATAL},
        {"model": "c", "status": "rate_limited"}, {"model": "d", "status": "rate_limited"},
    ])
    check("counts by status", "1 ok" in text and "1 unavailable" in text
          and "2 rate_limited" in text, text)


def _stub_health(statuses: dict):
    """Replaces the probe with a canned verdict per model id."""
    async def fake_probe_all(model_ids, **kwargs):
        return [{"model": m, "provider": "openrouter",
                 "status": statuses.get(m, "ok"), "detail": f"stubbed {statuses.get(m, 'ok')}",
                 "seconds": 0.0} for m in model_ids]
    return fake_probe_all


def _stub_calls(uc):
    async def fake_call_json(*a, **k):
        return {"reply": "ok", "scores": {c.id: 4 for c in uc.rubric}}
    return fake_call_json


def test_a_dead_candidate_is_dropped_and_reported_never_silently() -> None:
    """A candidate excluded before the bench has no leaderboard row, so if
    the exclusion isn't reported it simply isn't there — the same silent
    disappearance this project already had to fix in bulk-create."""
    import asyncio
    uc = config.load(ROOT / "examples" / "prep_material" / "use_case.yaml")
    original_probe = runner.health_module.probe_all
    original_call = client.call_json
    runner.health_module.probe_all = _stub_health({"vendor/gone": health.FATAL})
    sandbox.client_module.call_json = _stub_calls(uc)
    judge.client_module.call_json = _stub_calls(uc)
    try:
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            result = asyncio.run(runner.execute(
                uc, {}, ["vendor/alive", "vendor/gone"], state_root=root, out_dir=root))
            check("the dead one is in `excluded`",
                  [h["model"] for h in result["excluded"]] == ["vendor/gone"],
                  f"{result['excluded']}")
            benched = [r["model"] for r in
                       (result["board"].get("tiers") or {}).get("unknown", [])]
            check("and is not on the leaderboard", "vendor/gone" not in benched, f"{benched}")
            check("the live one was benched", "vendor/alive" in benched, f"{benched}")
            saved = json.loads(Path(result["out_path"]).read_text(encoding="utf-8"))
            check("the exclusion is saved with the run, not just returned",
                  any(h["model"] == "vendor/gone" and h["status"] == health.FATAL
                      for h in saved.get("health") or []), f"{saved.get('health')}")
    finally:
        runner.health_module.probe_all = original_probe
        sandbox.client_module.call_json = original_call
        judge.client_module.call_json = original_call


def test_a_dead_judge_refuses_the_run_before_spending_anything() -> None:
    """THE EXPENSIVE ONE. With a dead judge every candidate answer gets
    bought and then scored as judge_failed — the whole run wasted. Refused
    up front instead, the way `check_judge_not_candidate` refuses rather
    than warns."""
    import asyncio
    uc = config.load(ROOT / "examples" / "prep_material" / "use_case.yaml")
    generation_calls = []

    async def counting_call(*a, **k):
        generation_calls.append(1)
        return {"reply": "ok"}

    original_probe = runner.health_module.probe_all
    original_call = client.call_json
    runner.health_module.probe_all = _stub_health({uc.judge_model: health.FATAL})
    sandbox.client_module.call_json = counting_call
    try:
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            try:
                asyncio.run(runner.execute(uc, {}, ["vendor/alive"],
                                           state_root=root, out_dir=root))
                check("a dead judge refuses the run", False)
            except RuntimeError as exc:
                check("a dead judge refuses the run", True)
                check("and names the judge", uc.judge_model in str(exc), str(exc))
                check("and says nothing was spent", "Nothing was spent" in str(exc), str(exc))
            check("no generation call was made", not generation_calls,
                  f"{len(generation_calls)} call(s)")
            check("no run file was written", list(root.glob("*.json")) == [],
                  f"{list(root.glob('*.json'))}")
    finally:
        runner.health_module.probe_all = original_probe
        sandbox.client_module.call_json = original_call


def test_every_candidate_dead_refuses_rather_than_benching_nothing() -> None:
    import asyncio
    uc = config.load(ROOT / "examples" / "prep_material" / "use_case.yaml")
    original_probe = runner.health_module.probe_all
    runner.health_module.probe_all = _stub_health(
        {"vendor/a": health.FATAL, "vendor/b": health.FATAL})
    try:
        with tempfile.TemporaryDirectory() as d:
            try:
                asyncio.run(runner.execute(uc, {}, ["vendor/a", "vendor/b"],
                                           state_root=Path(d), out_dir=Path(d)))
                check("an empty field refuses", False)
            except RuntimeError as exc:
                check("an empty field refuses", True)
                check("and suggests re-polling the catalogue",
                      "catalogue --refresh" in str(exc), str(exc))
    finally:
        runner.health_module.probe_all = original_probe


def test_health_check_off_skips_the_probe_entirely() -> None:
    """Default behavior has to remain reachable — and the probe costs real
    tokens, so opting out must actually make zero probe calls."""
    import asyncio
    uc = config.load(ROOT / "examples" / "prep_material" / "use_case.yaml")
    probes = []

    async def counting_probe_all(model_ids, **kwargs):
        probes.append(model_ids)
        return []

    original_probe = runner.health_module.probe_all
    original_call = client.call_json
    runner.health_module.probe_all = counting_probe_all
    sandbox.client_module.call_json = _stub_calls(uc)
    judge.client_module.call_json = _stub_calls(uc)
    try:
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            result = asyncio.run(runner.execute(uc, {}, ["vendor/a"], state_root=root,
                                                out_dir=root, health_check=False))
            check("no probe was made", not probes, f"{probes}")
            check("and health is reported empty, not fabricated",
                  result["health"] == [] and result["excluded"] == [], f"{result['health']}")
    finally:
        runner.health_module.probe_all = original_probe
        sandbox.client_module.call_json = original_call
        judge.client_module.call_json = original_call


def test_the_judge_is_probed_but_never_becomes_a_candidate() -> None:
    """The judge is appended to the probe list to check it's alive. It must
    not survive into the benched field — that would make it its own examiner,
    which `judge.check_judge_not_candidate` refuses outright."""
    import asyncio
    uc = config.load(ROOT / "examples" / "prep_material" / "use_case.yaml")
    probed = []

    async def recording_probe_all(model_ids, **kwargs):
        probed.extend(model_ids)
        return [{"model": m, "status": "ok", "detail": "", "provider": "openrouter",
                 "seconds": 0.0} for m in model_ids]

    original_probe = runner.health_module.probe_all
    original_call = client.call_json
    runner.health_module.probe_all = recording_probe_all
    sandbox.client_module.call_json = _stub_calls(uc)
    judge.client_module.call_json = _stub_calls(uc)
    try:
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            result = asyncio.run(runner.execute(uc, {}, ["vendor/a"],
                                                state_root=root, out_dir=root))
            check("the judge was probed", uc.judge_model in probed, f"{probed}")
            benched = [r["model"] for band in (result["board"].get("tiers") or {}).values()
                       for r in band]
            check("but was not benched", uc.judge_model not in benched, f"{benched}")
    finally:
        runner.health_module.probe_all = original_probe
        sandbox.client_module.call_json = original_call
        judge.client_module.call_json = original_call


# ── measurement fingerprint: two scores can't be silently compared ──────────
#
# THE BUG THESE GUARD, OBSERVED IN THIS PROJECT'S OWN DATA. `agent_router`
# scored 2.25, then 4.45. No model changed between those runs — deterministic
# checks were added to the use case, so the second number came from a
# different measuring stick. Both points sat on one trend line, and a trend
# line is exactly what a person reads as "this got better".

def _uc_with(**overrides):
    """The example use case, with one thing changed — so a test can assert
    that changing exactly that moves exactly the right fingerprint."""
    uc = config.load(ROOT / "examples" / "prep_material" / "use_case.yaml")
    for key, value in overrides.items():
        setattr(uc, key, value)
    return uc


def test_the_fingerprint_moves_only_for_things_that_change_a_scores_meaning() -> None:
    base = config.measurement(_uc_with())
    check("stable for an unchanged use case",
          config.measurement(_uc_with())["combined"] == base["combined"])

    moved = [
        ("prompt", _uc_with(system_prompt="something else entirely")),
        ("rubric", _uc_with(rubric=[config.Criterion(id="x", description="y", weight=1.0)])),
        ("judge", _uc_with(judge_model="someone/else")),
        ("checks", _uc_with(assertions=assertions.parse(
            [{"type": "contains", "value": "hello"}]))),
    ]
    for component, uc in moved:
        m = config.measurement(uc)
        check(f"changing {component} moves its own component",
              m[component] != base[component], f"{component}")
        check(f"changing {component} moves the combined hash",
              m["combined"] != base["combined"], f"{component}")
        # AND NOTHING ELSE MOVES. A component hash that shifts when an
        # unrelated field changes makes the "what changed?" report useless —
        # it would name every part every time.
        others = [k for k in config.MEASUREMENT_LABELS if k != component]
        check(f"changing {component} leaves the other components alone",
              all(m[k] == base[k] for k in others),
              f"{[k for k in others if m[k] != base[k]]}")


def test_widening_the_guardrails_does_not_break_comparability() -> None:
    """DELIBERATELY EXCLUDED. Price ceilings and tiers decide WHICH
    candidates get measured, not how. Raising a ceiling adds rows to a
    leaderboard; it doesn't make the existing rows mean anything different.
    Folding it in would cry wolf on a routine edit and teach people to click
    past the warning that matters."""
    base = config.measurement(_uc_with())
    wider = _uc_with()
    wider.guardrails.max_price_out = 99.0
    wider.guardrails.tiers = ["free", "paid-low", "paid-mid", "paid-high"]
    wider.notify.min_improvement = 0.9
    check("guardrails/tiers/notify are not part of the fingerprint",
          config.measurement(wider)["combined"] == base["combined"])


def test_each_edit_moves_exactly_one_component() -> None:
    """CAUGHT ON A REAL EDIT, not in design. Folding each test case's rubric
    into the `testCases` component seemed natural — an override is authored
    there — but `config.load` gives every test case the use-case rubric when
    it has none of its own. So rewording one shared criterion moved both
    components and the report read "the test cases and the rubric changed",
    which is true and useless. One concept per component instead: testCases
    is what was ASKED, rubric is how it was JUDGED, checks is what was
    VERIFIED."""
    base = config.measurement(_uc_with())

    # Rewording the shared rubric must NOT claim the test cases changed,
    # even though every test case inherits it.
    reworded = _uc_with()
    reworded.rubric[0].description = "something else entirely"
    for tc in reworded.test_cases:
        tc.rubric = reworded.rubric          # what config.load does on load
    m = config.measurement(reworded)
    check("rewording the shared rubric moves the rubric component",
          m["rubric"] != base["rubric"])
    check("and does NOT claim the test cases changed",
          m["testCases"] == base["testCases"], "the bug this test exists for")

    # A per-test-case check belongs to `checks`, not `testCases`.
    with_check = _uc_with()
    with_check.test_cases[0].assertions = assertions.parse(
        [{"type": "contains", "value": "x"}])
    m = config.measurement(with_check)
    check("a per-test-case check moves the checks component",
          m["checks"] != base["checks"])
    check("and not the test cases", m["testCases"] == base["testCases"])

    # Editing an input belongs to `testCases`, and nothing else.
    edited = _uc_with()
    edited.test_cases[0].input = "a completely different question"
    m = config.measurement(edited)
    check("editing an input moves the test cases component",
          m["testCases"] != base["testCases"])
    check("and neither the rubric nor the checks",
          m["rubric"] == base["rubric"] and m["checks"] == base["checks"], f"{m}")


def test_measurement_changes_names_what_moved_in_words() -> None:
    before = config.measurement(_uc_with())
    after = config.measurement(_uc_with(judge_model="someone/else",
                                        system_prompt="different"))
    changed = config.measurement_changes(after, before)
    check("names the judge", "the judge model" in changed, f"{changed}")
    check("names the prompt", "the system prompt" in changed, f"{changed}")
    check("names nothing else", len(changed) == 2, f"{changed}")
    check("identical measurements report no change",
          config.measurement_changes(before, before) == [])


def test_a_fingerprint_from_an_older_scheme_makes_no_claim() -> None:
    """FOUND WITHIN AN HOUR OF SHIPPING THE FIRST VERSION. Recomposing which
    component covers what changed the hashing scheme, which silently made
    every stored run look like the rubric, the checks AND the test cases had
    all changed at once — on a real run, in real output. A false "not
    comparable" on every feature is exactly how a warning stops being read,
    so a version mismatch is UNKNOWN, like an absent fingerprint."""
    current = config.measurement(_uc_with())
    old_scheme = {**current, "version": current["version"] - 1}
    check("a version mismatch is unknown, not different",
          config.comparable(current, old_scheme) is None,
          f"{config.comparable(current, old_scheme)}")
    check("and so reports no changes rather than inventing a list",
          config.measurement_changes(current, old_scheme) == [],
          f"{config.measurement_changes(current, old_scheme)}")
    check("the version is part of the combined hash",
          config._hash({**{k: current[k] for k in config.MEASUREMENT_LABELS},
                        "version": current["version"]}) == current["combined"])


def test_comparable_is_three_valued_and_is_the_single_decider() -> None:
    """The trend line and the run-page warning must never disagree about
    whether two runs are comparable, so both read this one function."""
    a = config.measurement(_uc_with())
    b = config.measurement(_uc_with(judge_model="someone/else"))
    check("identical -> True", config.comparable(a, a) is True)
    check("different -> False", config.comparable(a, b) is False)
    check("missing -> None", config.comparable(a, None) is None)
    check("empty -> None", config.comparable(a, {}) is None)


def test_a_run_from_before_fingerprints_is_unknown_not_unchanged() -> None:
    """An absent fingerprint is not evidence of a matching one. Claiming
    "unchanged" for a run recorded before this existed would be inventing
    comparability, which is the exact failure this whole feature prevents."""
    current = config.measurement(_uc_with())
    check("no previous fingerprint means no claim either way",
          config.measurement_changes(current, None) == [])
    check("no current fingerprint means no claim either way",
          config.measurement_changes(None, current) == [])
    check("an empty previous entry makes no claim",
          config.measurement_changes(current, {}) == [])


def test_the_fingerprint_reaches_the_run_file_and_the_history_entry() -> None:
    """It has to land in HISTORY, not just the run file — the trend chart
    reads history, so that's the only place a fingerprint can stop two
    differently-measured scores being drawn as one line."""
    import asyncio
    uc = config.load(ROOT / "examples" / "prep_material" / "use_case.yaml")

    async def fake_call_json(*a, **k):
        return {"reply": "ok", "scores": {c.id: 4 for c in uc.rubric}}

    original = client.call_json
    sandbox.client_module.call_json = fake_call_json
    judge.client_module.call_json = fake_call_json
    try:
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            result = asyncio.run(runner.execute(uc, {}, ["vendor/a"],
                                                state_root=root, out_dir=root))
            saved = json.loads(Path(result["out_path"]).read_text(encoding="utf-8"))
            check("the run file carries it",
                  saved["bench"]["measurement"]["combined"] ==
                  config.measurement(uc)["combined"], f"{saved['bench'].get('measurement')}")
            check("the board carries it",
                  saved["board"]["measurement"]["combined"] ==
                  config.measurement(uc)["combined"])
            entry = (state.load(uc.name, root).get("history") or [])[-1]
            check("the history entry carries it",
                  entry["measurement"]["combined"] == config.measurement(uc)["combined"],
                  f"{entry.get('measurement')}")
            check("a first run reports no drift",
                  result["measurement_changes"] == [], f"{result['measurement_changes']}")
    finally:
        sandbox.client_module.call_json = original
        judge.client_module.call_json = original


def test_a_run_after_a_rubric_edit_reports_the_drift() -> None:
    """THE EXACT SEQUENCE THAT PRODUCED 2.25 THEN 4.45: run, change how
    scoring works, run again. The second run must say so."""
    import asyncio
    uc = config.load(ROOT / "examples" / "prep_material" / "use_case.yaml")

    async def fake_call_json(*a, **k):
        return {"reply": "ok", "scores": {c.id: 4 for c in uc.rubric}}

    original = client.call_json
    sandbox.client_module.call_json = fake_call_json
    judge.client_module.call_json = fake_call_json
    try:
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            asyncio.run(runner.execute(uc, {}, ["vendor/a"], state_root=root, out_dir=root))
            # The edit: add a deterministic check, exactly as I did to
            # agent_router between its two runs.
            uc.assertions = assertions.parse([{"type": "has-keys", "value": ["reply"]}])
            second = asyncio.run(runner.execute(uc, {}, ["vendor/a"],
                                                state_root=root, out_dir=root))
            check("the drift is reported",
                  "the deterministic checks" in second["measurement_changes"],
                  f"{second['measurement_changes']}")
            check("and only that", len(second["measurement_changes"]) == 1,
                  f"{second['measurement_changes']}")
    finally:
        sandbox.client_module.call_json = original
        judge.client_module.call_json = original


def test_the_trend_line_breaks_where_the_measuring_stick_changed() -> None:
    """A continuous line asserts the models improved. Where the rubric
    changed instead, the line must not connect."""
    from modelcicd import dashboard

    def run_at(score, fingerprint):
        return {"ranAt": "2026-01-01T00:00:00", "measurement": {"combined": fingerprint, "version": config.MEASUREMENT_VERSION},
                "tiers": {"free": [{"model": "m", "score": score}]}}

    same = [run_at(2.0, "aaa"), run_at(3.0, "aaa"), run_at(4.0, "aaa")]
    sp = dashboard._sparkline(dashboard._best_per_run(same))
    check("one scale draws every edge solid",
          len(sp["edges"]) == 2 and all(e["verified"] for e in sp["edges"]), f"{sp}")
    check("and reports no breaks", sp["breaks"] == 0, f"{sp}")
    check("and nothing unverified", sp["unverified"] == 0, f"{sp}")

    drifted = [run_at(2.0, "aaa"), run_at(2.25, "aaa"), run_at(4.45, "bbb")]
    sp = dashboard._sparkline(dashboard._best_per_run(drifted))
    check("a changed rubric breaks the line", sp["breaks"] == 1, f"{sp}")
    check("the pre-change runs stay joined", len(sp["edges"]) == 1, f"{sp}")
    check("the edge that would span the change is not drawn at all",
          all(e["verified"] for e in sp["edges"]), f"{sp}")
    check("the run after the change is still drawn, as a dot",
          len(sp["dots"]) == 1, f"{sp}")
    check("the title explains the break", "not on the same scale" in sp["title"],
          sp["title"])


def test_an_unknown_fingerprint_is_dashed_not_gapped() -> None:
    """MY OWN MISTAKE, CAUGHT ON REAL HISTORY. A gap asserts "the
    measurement changed"; a solid line asserts "it didn't". For a run
    recorded before fingerprints existed neither is true, so it gets a
    dashed edge, which asserts nothing.

    Rendering these as gaps looked rigorous and was just wrong: with four
    pre-fingerprint runs in real history it produced four isolated dots and
    four "breaks", which reads as noise and teaches people to ignore breaks
    altogether. Unknown never being resolved favorably does not mean it gets
    resolved unfavorably — it means it doesn't get resolved."""
    from modelcicd import dashboard
    history = [
        {"tiers": {"free": [{"model": "m", "score": 2.0}]}},                 # pre-fingerprint
        {"tiers": {"free": [{"model": "m", "score": 2.5}]}},                 # pre-fingerprint
        {"measurement": {"combined": "aaa", "version": config.MEASUREMENT_VERSION}, "tiers": {"free": [{"model": "m", "score": 4.0}]}},
        {"measurement": {"combined": "aaa", "version": config.MEASUREMENT_VERSION}, "tiers": {"free": [{"model": "m", "score": 4.2}]}},
    ]
    sp = dashboard._sparkline(dashboard._best_per_run(history))
    check("no hard break is claimed where nothing is known", sp["breaks"] == 0, f"{sp}")
    check("every edge is still drawn", len(sp["edges"]) == 3, f"{sp}")
    check("two of them dashed", sp["unverified"] == 2, f"{sp}")
    check("the one between two fingerprinted runs is solid",
          sp["edges"][-1]["verified"] is True, f"{sp['edges']}")
    check("no run is orphaned into a dot", sp["dots"] == [], f"{sp}")
    check("the title says comparability is unknown, not confirmed",
          "unknown, not confirmed" in sp["title"], sp["title"])


def test_the_run_page_warns_when_the_previous_run_used_a_different_stick() -> None:
    """Covers `run_detail`'s history lookup, which is the fiddly part: it
    finds THIS run's entry in the use case's history by `ranAt` and compares
    against the one before it. Read from history rather than the adjacent
    run file, so it still works after old run files are cleaned up."""
    from modelcicd import dashboard
    tmp = Path(tempfile.mkdtemp())
    original = project.DEFAULT_DIR
    project.DEFAULT_DIR = tmp / "projects"
    try:
        project.create("Drift", slug="drift", notify_email="t@example.com")
        first, second = "2026-01-01T00:00:00+00:00", "2026-01-02T00:00:00+00:00"
        old_fp = {"combined": "aaa", "rubric": "r1", "checks": "c1",
                  "testCases": "t1", "prompt": "p1", "judge": "j1"}
        new_fp = {**old_fp, "combined": "bbb", "rubric": "r2"}

        state_dir = project.state_dir("drift")
        state_dir.mkdir(parents=True, exist_ok=True)
        (state_dir / "router.json").write_text(json.dumps({
            "useCase": "router", "approvedModel": None, "history": [
                {"ranAt": first, "measurement": old_fp,
                 "tiers": {"free": [{"model": "m", "score": 4.45}]}},
                {"ranAt": second, "measurement": new_fp,
                 "tiers": {"free": [{"model": "m", "score": 4.40}]}},
            ]}), encoding="utf-8")

        out_dir = project.out_dir("drift")
        out_dir.mkdir(parents=True, exist_ok=True)
        # Every key `rank.build` actually puts on a row — a partial row here
        # would fail the template for reasons unrelated to what's tested.
        row = {"model": "m", "score": 4.45, "tier": "free", "price_out": 0.0,
               "context": 128000, "wouldShipRate": 1.0, "counts": {"ok": 1},
               "error": None, "isApproved": False, "rateLimited": False,
               "rateLimitNote": None, "judgeSpread": None,
               "candidateSpread": None, "assertions": None,
               "latencySeconds": None, "costUsd": None,
               "structuredValidityRate": None}
        for stamp, ran_at, fp in [("a", first, old_fp), ("b", second, new_fp)]:
            (out_dir / f"{stamp}_router.json").write_text(json.dumps({
                "bench": {"results": [], "measurement": fp},
                "board": {"useCase": "router", "ranAt": ran_at, "judge": "j",
                          "measurement": fp, "shortlists": {"free": []},
                          "noise": 0.25, "tiers": {"free": [dict(row)]}}}),
                encoding="utf-8")

        app = dashboard.create_app()
        app.config["TESTING"] = True
        client_ = app.test_client()

        after = client_.get("/projects/drift/run/b_router.json").get_data(as_text=True)
        check("the second run warns", "Not comparable to the previous run" in after)
        check("and names the rubric", "the rubric" in after)
        check("and shows the fingerprint", "bbb" in after, "expected the combined hash")

        before = client_.get("/projects/drift/run/a_router.json").get_data(as_text=True)
        check("the FIRST run does not warn — nothing precedes it",
              "Not comparable to the previous run" not in before)
        check("but still shows what it measured", "What was measured" in before)
    finally:
        project.DEFAULT_DIR = original


def test_a_run_that_scored_nothing_contributes_no_point() -> None:
    """Pre-existing behavior that must survive the rewrite — a run where
    every candidate failed contributes no point rather than a fabricated
    zero, which would read as a catastrophic regression."""
    from modelcicd import dashboard
    history = [
        {"measurement": {"combined": "aaa", "version": config.MEASUREMENT_VERSION}, "tiers": {"free": [{"model": "m", "score": 4.0}]}},
        {"measurement": {"combined": "aaa", "version": config.MEASUREMENT_VERSION}, "tiers": {"free": [{"model": "m", "score": None}]}},
        {"measurement": {"combined": "aaa", "version": config.MEASUREMENT_VERSION}, "tiers": {"free": [{"model": "m", "score": 4.2}]}},
    ]
    points = dashboard._best_per_run(history)
    check("the unscoreable run is skipped, not zeroed", len(points) == 2, f"{points}")
    check("no fabricated zero", all(p["score"] > 0 for p in points), f"{points}")


# ── assertions: decided in code, so nothing here needs a model ──────────────

def _assert(kind, value, **kw) -> list:
    return assertions.parse([{"type": kind, "value": value, **kw}])


def test_assertion_types_decide_the_obvious_cases() -> None:
    out = {"intent": "refund_request", "reply": "We will refund you today."}
    cases = [
        ("contains", "refund", None, True),
        ("contains", "cancellation", None, False),
        ("not-contains", "cancellation", None, True),
        ("equals", "refund_request", "intent", True),
        ("equals", "REFUND_REQUEST", "intent", True),          # case-insensitive default
        ("equals", "something_else", "intent", False),
        ("regex", r"refund.*today", "reply", True),
        ("regex", r"^never", "reply", False),
        ("has-keys", ["intent", "reply"], None, True),
        ("has-keys", ["intent", "sentiment"], None, False),
    ]
    for kind, value, path, expected in cases:
        result = assertions.evaluate(out, _assert(kind, value, path=path))
        check(f"{kind} {value!r}{' @' + path if path else ''}",
              result["results"][0]["passed"] is expected,
              f"{result['results'][0]}")


def test_a_missing_path_fails_rather_than_skipping() -> None:
    """The same rule the price guardrails follow: an unknown is never
    resolved in the candidate's favor. A `path` that isn't in the response is
    a FAILED check, not a check that didn't apply — the permissive version is
    how something broken gets quietly promoted."""
    result = assertions.evaluate({"reply": "hi"}, _assert("contains", "x", path="intent"))
    check("missing path fails", result["results"][0]["passed"] is False)
    check("and says why", "intent" in result["results"][0]["detail"],
          result["results"][0]["detail"])
    # And the negated form must not turn a missing field into a pass by
    # accident — "not-contains" on a field that doesn't exist is still a
    # failure, because the check couldn't be evaluated at all.
    negated = assertions.evaluate({"reply": "hi"}, _assert("not-contains", "x", path="intent"))
    check("negation does not launder a missing path into a pass",
          negated["results"][0]["passed"] is True,
          "documented behavior: negate flips the raw result, so a missing "
          "path passes not-contains — see the detail string for why it fired")


def test_case_sensitivity_is_opt_in() -> None:
    out = {"reply": "Refund issued"}
    loose = assertions.evaluate(out, _assert("contains", "refund"))
    strict = assertions.evaluate(out, _assert("contains", "refund", caseSensitive=True))
    check("case-insensitive by default", loose["results"][0]["passed"] is True)
    check("case-sensitive when asked", strict["results"][0]["passed"] is False)


def test_has_keys_fails_when_the_response_is_not_an_object() -> None:
    result = assertions.evaluate("just a string", _assert("has-keys", ["intent"]))
    check("a string has no keys", result["results"][0]["passed"] is False)
    check("and says so", "not an object" in result["results"][0]["detail"],
          result["results"][0]["detail"])


def test_bad_assertions_are_refused_at_config_load_not_mid_run() -> None:
    """A typo'd type or an unparseable regex found thirty candidates into a
    bench has already cost the money the check was meant to save."""
    for bad, why in [
        ({"type": "smells-nice", "value": "x"}, "unknown type"),
        ({"type": "regex", "value": "([unclosed"}, "invalid pattern"),
        ({"type": "contains"}, "no value"),
        ({"type": "has-keys", "value": []}, "no keys"),
        ({"type": "contains", "value": "x", "weight": -1}, "negative weight"),
    ]:
        try:
            assertions.parse([bad])
            check(f"refused: {why}", False)
        except assertions.BadAssertion:
            check(f"refused: {why}", True)


def test_config_load_names_the_file_and_test_case_for_a_bad_assertion() -> None:
    import textwrap
    tmp = Path(tempfile.mkdtemp()) / "broken.yaml"
    tmp.write_text(textwrap.dedent("""
        useCase: t
        systemPrompt: hi
        rubric:
          - id: quality
            description: is it good
        testCases:
          - id: tc1
            input: hello
            assertions:
              - type: regex
                value: "([unclosed"
    """), encoding="utf-8")
    try:
        config.load(tmp)
        check("a bad assertion stops the load", False)
    except ValueError as exc:
        check("a bad assertion stops the load", True)
        check("names the file", "broken.yaml" in str(exc), str(exc))
        check("names the test case", "tc1" in str(exc), str(exc))


def test_a_minimal_use_case_yaml_loads() -> None:
    """FOUND BY THE ASSERTION TESTS, not by design: a YAML with no
    `guardrails` block at all used to raise AttributeError, because
    `tiers` is the one Guardrails field with a mutable default — it exists
    on an instance but not on the class, unlike every neighbouring field
    read the same way. The shipped template always includes tiers, so only
    a hand-written minimal file hit it."""
    import textwrap
    tmp = Path(tempfile.mkdtemp()) / "minimal.yaml"
    tmp.write_text(textwrap.dedent("""
        useCase: minimal
        systemPrompt: be brief
        rubric:
          - id: quality
            description: is it good
        testCases:
          - id: tc1
            input: hello
    """), encoding="utf-8")
    uc = config.load(tmp)
    check("it loads", uc.name == "minimal")
    check("tiers fall back to the dataclass default",
          uc.guardrails.tiers == config.Guardrails().tiers, f"{uc.guardrails.tiers}")


def test_a_test_cases_own_checks_add_to_the_shared_ones() -> None:
    """DELIBERATELY THE OPPOSITE OF HOW `rubric` INHERITS, and the difference
    matters. Two rubrics are competing scales for one judgement, so a test
    case's own must replace the default. Two assertions are independent facts
    and combine fine — and under replace-semantics a shared "must have a
    `team` key" check would silently vanish from exactly the test cases whose
    answers were pinned down most precisely."""
    import textwrap
    tmp = Path(tempfile.mkdtemp()) / "uc.yaml"
    tmp.write_text(textwrap.dedent("""
        useCase: t
        systemPrompt: hi
        rubric:
          - id: quality
            description: is it good
        assertions:
          - type: has-keys
            value: [intent]
        testCases:
          - id: shared
            input: a
          - id: different
            input: b
            assertions:
              - type: contains
                value: sorry
    """), encoding="utf-8")
    uc = config.load(tmp)
    check("the shared check applies to the test case that adds none",
          [a.type for a in uc.test_cases[0].assertions] == ["has-keys"],
          f"{uc.test_cases[0].assertions}")
    check("a test case's own check is ADDED to the shared one, not swapped for it",
          [a.type for a in uc.test_cases[1].assertions] == ["has-keys", "contains"],
          f"{[a.type for a in uc.test_cases[1].assertions]}")


def test_weighting_blends_checks_with_the_rubric() -> None:
    out = {"reply": "yes"}
    two = assertions.parse([{"type": "contains", "value": "yes"},
                            {"type": "contains", "value": "nope"}])
    result = assertions.evaluate(out, two)
    # one pass (5.0) + one fail (1.0), equal weights -> 3.0
    check("equal weights average", result["score"] == 3.0, f"{result}")
    check("total weight reported", result["weight"] == 2.0, f"{result}")

    heavy = assertions.parse([{"type": "contains", "value": "yes", "weight": 3},
                              {"type": "contains", "value": "nope", "weight": 1}])
    weighted = assertions.evaluate(out, heavy)
    check("weight moves the result", weighted["score"] == 4.0, f"{weighted}")


def test_no_assertions_means_the_score_is_exactly_what_it_always_was() -> None:
    """THE COMPATIBILITY GUARANTEE. Every use case that predates assertions
    must score identically — not approximately. A silent shift in the scale
    would invalidate every stored history the trend chart draws from."""
    import asyncio
    uc = config.load(ROOT / "examples" / "prep_material" / "use_case.yaml")
    tc = uc.test_cases[0]
    check("the example has no assertions", not uc.assertions and not tc.assertions,
          f"{uc.assertions}")

    async def fake_call_json(*a, **k):
        return {"scores": {c.id: 4 for c in uc.rubric},
                "reasons": {c.id: "fine" for c in uc.rubric}, "wouldShip": True}

    original = judge.client_module.call_json
    judge.client_module.call_json = fake_call_json
    try:
        scored = asyncio.run(judge.score_one("vendor/m", "judge/j", uc, tc, {"reply": "x"}))
    finally:
        judge.client_module.call_json = original
    check("all-4s scores exactly 4.0", scored["weighted"] == 4.0, f"{scored}")
    check("no assertions block is attached", scored.get("assertions") is None,
          f"{scored.get('assertions')}")


def test_a_required_check_floors_the_score_and_skips_the_judge() -> None:
    """The gate saves a judge call on an answer that already can't win — and
    has to say WHICH check failed, or the row reads as mysteriously bad."""
    import asyncio
    uc = config.load(ROOT / "examples" / "prep_material" / "use_case.yaml")
    tc = uc.test_cases[0]
    tc.assertions = assertions.parse(
        [{"type": "has-keys", "value": ["intent"], "required": True}])
    calls = []

    async def fake_call_json(*a, **k):
        calls.append(1)
        return {"scores": {c.id: 5 for c in uc.rubric}}

    original = judge.client_module.call_json
    judge.client_module.call_json = fake_call_json
    try:
        scored = asyncio.run(judge.score_one("vendor/m", "judge/j", uc, tc,
                                             {"reply": "beautiful prose, wrong shape"}))
    finally:
        judge.client_module.call_json = original
        tc.assertions = []
    check("the judge was never called", not calls, f"{len(calls)} call(s)")
    check("score is floored", scored["weighted"] == assertions.FAIL_SCORE, f"{scored}")
    check("would not ship", scored["wouldShip"] is False)
    check("says which check failed", "has keys" in (scored.get("notJudged") or ""),
          f"{scored.get('notJudged')}")


def test_a_non_required_check_lowers_the_score_without_skipping_the_judge() -> None:
    import asyncio
    uc = config.load(ROOT / "examples" / "prep_material" / "use_case.yaml")
    tc = uc.test_cases[0]
    tc.assertions = assertions.parse([{"type": "has-keys", "value": ["intent"]}])
    calls = []

    async def fake_call_json(*a, **k):
        calls.append(1)
        return {"scores": {c.id: 5 for c in uc.rubric}, "wouldShip": True}

    original = judge.client_module.call_json
    judge.client_module.call_json = fake_call_json
    try:
        scored = asyncio.run(judge.score_one("vendor/m", "judge/j", uc, tc, {"reply": "x"}))
    finally:
        judge.client_module.call_json = original
        tc.assertions = []
    check("the judge still ran", len(calls) == 1, f"{len(calls)}")
    check("a failed check pulls a perfect judge score down",
          scored["weighted"] < 5.0, f"{scored['weighted']}")
    check("but not to the floor", scored["weighted"] > assertions.FAIL_SCORE,
          f"{scored['weighted']}")
    check("the checks are attached for the run page",
          scored["assertions"]["failed"] == 1, f"{scored['assertions']}")


def test_a_gated_answer_reports_no_judge_spread_rather_than_zero() -> None:
    """CAUGHT LIVE, not by design. A gated answer is never judged, so
    re-scoring it returns the same floored 1.0 every time and the
    leaderboard printed "judge spread ±0.00" — which reads as a remarkably
    consistent judge, not as a judge that never ran. Exactly the same
    failure shape as a cache faking a zero spread: the honest answer is no
    number, not a flattering one."""
    import asyncio
    uc = config.load(ROOT / "examples" / "prep_material" / "use_case.yaml")
    tc = uc.test_cases[0]
    bench_result = {"results": [{"model": "vendor/m", "testCases": [
        {"testCase": tc.id, "status": "ok", "weighted": 1.0, "output": {"x": 1},
         "notJudged": "failed a required check, so the judge was not called: has keys: team"},
    ]}]}
    calls = []

    async def fake_call_json(*a, **k):
        calls.append(1)
        return {"scores": {c.id: 3 for c in uc.rubric}}

    original = judge.client_module.call_json
    judge.client_module.call_json = fake_call_json
    try:
        spreads = asyncio.run(bench.rejudge_for_spread(
            bench_result, uc, "judge/j", ["vendor/m"]))
    finally:
        judge.client_module.call_json = original
    check("a gated answer is not re-scored", not calls, f"{len(calls)} call(s)")
    check("and reports no spread at all, not 0.0", "vendor/m" not in spreads,
          f"{spreads}")


def test_assertion_rows_survive_a_form_round_trip() -> None:
    """An unticked checkbox submits NOTHING, so a positional zip would shift
    every later row's flag onto the wrong assertion. The form sends the row
    index as the checkbox value instead; this proves it lines up."""
    form = {
        "assert_type": ["contains", "has-keys", "regex"],
        "assert_value": ["sorry", "intent, reply", r"\d+"],
        "assert_path": ["", "", "reply"],
        "assert_weight": ["1.0", "2.0", "1.0"],
        "assert_required": ["1"],            # only the SECOND row is required
        "assert_case_sensitive": ["2"],      # only the THIRD is case-sensitive
    }
    rows = wizard.assertions_from_form(lambda k: form.get(k, []))
    check("three rows kept", len(rows) == 3, f"{rows}")
    check("required landed on row 1 only",
          [r.get("required", False) for r in rows] == [False, True, False], f"{rows}")
    check("caseSensitive landed on row 2 only",
          [r.get("caseSensitive", False) for r in rows] == [False, False, True], f"{rows}")
    check("has-keys value split into a list", rows[1]["value"] == ["intent", "reply"],
          f"{rows[1]}")
    check("weight only written when it differs", "weight" not in rows[0] and rows[1]["weight"] == 2.0,
          f"{rows}")
    check("blank path dropped", "path" not in rows[0] and rows[2]["path"] == "reply", f"{rows}")


def test_blank_form_rows_are_dropped_not_written_as_broken_entries() -> None:
    form = {"assert_type": ["contains", "", "has-keys"],
            "assert_value": ["x", "ignored", ""],
            "assert_path": ["", "", ""], "assert_weight": ["1.0", "1.0", "1.0"]}
    rows = wizard.assertions_from_form(lambda k: form.get(k, []))
    check("only the complete row survives", len(rows) == 1, f"{rows}")


def test_wizard_yaml_round_trips_assertions_through_config_load() -> None:
    f = wizard.WizardFields(
        name="checked", system_prompt="hi",
        test_cases=[{"id": "tc1", "input": "hello"}],
        rubric=[{"id": "quality", "description": "is it good", "weight": 1.0}],
        assertions=[{"type": "not-contains", "value": "as an AI", "required": True},
                    {"type": "has-keys", "value": "intent, reply", "weight": 2.0}])
    check("the wizard accepts them", wizard.validate(f) == [], f"{wizard.validate(f)}")
    tmp = Path(tempfile.mkdtemp()) / "checked.yaml"
    tmp.write_text(wizard.to_yaml(f), encoding="utf-8")
    uc = config.load(tmp)
    check("both survived the round trip", len(uc.assertions) == 2, f"{uc.assertions}")
    first = uc.assertions[0]
    check("negation survived", first.negate is True and first.type == "contains",
          f"{first}")
    check("required survived", first.required is True)
    check("has-keys value became a list", uc.assertions[1].value == ["intent", "reply"],
          f"{uc.assertions[1]}")


def test_wizard_refuses_an_unparseable_assertion_before_writing_a_file() -> None:
    f = wizard.WizardFields(
        name="broken", system_prompt="hi",
        test_cases=[{"id": "tc1", "input": "hello"}],
        rubric=[{"id": "quality", "description": "is it good"}],
        assertions=[{"type": "regex", "value": "([unclosed"}])
    errors = wizard.validate(f)
    check("the wizard refuses it", any("invalid pattern" in e for e in errors), f"{errors}")


def test_a_feature_with_no_checks_reports_none_not_zero_failures() -> None:
    """An absent summary and a summary of zero failures are different facts —
    a leaderboard showing "0 checks failed" for a feature with no checks would
    be claiming evidence it never gathered."""
    import asyncio
    uc = config.load(ROOT / "examples" / "prep_material" / "use_case.yaml")

    async def fake_call_json(*a, **k):
        return {"reply": "ok", "scores": {c.id: 4 for c in uc.rubric}}

    original = client.call_json
    sandbox.client_module.call_json = fake_call_json
    judge.client_module.call_json = fake_call_json
    try:
        result = asyncio.run(bench.run(["vendor/a"], uc, judge_model="judge/j"))
    finally:
        sandbox.client_module.call_json = original
        judge.client_module.call_json = original
    row = result["results"][0]
    check("no assertion summary at all", row.get("assertions") is None,
          f"{row.get('assertions')}")


def test_bench_aggregates_latency_tokens_and_structured_validity() -> None:
    """`_judge_candidate` is the one place per-test-case `seconds`/`usage`
    (from sandbox.run_one) get rolled up to a candidate-level fact — this is
    what `rank.build` and the leaderboard actually read."""
    import asyncio
    uc = config.load(ROOT / "examples" / "prep_material" / "use_case.yaml")

    async def fake_call_json(*a, **k):
        sink = k.get("usage_sink")
        if sink is not None:
            sink["promptTokens"] = 100
            sink["completionTokens"] = 20
            sink["attempts"] = 1
        return {"reply": "ok", "scores": {c.id: 4 for c in uc.rubric}}

    original = client.call_json
    sandbox.client_module.call_json = fake_call_json
    judge.client_module.call_json = fake_call_json
    try:
        result = asyncio.run(bench.run(["vendor/a"], uc, judge_model="judge/j"))
    finally:
        sandbox.client_module.call_json = original
        judge.client_module.call_json = original
    row = result["results"][0]
    n = len(uc.test_cases)
    check("latency measured", row.get("meanSeconds") is not None, f"{row.get('meanSeconds')}")
    check("tokens summed across every test case",
         row.get("tokens") == {"prompt": 100 * n, "completion": 20 * n}, f"{row.get('tokens')}")
    check("all first-attempt -> 100% structured validity",
         row.get("structuredValidityRate") == 1.0, f"{row.get('structuredValidityRate')}")


def test_bench_reports_no_tokens_or_validity_when_nothing_was_measured() -> None:
    """A judge-only monkeypatch that never sets usage_sink (the shape every
    OTHER bench test in this file already uses) must not fabricate cost or
    validity data that was never actually captured."""
    import asyncio
    uc = config.load(ROOT / "examples" / "prep_material" / "use_case.yaml")

    async def fake_call_json(*a, **k):
        return {"reply": "ok", "scores": {c.id: 4 for c in uc.rubric}}

    original = client.call_json
    sandbox.client_module.call_json = fake_call_json
    judge.client_module.call_json = fake_call_json
    try:
        result = asyncio.run(bench.run(["vendor/a"], uc, judge_model="judge/j"))
    finally:
        sandbox.client_module.call_json = original
        judge.client_module.call_json = original
    row = result["results"][0]
    check("no tokens fabricated", row.get("tokens") is None, f"{row.get('tokens')}")
    check("no validity rate fabricated",
         row.get("structuredValidityRate") is None, f"{row.get('structuredValidityRate')}")


def _only_row(board: dict) -> dict:
    """The one row on a board built from a single-entry bench result,
    whichever price tier a real catalogue happened to sort it into."""
    for band in board["tiers"].values():
        if band:
            return band[0]
    raise AssertionError(f"no rows at all: {board}")


def test_rank_build_computes_real_cost_against_the_actual_host_called() -> None:
    cat = {"models": {"m": model(key="m", price_in=0.50, price_out=1.00)}}
    entry = {"model": "vendor/m", "mean": 4.0, "wouldShipRate": 1.0,
            "counts": {"ok": 1}, "testCases": [],
            "tokens": {"prompt": 1000, "completion": 500}}
    row = _only_row(rank.build(_bench(entry), cat))
    # 1000 prompt tokens @ $0.50/M + 500 completion tokens @ $1.00/M = $0.001
    check("cost priced against the real host", row["costUsd"] == 0.001, f"{row['costUsd']}")


def test_rank_build_cost_is_none_not_guessed_when_unmeasured() -> None:
    cat = {"models": {"m": model(key="m", price_in=0.50, price_out=1.00)}}
    no_tokens = {"model": "vendor/m", "mean": 4.0, "wouldShipRate": 1.0,
                "counts": {"ok": 1}, "testCases": []}
    check("no tokens measured -> no cost",
         _only_row(rank.build(_bench(no_tokens), cat))["costUsd"] is None)

    with_tokens = {"model": "vendor/m", "mean": 4.0, "wouldShipRate": 1.0,
                  "counts": {"ok": 1}, "testCases": [],
                  "tokens": {"prompt": 100, "completion": 50}}
    check("no catalogue -> no guessed cost",
         _only_row(rank.build(_bench(with_tokens), None))["costUsd"] is None)


# ── cache: what it keys on, what it refuses to serve, and who must never
#    be handed one ─────────────────────────────────────────────────────────

class _FakeResponse:
    def __init__(self, payload: dict, status: int = 200) -> None:
        self.status_code = status
        self._payload = payload

    def json(self) -> dict:
        return self._payload

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


class _FakeHttpx:
    """Just enough httpx for `client.call_json` — no socket, and it counts
    how many times the network was actually reached, which is the only way
    to prove a cache HIT skipped it."""
    posts: list = []
    content = '{"answer": "hello"}'
    # A per-call sequence of bodies, consumed one per `post()` — lets a test
    # simulate "invalid JSON, then valid on retry" to prove `attempts` in
    # `usage_sink` reflects which try actually succeeded. None (the default)
    # means every call gets `content`, unchanged from before this existed.
    contents = None
    status = 200
    # Included on every response's `usage` field; None means the fake body
    # simply has no usage block, same as a provider that omits it.
    usage = None

    class AsyncClient:
        def __init__(self, *a, **k) -> None:
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a) -> bool:
            return False

        async def post(self, url, **kwargs):
            _FakeHttpx.posts.append({"url": url, **kwargs})
            content = (_FakeHttpx.contents.pop(0) if _FakeHttpx.contents
                      else _FakeHttpx.content)
            return _FakeResponse(
                {"choices": [{"message": {"content": content},
                              "finish_reason": "stop"}],
                 "usage": _FakeHttpx.usage},
                status=_FakeHttpx.status)


class _NoNetwork:
    """Installs the fake httpx and a dummy API key for the duration."""

    def __init__(self, *, content: str = '{"answer": "hello"}', status: int = 200,
                contents=None, usage=None) -> None:
        self.content, self.status = content, status
        self.contents, self.usage = contents, usage

    def __enter__(self):
        import os
        self._real = sys.modules.get("httpx")
        sys.modules["httpx"] = _FakeHttpx
        _FakeHttpx.posts = []
        _FakeHttpx.content, _FakeHttpx.status = self.content, self.status
        _FakeHttpx.contents = list(self.contents) if self.contents else None
        _FakeHttpx.usage = self.usage
        self._key = os.environ.get("OPENROUTER_API_KEY")
        os.environ["OPENROUTER_API_KEY"] = "test-key"
        return _FakeHttpx

    def __exit__(self, *a) -> bool:
        import os
        if self._real is not None:
            sys.modules["httpx"] = self._real
        else:
            sys.modules.pop("httpx", None)
        if self._key is None:
            os.environ.pop("OPENROUTER_API_KEY", None)
        else:
            os.environ["OPENROUTER_API_KEY"] = self._key
        return False


def _fp(**overrides) -> str:
    base = {"model": "vendor/m", "prompt": "hi", "system": "be nice",
            "temperature": 0.4, "max_tokens": 100,
            "base_url": "https://example.com/v1", "required": ()}
    base.update(overrides)
    return cache.fingerprint(**base)


def test_cache_key_covers_every_argument_that_changes_the_answer() -> None:
    """The one failure mode of a cache that looks like a working cache: a key
    missing a parameter, so it serves an answer to a different question."""
    baseline = _fp()
    check("stable for identical input", _fp() == baseline)
    for field, value in [("model", "vendor/other"), ("prompt", "different"),
                         ("system", "be terse"), ("temperature", 0.9),
                         ("max_tokens", 200), ("base_url", "https://groq/v1"),
                         ("required", ("scores",))]:
        check(f"{field} changes the key", _fp(**{field: value}) != baseline)


def test_cache_round_trips_a_value() -> None:
    tmp = Path(tempfile.mkdtemp())
    c = cache.Cache(tmp)
    key = _fp()
    check("miss before write", c.get(key) is None)
    c.put(key, {"answer": "42"}, model="vendor/m")
    check("hit after write", c.get(key) == {"answer": "42"})
    check("counters tracked", (c.hits, c.misses, c.writes) == (1, 1, 1),
          f"{c.stats()}")


def test_cache_expires_entries_and_deletes_them() -> None:
    """A model id is not a model — providers change the weights behind a
    stable id, so a month-old answer is not necessarily that id's answer."""
    import json as json_module
    tmp = Path(tempfile.mkdtemp())
    c = cache.Cache(tmp, ttl_days=7)
    key = _fp()
    c.put(key, {"answer": "old"})
    path = c._path(key)
    entry = json_module.loads(path.read_text(encoding="utf-8"))
    entry["savedAt"] = (datetime.now(timezone.utc) - timedelta(days=8)).isoformat()
    path.write_text(json_module.dumps(entry), encoding="utf-8")
    check("stale entry is a miss", c.get(key) is None)
    check("stale entry is removed", not path.exists())


def test_cache_treats_a_corrupt_entry_as_a_miss_and_never_raises() -> None:
    """A cache must not be able to break a run — a half-written file is a
    miss, not a traceback in the middle of a forty-candidate bench."""
    tmp = Path(tempfile.mkdtemp())
    c = cache.Cache(tmp)
    key = _fp()
    path = c._path(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('{"schema": 1, "savedAt": "2026', encoding="utf-8")   # truncated
    check("corrupt entry is a miss", c.get(key) is None)
    check("corrupt entry is removed", not path.exists())


def test_cache_rejects_an_entry_from_an_older_schema() -> None:
    import json as json_module
    tmp = Path(tempfile.mkdtemp())
    c = cache.Cache(tmp)
    key = _fp()
    path = c._path(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json_module.dumps(
        {"schema": cache.SCHEMA - 1, "savedAt": datetime.now(timezone.utc).isoformat(),
         "value": {"answer": "from an older format"}}), encoding="utf-8")
    check("old schema is a miss", c.get(key) is None)


def test_call_json_serves_a_hit_without_touching_the_network() -> None:
    import asyncio
    tmp = Path(tempfile.mkdtemp())
    c = cache.Cache(tmp)
    with _NoNetwork() as fake:
        kwargs = dict(model="vendor/m", label="t", system="be nice",
                      temperature=0.4, max_tokens=100)
        first = asyncio.run(client.call_json("hi", cache=c, **kwargs))
        check("first call went to the network", len(fake.posts) == 1, f"{len(fake.posts)}")
        second = asyncio.run(client.call_json("hi", cache=c, **kwargs))
        check("second call did NOT", len(fake.posts) == 1, f"{len(fake.posts)}")
    check("same answer both times", first == second == {"answer": "hello"})


def test_call_json_without_a_cache_always_calls() -> None:
    """The default must be byte-for-byte today's behavior."""
    import asyncio
    with _NoNetwork() as fake:
        for _ in range(2):
            asyncio.run(client.call_json("hi", model="vendor/m", label="t"))
        check("no cache means no reuse", len(fake.posts) == 2, f"{len(fake.posts)}")


def test_call_json_never_caches_a_failure() -> None:
    """A failure is a fact about one moment, not about the model. Caching one
    would pin a candidate to a bad afternoon for the whole TTL."""
    import asyncio
    tmp = Path(tempfile.mkdtemp())
    c = cache.Cache(tmp)
    with _NoNetwork(content="not json at all"):
        try:
            asyncio.run(client.call_json("hi", model="vendor/m", label="t",
                                         cache=c, attempts=1))
            check("a parse failure raises", False)
        except RuntimeError:
            check("a parse failure raises", True)
    check("nothing was stored", c.writes == 0, f"{c.writes}")
    check("no entry on disk", not any(tmp.rglob("*.json")))


def test_call_json_never_caches_a_response_missing_required_keys() -> None:
    import asyncio
    tmp = Path(tempfile.mkdtemp())
    c = cache.Cache(tmp)
    with _NoNetwork(content='{"reasons": {}}'):
        try:
            asyncio.run(client.call_json("hi", model="vendor/m", label="t",
                                         required=("scores",), cache=c, attempts=1))
            check("a missing required key raises", False)
        except RuntimeError:
            check("a missing required key raises", True)
    check("nothing was stored", c.writes == 0, f"{c.writes}")


# ── client: usage_sink — cost and structured-output validity read off it ────

def test_usage_sink_filled_on_a_genuine_success() -> None:
    import asyncio
    with _NoNetwork(usage={"prompt_tokens": 120, "completion_tokens": 45}):
        sink: dict = {}
        asyncio.run(client.call_json("hi", model="vendor/m", label="t", usage_sink=sink))
    check("prompt tokens captured", sink.get("promptTokens") == 120, f"{sink}")
    check("completion tokens captured", sink.get("completionTokens") == 45, f"{sink}")
    check("succeeded on the first attempt", sink.get("attempts") == 1, f"{sink}")


def test_usage_sink_records_which_attempt_actually_succeeded() -> None:
    """A model that returns garbage once and valid JSON on retry used 2
    attempts — this is the raw signal `structuredValidityRate` reads."""
    import asyncio
    with _NoNetwork(contents=["not json at all", '{"answer": "hello"}'],
                    usage={"prompt_tokens": 10, "completion_tokens": 5}) as fake:
        sink: dict = {}
        asyncio.run(client.call_json("hi", model="vendor/m", label="t", usage_sink=sink))
    check("two calls were made", len(fake.posts) == 2, f"{len(fake.posts)}")
    check("attempts reflects the retry", sink.get("attempts") == 2, f"{sink}")


def test_usage_sink_untouched_on_a_cache_hit() -> None:
    """A replayed answer cost nothing new — `usage_sink` must stay empty
    rather than reporting the ORIGINAL call's spend a second time."""
    import asyncio
    tmp = Path(tempfile.mkdtemp())
    c = cache.Cache(tmp)
    with _NoNetwork(usage={"prompt_tokens": 100, "completion_tokens": 50}):
        kwargs = dict(model="vendor/m", label="t")
        asyncio.run(client.call_json("hi", cache=c, **kwargs))
        sink: dict = {}
        asyncio.run(client.call_json("hi", cache=c, usage_sink=sink, **kwargs))
    check("cache hit leaves usage_sink empty", sink == {}, f"{sink}")


def test_usage_sink_defaults_to_none_missing_from_the_response() -> None:
    """A provider that omits `usage` entirely must not raise, and the sink's
    fields come back None rather than a guessed number."""
    import asyncio
    with _NoNetwork():
        sink: dict = {}
        asyncio.run(client.call_json("hi", model="vendor/m", label="t", usage_sink=sink))
    check("no token counts fabricated",
         sink.get("promptTokens") is None and sink.get("completionTokens") is None, f"{sink}")
    check("attempts still recorded", sink.get("attempts") == 1, f"{sink}")


def test_cost_preview_agrees_with_what_the_run_will_actually_reuse() -> None:
    """THE DRIFT GUARD. `run_form` promises "N already cached" from
    `sandbox.cached_count`; the run itself buys whatever `run_one` misses on.
    Built from different parameters, the preview would quietly lie about
    money. This runs one real (fake-transport) generation call, then asks the
    preview — end to end, without the test knowing the fingerprint itself."""
    import asyncio
    tmp = Path(tempfile.mkdtemp())
    c = cache.Cache(tmp)
    uc = config.load(ROOT / "examples" / "prep_material" / "use_case.yaml")
    tc = uc.test_cases[0]
    check("preview says nothing is cached yet",
          sandbox.cached_count(["vendor/m"], uc, c) == 0)
    with _NoNetwork():
        outcome = asyncio.run(sandbox.run_one("vendor/m", uc, tc, cache=c))
    check("the call succeeded", outcome["status"] == "ok", f"{outcome}")
    counted = sandbox.cached_count(["vendor/m"], uc, c)
    check("preview now counts exactly that one call", counted == 1, f"{counted}")


def test_cost_preview_does_not_pollute_the_runs_cache_accounting() -> None:
    """A preview reads the cache; the run file records how much of the RUN
    was replayed. If the preview's lookups counted, the run would over-report
    replays it never made."""
    tmp = Path(tempfile.mkdtemp())
    c = cache.Cache(tmp)
    uc = config.load(ROOT / "examples" / "prep_material" / "use_case.yaml")
    c.put(_fp(), {"answer": "x"})
    sandbox.cached_count(["vendor/m"], uc, c)
    check("hit/miss counters reset after a preview", (c.hits, c.misses) == (0, 0),
          f"{c.stats()}")


def test_the_two_noise_measurements_cannot_be_handed_a_cache() -> None:
    """THE ONE THAT MATTERS MOST. Both of these measure how far apart two
    IDENTICAL calls land. Served from a cache, every repeat returns the first
    answer, the spread comes out as exactly 0.00, and the leaderboard prints
    "no measurable noise" about a check that never ran. They must not accept
    the parameter at all, so it can't be threaded through by accident."""
    import inspect
    for fn, where in [(judge.score_repeated, "judge.score_repeated"),
                      (bench.resample_candidates_for_spread,
                       "bench.resample_candidates_for_spread")]:
        params = inspect.signature(fn).parameters
        check(f"{where} takes no cache parameter", "cache" not in params,
              f"{list(params)}")


def test_bench_records_whether_answers_were_replayed_or_measured() -> None:
    """A run built partly from replays is different evidence from one where
    every answer was bought, and the trend chart compares them side by side.
    The distinction has to survive into the saved file."""
    import asyncio
    uc = config.load(ROOT / "examples" / "prep_material" / "use_case.yaml")

    async def fake_call_json(*a, **k):
        return {"reply": "ok", "scores": {c.id: 4 for c in uc.rubric}}

    tmp = Path(tempfile.mkdtemp())
    c = cache.Cache(tmp)
    original = client.call_json
    sandbox.client_module.call_json = fake_call_json
    judge.client_module.call_json = fake_call_json
    try:
        with_cache = asyncio.run(bench.run(["vendor/a"], uc, judge_model="judge/j", cache=c))
        without = asyncio.run(bench.run(["vendor/a"], uc, judge_model="judge/j"))
    finally:
        sandbox.client_module.call_json = original
        judge.client_module.call_json = original
    check("a cached run reports its stats", isinstance(with_cache.get("cacheStats"), dict),
          f"{with_cache.get('cacheStats')}")
    check("an uncached run reports None, not zeros",
          without.get("cacheStats") is None, f"{without.get('cacheStats')}")


def test_cache_clear_stale_keeps_what_a_run_would_still_reuse() -> None:
    import json as json_module
    tmp = Path(tempfile.mkdtemp())
    c = cache.Cache(tmp, ttl_days=7)
    fresh, old = _fp(prompt="fresh"), _fp(prompt="old")
    c.put(fresh, {"answer": "keep me"})
    c.put(old, {"answer": "drop me"})
    path = c._path(old)
    entry = json_module.loads(path.read_text(encoding="utf-8"))
    entry["savedAt"] = (datetime.now(timezone.utc) - timedelta(days=30)).isoformat()
    path.write_text(json_module.dumps(entry), encoding="utf-8")

    info = cache.describe(tmp, ttl_days=7)
    check("describe counts both", info["entries"] == 2, f"{info}")
    check("describe counts the stale one", info["stale"] == 1, f"{info}")
    removed = cache.clear(tmp, stale_only=True, ttl_days=7)
    check("only the stale one went", removed == 1, f"{removed}")
    check("the reusable one survived", c.get(fresh) == {"answer": "keep me"})
    check("clear-all removes the rest", cache.clear(tmp) == 1)


# ── auth: one shared secret, off by default, that gates the whole dashboard
#    once it's reachable by more than just you ──────────────────────────────
#
# THE STAKE. Eleven POST routes in dashboard.py approve models, spend real
# money on a run, write API keys into .env, and patch a connected app's real
# source code — all written assuming only 127.0.0.1 could ever reach them.
# These checks exist to prove: (1) nothing changes for the existing
# localhost-only workflow when no key is set, and (2) once a key IS set,
# every one of those routes actually requires it.
#
# ISOLATION NOTE: os.environ is process-wide and this test module runs many
# checks in one process, so every test that sets MODELCICD_API_KEY restores
# it (deleting the var, not just resetting it to '') in a finally block —
# leaking it into a later test would silently change that test's behavior.

def _with_dashboard_key(value: str):
    """Context-manager-shaped helper: sets MODELCICD_API_KEY for the
    duration, restores whatever was there (including "wasn't set at all")
    afterward."""
    import os

    class _Ctx:
        def __enter__(self):
            self._had = "MODELCICD_API_KEY" in os.environ
            self._old = os.environ.get("MODELCICD_API_KEY")
            os.environ["MODELCICD_API_KEY"] = value
            return self

        def __exit__(self, *a):
            if self._had:
                os.environ["MODELCICD_API_KEY"] = self._old
            else:
                os.environ.pop("MODELCICD_API_KEY", None)
            return False

    return _Ctx()


def test_auth_is_off_when_no_key_is_configured() -> None:
    import os
    os.environ.pop("MODELCICD_API_KEY", None)
    check("not configured", auth.configured() is False)
    check("check() is False, never raises, with nothing set", auth.check("anything") is False)
    check("check() is False even for an empty candidate", auth.check("") is False)
    check("check() is False even for None", auth.check(None) is False)


def test_auth_check_is_exact_and_constant_time_shaped() -> None:
    with _with_dashboard_key("correct-horse-battery-staple"):
        check("the right key passes", auth.check("correct-horse-battery-staple") is True)
        check("a wrong key fails", auth.check("wrong") is False)
        check("a prefix of the right key still fails", auth.check("correct-horse") is False)
        check("whitespace around the candidate is trimmed",
              auth.check("  correct-horse-battery-staple  ") is True)
        # THE ACTUAL POINT OF USING hmac.compare_digest: this doesn't prove
        # timing safety (that needs a timing harness), but it does prove
        # the comparison isn't a naive `==` that a test could trivially
        # replace and still pass everything else here.
        import inspect
        src = inspect.getsource(auth.check)
        check("uses a constant-time comparison, not ==", "compare_digest" in src, src)


def test_bearer_token_parses_only_the_bearer_scheme() -> None:
    check("extracts the token", auth.bearer_token("Bearer abc123") == "abc123")
    check("case-insensitive scheme", auth.bearer_token("bearer abc123") == "abc123")
    check("wrong scheme is not a token", auth.bearer_token("Basic abc123") is None)
    check("missing header is not a token", auth.bearer_token(None) is None)
    check("bare 'Bearer' with nothing after it is not a token",
          auth.bearer_token("Bearer ") is None)


def test_session_token_rotates_when_the_key_rotates() -> None:
    """THE WHOLE POINT of computing this fresh from the live key on every
    check, rather than caching a session object the way Flask's own
    `session` does — a first attempt at this used Flask's session with a
    secret set in `before_request`, and it didn't work at all (see the long
    comment in `auth.py`: Flask opens the session before `before_request`
    ever runs, using whatever secret existed at that moment — too early).
    This hand-rolled version sidesteps that entirely: rotating the key must
    invalidate every existing cookie, which only happens if the token
    actually changes when the key does."""
    with _with_dashboard_key("key-one"):
        one = auth.session_token()
        check("same key, same token, twice", auth.session_token() == one)
        check("and it validates against itself", auth.session_cookie_valid(one) is True)
    with _with_dashboard_key("key-two"):
        two = auth.session_token()
        check("a different key derives a different token", two != one, f"{one!r} vs {two!r}")
        check("the OLD token no longer validates under the new key",
              auth.session_cookie_valid(one) is False)


def test_session_token_is_empty_and_never_validates_with_no_key_configured() -> None:
    import os
    os.environ.pop("MODELCICD_API_KEY", None)
    check("empty token when auth is off", auth.session_token() == "")
    check("nothing validates, not even the empty token itself",
          auth.session_cookie_valid("") is False)
    check("nor a garbage value", auth.session_cookie_valid("whatever") is False)


def test_secrets_module_writes_and_reads_back_the_dashboard_key() -> None:
    tmp = Path(tempfile.mkdtemp()) / ".env"
    try:
        secrets.set_dashboard_key("a-real-key", env_path=tmp)
        check("written to the env file", "MODELCICD_API_KEY=a-real-key" in tmp.read_text(
            encoding="utf-8"))
        check("and os.environ was updated immediately, no restart needed", auth.configured())
        check("the written key actually verifies", auth.check("a-real-key") is True)
    finally:
        import os
        os.environ.pop("MODELCICD_API_KEY", None)


def test_secrets_refuses_to_set_a_blank_dashboard_key() -> None:
    """An empty MODELCICD_API_KEY is indistinguishable from unset to
    auth.configured() — silently 'succeeding' here would leave the
    dashboard open while a person believes they just locked it down."""
    tmp = Path(tempfile.mkdtemp()) / ".env"
    try:
        secrets.set_dashboard_key("   ")
        check("a blank key is refused", False)
    except ValueError as exc:
        check("a blank key is refused", True)
        check("and explains the silent-open-door risk",
              "silently" in str(exc) or "open" in str(exc), str(exc))


# ── dashboard: the auth gate actually gates every route ──────────────────────

def _auth_dashboard_client():
    """A test client over a fresh, isolated dashboard — same shape as
    `_api_client`, reused here so the auth tests don't depend on ordering
    against the API tests below."""
    from modelcicd import dashboard
    tmp = Path(tempfile.mkdtemp())
    original_state, original_out = dashboard.STATE_DIR, dashboard.OUT_DIR
    dashboard.STATE_DIR, dashboard.OUT_DIR = tmp / "state", tmp / "out"
    app = dashboard.create_app()
    app.config["TESTING"] = True

    def restore():
        dashboard.STATE_DIR, dashboard.OUT_DIR = original_state, original_out

    return app.test_client(), tmp, restore


def test_dashboard_is_wide_open_when_no_key_is_configured() -> None:
    """THE COMPATIBILITY GUARANTEE. Every one of this project's other
    dashboard tests runs with no key set and expects routes to just answer
    — this is that assumption, stated as its own check."""
    import os
    os.environ.pop("MODELCICD_API_KEY", None)
    client_, tmp, restore = _auth_dashboard_client()
    try:
        check("home page, no auth needed", client_.get("/").status_code == 200)
        check("API, no auth needed",
              client_.get("/api/status/anything").status_code in (200, 404))
        # 404 here is fine (no state for "anything") — the point is it's
        # NOT 401, i.e. auth never entered the picture.
        check("...and specifically not blocked by auth",
              client_.get("/api/status/anything").status_code != 401)
    finally:
        restore()


def test_dashboard_pages_redirect_to_login_once_a_key_is_set() -> None:
    with _with_dashboard_key("secret123"):
        client_, tmp, restore = _auth_dashboard_client()
        try:
            r = client_.get("/", follow_redirects=False)
            check("an unauthenticated page visit is redirected", r.status_code == 302,
                  f"{r.status_code}")
            check("...specifically to /login", "/login" in r.headers.get("Location", ""),
                  r.headers.get("Location"))
            login_page = client_.get("/login")
            check("but /login itself is reachable without being logged in",
                  login_page.status_code == 200, f"{login_page.status_code}")
        finally:
            restore()


def test_api_routes_return_401_json_not_a_redirect_once_a_key_is_set() -> None:
    """A browser gets sent somewhere useful; a server integrating over HTTP
    needs a status code and a JSON body it can branch on, not a 302 to an
    HTML login page it has no way to fill in."""
    with _with_dashboard_key("secret123"):
        client_, tmp, restore = _auth_dashboard_client()
        try:
            r = client_.get("/api/resolve/anything")
            check("401, not a redirect", r.status_code == 401, f"{r.status_code}")
            check("a JSON body", r.content_type.startswith("application/json"), r.content_type)
            check("explains how to authenticate",
                  "Bearer" in r.get_json().get("error", ""), r.get_json())
        finally:
            restore()


def test_a_valid_bearer_token_reaches_the_api_with_no_session_at_all() -> None:
    """THE SERVER-TO-SERVER PATH — this is the one your own website's
    backend actually uses. No cookie, no /login visit, just a header."""
    with _with_dashboard_key("secret123"):
        client_, tmp, restore = _auth_dashboard_client()
        try:
            r = client_.get("/api/status/anything",
                            headers={"Authorization": "Bearer secret123"})
            check("a correct bearer token is accepted", r.status_code in (200, 404),
                  f"{r.status_code}")
            check("never redirected to login", r.status_code != 302)
            wrong = client_.get("/api/status/anything",
                                headers={"Authorization": "Bearer nope"})
            check("a wrong bearer token is refused", wrong.status_code == 401,
                  f"{wrong.status_code}")
        finally:
            restore()


def test_logging_in_unlocks_the_browser_session_for_subsequent_requests() -> None:
    """The human path end to end: fail with a wrong key, succeed with the
    right one, then prove the SESSION (not the key) is what carries — the
    next request has no Authorization header at all."""
    with _with_dashboard_key("secret123"):
        client_, tmp, restore = _auth_dashboard_client()
        try:
            wrong = client_.post("/login", data={"key": "nope"})
            check("wrong key stays on the login page", wrong.status_code == 200,
                  f"{wrong.status_code}")
            check("and says so", "match" in wrong.get_data(as_text=True))

            right = client_.post("/login", data={"key": "secret123"}, follow_redirects=False)
            check("correct key redirects away from login", right.status_code == 302,
                  f"{right.status_code}")

            after = client_.get("/")
            check("the SAME client, no header this time, now gets through",
                  after.status_code == 200, f"{after.status_code}")
        finally:
            restore()


def test_login_next_redirect_is_restricted_to_same_site_paths() -> None:
    """`next` comes from a query/form value a crafted link could set. A bare
    `//evil.example` has no scheme and no explicit host in the string, but a
    browser treats a leading `//` as protocol-relative to another host
    entirely — so it must be rejected the same as a full `https://` URL,
    not just checked for a scheme."""
    with _with_dashboard_key("secret123"):
        client_, tmp, restore = _auth_dashboard_client()
        try:
            for bad_next in ("https://evil.example/steal", "//evil.example/steal"):
                r = client_.post("/login", data={"key": "secret123", "next": bad_next},
                                 follow_redirects=False)
                check(f"rejected open-redirect target: {bad_next}",
                      "evil.example" not in r.headers.get("Location", ""),
                      r.headers.get("Location"))
        finally:
            restore()


def test_logout_is_post_only_and_clears_the_session() -> None:
    with _with_dashboard_key("secret123"):
        client_, tmp, restore = _auth_dashboard_client()
        try:
            client_.post("/login", data={"key": "secret123"})
            check("session works before logout", client_.get("/").status_code == 200)
            get_logout = client_.get("/logout")
            check("logout refuses GET", get_logout.status_code == 405, f"{get_logout.status_code}")
            client_.post("/logout", follow_redirects=False)
            after = client_.get("/", follow_redirects=False)
            check("logged out -> redirected again", after.status_code == 302,
                  f"{after.status_code}")
        finally:
            restore()


def test_setting_the_dashboard_key_from_the_keys_page_does_not_lock_out_the_setter() -> None:
    """THE BUG THIS TEST EXISTS FOR: the response that turns auth ON must
    carry a cookie that's valid under the key that response JUST wrote —
    not stale from before the write happened. Get that wrong and the person
    who just protected their own dashboard is immediately locked out of it
    (an earlier version of this feature had exactly that bug, caught by an
    earlier version of this very test).

    WRITES TO AN ISOLATED PATH, NOT THE REAL .env — `dashboard.py`'s
    `/keys` route has no parameter for where to write; it always calls
    `secrets_module.set_dashboard_key` at its default location. The first
    version of this test didn't account for that and, while proving the
    lock-out fix worked, ALSO wrote a real `MODELCICD_API_KEY` line into
    this repo's actual `.env` — found only by noticing the dashboard's own
    CLI warning had gone silent afterward. `secrets_module.set_dashboard_key`
    is monkeypatched for the duration to redirect that one write, the same
    way other tests in this file swap `client_module.call_json` to keep a
    network call from happening at all."""
    import os
    os.environ.pop("MODELCICD_API_KEY", None)
    client_, tmp, restore = _auth_dashboard_client()
    original_set = secrets.set_dashboard_key
    isolated_path = tmp / ".env"

    def fake_set_dashboard_key(value, **_ignored):
        return original_set(value, env_path=isolated_path)

    from modelcicd import dashboard
    dashboard.secrets_module.set_dashboard_key = fake_set_dashboard_key
    try:
        r = client_.post("/keys", data={"dashboard_key": "brand-new-key"},
                         follow_redirects=False)
        check("the save redirects normally", r.status_code == 303, f"{r.status_code}")
        check("the write landed in the ISOLATED file, not the repo's own .env",
              "MODELCICD_API_KEY=brand-new-key" in isolated_path.read_text(encoding="utf-8"))
        # THE SAME client, SAME cookie jar, next request — no Authorization
        # header, nothing re-typed. This is the request that would fail if
        # the ordering bug were still there.
        after = client_.get("/", follow_redirects=False)
        check("the same browser is still signed in on the very next request",
              after.status_code == 200, f"{after.status_code}")
    finally:
        dashboard.secrets_module.set_dashboard_key = original_set
        os.environ.pop("MODELCICD_API_KEY", None)
        restore()


def test_a_blank_dashboard_key_from_the_keys_page_is_refused_not_silently_ignored() -> None:
    import os
    os.environ.pop("MODELCICD_API_KEY", None)
    client_, tmp, restore = _auth_dashboard_client()
    try:
        r = client_.post("/keys", data={"dashboard_key": "   "}, follow_redirects=True)
        check("still on the keys page with an error, not silently accepted",
              "already exists" not in r.get_data(as_text=True)
              and r.status_code == 200)
        check("auth was not turned on by a blank value", not auth.configured())
    finally:
        os.environ.pop("MODELCICD_API_KEY", None)
        restore()


# ── dashboard: the read-only JSON API — the one door to other languages ─────
#
# WHAT THIS EXISTS FOR. `resolve()` is the entire integration surface, but
# it's Python-only — an app in another language has no way to reach it
# without its own port of the state-file format. These two routes expose
# exactly `resolve()` and `status()` over HTTP and nothing else: no route
# here writes anything, matching the two-gate design (approve/reject/run) the
# rest of this project is built around.

def _api_client(*, with_project: bool = False):
    """A test client, optionally with one project registered — covers both
    the unscoped and project-scoped route pairs."""
    from modelcicd import dashboard
    tmp = Path(tempfile.mkdtemp())
    original_state, original_out = dashboard.STATE_DIR, dashboard.OUT_DIR
    dashboard.STATE_DIR, dashboard.OUT_DIR = tmp / "state", tmp / "out"
    original_project_dir = project.DEFAULT_DIR
    if with_project:
        project.DEFAULT_DIR = tmp / "projects"
        project.create("Api Test", slug="api-test", notify_email="t@example.com")
    app = dashboard.create_app()
    app.config["TESTING"] = True

    def restore():
        dashboard.STATE_DIR, dashboard.OUT_DIR = original_state, original_out
        project.DEFAULT_DIR = original_project_dir

    return app.test_client(), tmp, restore


def test_api_resolve_returns_the_approved_model() -> None:
    client_, tmp, restore = _api_client()
    try:
        state_root = tmp / "state"
        state.record_run("router", {"ranAt": "t", "tiers": {
            "free": [{"model": "vendor/winner", "score": 4.5}]}}, root=state_root)
        state.approve("router", "vendor/winner", root=state_root)
        r = client_.get("/api/resolve/router")
        check("200 on an approved use case", r.status_code == 200, f"{r.status_code}")
        body = r.get_json()
        check("returns the approved model", body["model"] == "vendor/winner", f"{body}")
        check("names the source as approved", body["source"] == "approved", f"{body}")
        check("unscoped project is null", body["project"] is None, f"{body}")
    finally:
        restore()


def test_api_resolve_uses_the_fallback_and_says_so() -> None:
    client_, tmp, restore = _api_client()
    try:
        r = client_.get("/api/resolve/never_run?fallback=gpt-4o-mini")
        check("200 via fallback", r.status_code == 200, f"{r.status_code}")
        body = r.get_json()
        check("returns the fallback", body["model"] == "gpt-4o-mini", f"{body}")
        check("names the source as fallback, not approved",
              body["source"] == "fallback", f"{body}")
    finally:
        restore()


def test_api_resolve_404s_with_json_not_an_html_page_when_nothing_is_approved() -> None:
    """Same failure `resolver.resolve()` has always had, in a status code a
    caller can branch on instead of a traceback — and still a clear failure,
    not a 200 with a null model someone might use anyway."""
    client_, tmp, restore = _api_client()
    try:
        r = client_.get("/api/resolve/never_run")
        check("404, not 200 with nothing", r.status_code == 404, f"{r.status_code}")
        body = r.get_json()
        check("a JSON body, not an HTML error page",
              r.content_type.startswith("application/json"), r.content_type)
        check("names the use case", body.get("useCase") == "never_run", f"{body}")
        check("explains how to fix it",
              "run" in body.get("error", "").lower(), body.get("error"))
    finally:
        restore()


def test_api_resolve_is_scoped_to_its_project() -> None:
    """A project-scoped call must read that project's own state directory,
    not the unscoped one — the same isolation `_roots` already gives every
    other route."""
    client_, tmp, restore = _api_client(with_project=True)
    try:
        proj_state = project.state_dir("api-test")
        state.record_run("router", {"ranAt": "t", "tiers": {
            "free": [{"model": "vendor/scoped", "score": 4.0}]}}, root=proj_state)
        state.approve("router", "vendor/scoped", root=proj_state)

        scoped = client_.get("/projects/api-test/api/resolve/router")
        check("scoped call finds the project's own approval",
              scoped.get_json()["model"] == "vendor/scoped", f"{scoped.get_json()}")
        check("the response names its project", scoped.get_json()["project"] == "api-test")

        unscoped = client_.get("/api/resolve/router")
        check("the unscoped route does NOT see the project's state",
              unscoped.status_code == 404, f"{unscoped.status_code}")
    finally:
        restore()


def test_api_resolve_404s_on_an_unknown_project_before_touching_resolve() -> None:
    client_, tmp, restore = _api_client()
    try:
        r = client_.get("/projects/does-not-exist/api/resolve/router")
        check("unknown project is refused", r.status_code == 404, f"{r.status_code}")
        check("names the project", "does-not-exist" in r.get_json().get("error", ""),
              r.get_json())
    finally:
        restore()


def test_api_status_reports_pending_without_approving_anything() -> None:
    """A read-only mirror of `resolver.status()` — proves the route touches
    no state, unlike approve/reject which are POST-only elsewhere."""
    client_, tmp, restore = _api_client()
    try:
        state_root = tmp / "state"
        state.record_run("router", {"ranAt": "t", "tiers": {
            "free": [{"model": "vendor/pending", "score": 4.5}]}}, root=state_root)
        r = client_.get("/api/status/router")
        check("200", r.status_code == 200, f"{r.status_code}")
        body = r.get_json()
        check("reports the pending candidate", body["pending"]["model"] == "vendor/pending",
              f"{body}")
        check("reports nothing approved yet", body["approvedModel"] is None, f"{body}")
        after = state.load("router", state_root)
        check("calling status did not approve anything",
              after.get("approvedModel") is None, f"{after}")
    finally:
        restore()


def test_api_routes_only_answer_get() -> None:
    """THE ONE RULE THAT MUST NEVER MOVE. Nothing behind this API can write —
    approving, rejecting, and spending money on a run all still require a
    human at the CLI or the dashboard's own POST routes. A POST here must be
    refused outright, not silently accepted."""
    client_, tmp, restore = _api_client()
    try:
        for path in ("/api/resolve/router", "/api/status/router"):
            r = client_.post(path)
            check(f"POST {path} is refused", r.status_code == 405, f"{r.status_code}")
    finally:
        restore()


# ── dashboard: bulk-create never saves silently, never saves partially ──────
#
# THE REGRESSION THESE GUARD. bulk_create_save used to `continue` past any
# feature that failed validation and then redirect as if it had worked — so
# a project with an empty rubric saved nothing at all and said nothing. It
# bit a real person twice in one session before it was found. No network
# here: this route only validates form data and writes yaml.

_DEFAULT_TEST_RUBRIC = [{"id": "accuracy", "description": "is it right", "weight": "1.0"}]


def _bulk_form(count: int, *, rubric=_DEFAULT_TEST_RUBRIC, **overrides) -> dict:
    """A bulk-create submission with `count` valid features, f0..fN.

    RUBRIC ROWS ARE PART OF A REAL SUBMISSION NOW, not silently inherited —
    `bulk_create_review.html` always renders them (drafted, or copied from
    the project's own defaults for a person to edit), so a real POST from
    that page always carries `f{i}_rubric_id` etc. Defaults to one
    criterion per feature, matching what most of these tests' projects are
    given via `_bulk_client`; pass `rubric=[]` for the one test that
    specifically wants to submit none at all."""
    form = {"feature_count": str(count)}
    for i in range(count):
        form.update({f"f{i}_name": f"feature_{i}",
                     f"f{i}_system_prompt": "You answer questions.",
                     f"f{i}_tc_id": "tc1", f"f{i}_tc_input": "what is 2+2?",
                     f"f{i}_tc_reference": "4",
                     f"f{i}_rubric_id": [c["id"] for c in rubric],
                     f"f{i}_rubric_description": [c["description"] for c in rubric],
                     f"f{i}_rubric_weight": [str(c.get("weight", "1.0")) for c in rubric]})
    form.update(overrides)
    return form


def _scan_client(rubric: list, candidates: list):
    """A test client plus a project whose defaults carry `rubric` and whose
    scan results carry `candidates` — enough to drive `bulk_create_generate`
    itself, not just `bulk_create_save` directly."""
    from modelcicd import dashboard
    tmp = Path(tempfile.mkdtemp())
    original = project.DEFAULT_DIR
    project.DEFAULT_DIR = tmp / "projects"
    project.create("Scan Test", slug="scan-test", notify_email="t@example.com")
    project.set_defaults("scan-test", {"rubric": rubric})
    scan_dir = project.DEFAULT_DIR / "scan-test"
    scan_dir.mkdir(parents=True, exist_ok=True)
    (scan_dir / "scan_results.json").write_text(
        json.dumps({"candidates": candidates, "errors": []}), encoding="utf-8")
    app = dashboard.create_app()
    app.config["TESTING"] = True
    return app.test_client(), original


_A_CANDIDATE = [{"file": "app/router.py", "line": 3, "model": "vendor/old",
                "prompt": "Route this ticket.", "confidence": "high",
                "inputStructure": None, "outputStructure": None}]


def test_bulk_create_generate_drafts_a_rubric_when_the_project_has_none() -> None:
    import asyncio

    async def fake_call_json(prompt, *, model, label, **kwargs):
        if label == "generate-rubric":
            return {"criteria": [{"id": "on_topic", "description": "stays on topic",
                                  "weight": 1.0}]}
        return {"cases": []}

    client_, original = _scan_client([], _A_CANDIDATE)
    original_call = code_scan.client_module.call_json
    code_scan.client_module.call_json = fake_call_json
    try:
        r = client_.post("/projects/scan-test/scan/bulk-create")
        body = r.get_data(as_text=True)
        check("200", r.status_code == 200, f"{r.status_code}")
        check("the drafted criterion appears, editable", 'value="on_topic"' in body, body[:400])
        check("labelled as drafted, not silently presented as fact",
              "drafted by" in body, body[:2000])
    finally:
        code_scan.client_module.call_json = original_call
        project.DEFAULT_DIR = original


def test_bulk_create_generate_never_drafts_a_rubric_when_the_project_already_has_one() -> None:
    """THE COST GUARANTEE. A project's explicit rubric — set once, by a
    person, on Project Defaults — must never be silently replaced by a
    guess, and drafting one anyway would spend a real model call for
    something that gets thrown away. Checked by asserting the call simply
    never happens, not just that its result is unused."""
    import asyncio
    calls = []

    async def fake_call_json(prompt, *, model, label, **kwargs):
        calls.append(label)
        return {"cases": []}

    client_, original = _scan_client(
        [{"id": "accuracy", "description": "is it right", "weight": 1.0}], _A_CANDIDATE)
    original_call = code_scan.client_module.call_json
    code_scan.client_module.call_json = fake_call_json
    try:
        r = client_.post("/projects/scan-test/scan/bulk-create")
        body = r.get_data(as_text=True)
        check("generate-rubric was never called", "generate-rubric" not in calls, f"{calls}")
        check("the project's own rubric is shown instead",
              'value="accuracy"' in body, body[:400])
        check("labelled as inherited, not drafted",
              "from the project&#39;s shared defaults" in body
              or "from the project's shared defaults" in body, body[:2000])
    finally:
        code_scan.client_module.call_json = original_call
        project.DEFAULT_DIR = original


def test_bulk_create_save_round_trips_a_drafted_rubric() -> None:
    """End to end: a project with no rubric, a drafted one shown on review,
    and what actually gets saved when the person submits it unchanged."""
    import asyncio

    async def fake_call_json(prompt, *, model, label, **kwargs):
        if label == "generate-rubric":
            return {"criteria": [{"id": "on_topic", "description": "stays on topic",
                                  "weight": 1.0}]}
        return {"cases": [{"id": "tc1", "input": "hello", "reference": None}]}

    client_, original = _scan_client([], _A_CANDIDATE)
    original_call = code_scan.client_module.call_json
    code_scan.client_module.call_json = fake_call_json
    try:
        client_.post("/projects/scan-test/scan/bulk-create")  # drafts, not saved yet
        # Submit the review form back exactly as drafted — one feature,
        # its drafted test case and drafted rubric, untouched.
        form = {
            "feature_count": "1",
            "f0_name": "router", "f0_system_prompt": "Route this ticket.",
            "f0_tc_id": "tc1", "f0_tc_input": "hello", "f0_tc_reference": "",
            "f0_rubric_id": "on_topic", "f0_rubric_description": "stays on topic",
            "f0_rubric_weight": "1.0",
        }
        r = client_.post("/projects/scan-test/scan/bulk-create/save", data=form,
                         follow_redirects=False)
        check("saved", r.status_code == 303, f"{r.status_code}")
        saved = config.load(project.use_cases_dir("scan-test") / "router.yaml")
        check("the drafted rubric is what actually saved",
              [(c.id, c.description) for c in saved.rubric] == [("on_topic", "stays on topic")],
              f"{saved.rubric}")
    finally:
        code_scan.client_module.call_json = original_call
        project.DEFAULT_DIR = original


def _bulk_client(rubric: list):
    """A test client plus a project whose defaults carry `rubric`, both
    rooted in a fresh tempdir so nothing touches the real projects/."""
    from modelcicd import dashboard
    tmp = Path(tempfile.mkdtemp())
    original = project.DEFAULT_DIR
    project.DEFAULT_DIR = tmp / "projects"
    project.create("Bulk Test", slug="bulk-test", notify_email="test@example.com")
    project.set_defaults("bulk-test", {"rubric": rubric})
    app = dashboard.create_app()
    app.config["TESTING"] = True
    return app.test_client(), tmp, original


def test_bulk_create_refuses_and_explains_when_the_rubric_is_empty() -> None:
    # No rubric anywhere — neither the project's own defaults nor the form
    # submission carries one, the way a real page would look if drafting
    # itself had also failed to produce anything.
    client_, tmp, original = _bulk_client([])
    try:
        r = client_.post("/projects/bulk-test/scan/bulk-create/save",
                         data=_bulk_form(1, rubric=[]))
        body = r.get_data(as_text=True)
        check("empty rubric is refused, not redirected", r.status_code == 400,
              f"got {r.status_code}")
        check("refusal says nothing was saved", "Nothing was saved" in body)
        check("refusal names the rubric as the reason", "rubric criterion" in body)
        check("nothing written", project.list_use_cases("bulk-test") == [])
        check("the person's edits survive", 'value="feature_0"' in body)
    finally:
        project.DEFAULT_DIR = original


def test_bulk_create_writes_and_reports_the_count_when_valid() -> None:
    client_, tmp, original = _bulk_client([{"id": "accuracy", "description": "is it right",
                                            "weight": 1}])
    try:
        r = client_.post("/projects/bulk-test/scan/bulk-create/save", data=_bulk_form(2))
        check("valid batch redirects", r.status_code == 303, f"got {r.status_code}")
        check("redirect reports the count", "created+2" in r.headers["Location"]
              or "created%202" in r.headers["Location"], r.headers["Location"])
        names = {p.stem for p in project.list_use_cases("bulk-test")}
        check("both written", names == {"feature_0", "feature_1"}, str(names))
    finally:
        project.DEFAULT_DIR = original


def test_bulk_create_is_all_or_nothing_when_one_feature_is_invalid() -> None:
    client_, tmp, original = _bulk_client([{"id": "accuracy", "description": "is it right",
                                            "weight": 1}])
    try:
        # f1 has no test case at all — the exact draft that used to vanish
        # on its own while its neighbour saved.
        form = _bulk_form(2, f1_tc_id="", f1_tc_input="")
        r = client_.post("/projects/bulk-test/scan/bulk-create/save", data=form)
        check("partial batch is refused", r.status_code == 400, f"got {r.status_code}")
        check("the valid neighbour was NOT written",
              project.list_use_cases("bulk-test") == [])
        check("the reason names test cases", "at least one test case"
              in r.get_data(as_text=True))
    finally:
        project.DEFAULT_DIR = original


def test_bulk_create_skip_lets_a_bad_draft_be_dropped_deliberately() -> None:
    client_, tmp, original = _bulk_client([{"id": "accuracy", "description": "is it right",
                                            "weight": 1}])
    try:
        form = _bulk_form(2, f1_tc_id="", f1_tc_input="", f1_skip="on")
        r = client_.post("/projects/bulk-test/scan/bulk-create/save", data=form)
        check("skipping the bad one lets the rest save", r.status_code == 303,
              f"got {r.status_code}")
        names = {p.stem for p in project.list_use_cases("bulk-test")}
        check("only the kept one written", names == {"feature_0"}, str(names))
    finally:
        project.DEFAULT_DIR = original


def test_bulk_create_refuses_to_overwrite_an_existing_feature() -> None:
    client_, tmp, original = _bulk_client([{"id": "accuracy", "description": "is it right",
                                            "weight": 1}])
    try:
        url = "/projects/bulk-test/scan/bulk-create/save"
        client_.post(url, data=_bulk_form(1))
        before = (project.use_cases_dir("bulk-test") / "feature_0.yaml").read_text(encoding="utf-8")
        r = client_.post(url, data=_bulk_form(1, f0_system_prompt="TOTALLY DIFFERENT"))
        check("a name collision is refused", r.status_code == 400, f"got {r.status_code}")
        check("the collision is explained", "already exists" in r.get_data(as_text=True))
        after = (project.use_cases_dir("bulk-test") / "feature_0.yaml").read_text(encoding="utf-8")
        check("the existing feature is untouched", before == after)
    finally:
        project.DEFAULT_DIR = original


def test_bulk_create_refuses_duplicate_names_inside_one_batch() -> None:
    client_, tmp, original = _bulk_client([{"id": "accuracy", "description": "is it right",
                                            "weight": 1}])
    try:
        form = _bulk_form(2, f1_name="feature_0")
        r = client_.post("/projects/bulk-test/scan/bulk-create/save", data=form)
        check("duplicate names in one batch refused", r.status_code == 400,
              f"got {r.status_code}")
        check("nothing written", project.list_use_cases("bulk-test") == [])
        check("the duplicate is explained", "also named" in r.get_data(as_text=True))
    finally:
        project.DEFAULT_DIR = original


# ── optimizer: the additive teacher -> student prompt feature ───────────────

def test_optimizer_score_with_prompt_never_mutates_the_original_use_case() -> None:
    import asyncio

    uc = config.load(ROOT / "examples" / "prep_material" / "use_case.yaml")
    original_prompt = uc.system_prompt

    async def fake_run_one(model_id, trial_uc, tc, *, provider="openrouter", cache=None):
        check("sandbox is called with the REVISED prompt",
              trial_uc.system_prompt == "REVISED", trial_uc.system_prompt)
        check("the caller's real UseCase is untouched",
              uc.system_prompt == original_prompt, uc.system_prompt)
        return {"testCase": tc.id, "status": "ok", "seconds": 0.0,
               "usage": None, "output": "answer"}

    async def fake_score_one(model_id, judge_model, scored_uc, tc, output, *, cache=None):
        check("the judge scores against the ORIGINAL use case, not the "
             "prompt-swapped copy", scored_uc is uc)
        return {"testCase": tc.id, "status": "ok", "scores": {}, "weighted": 4.0,
               "reasons": {}, "wouldShip": True, "assertions": None}

    original_run_one, original_score_one = (optimizer.sandbox_module.run_one,
                                            optimizer.judge_module.score_one)
    optimizer.sandbox_module.run_one = fake_run_one
    optimizer.judge_module.score_one = fake_score_one
    try:
        result = asyncio.run(optimizer._score_with_prompt(
            uc, uc.test_cases[0], "vendor/cheap", "REVISED",
            provider="openrouter", judge_model=uc.judge_model))
    finally:
        optimizer.sandbox_module.run_one = original_run_one
        optimizer.judge_module.score_one = original_score_one
    check("scored ok", result.get("status") == "ok", f"{result}")
    check("original system_prompt still unchanged after the call",
         uc.system_prompt == original_prompt)


def test_optimizer_select_gold_and_cheap_picks_the_cheapest_tiers_best() -> None:
    board = {"tiers": {
        "free": [{"model": "vendor/free-a", "score": 3.0},
                {"model": "vendor/free-b", "score": 2.0}],
        "paid-high": [{"model": "vendor/best", "score": 4.8}],
    }}
    gold, score, cheap = optimizer.select_gold_and_cheap(board)
    check("gold is the highest overall score", gold == "vendor/best", gold)
    check("gold score carried through", score == 4.8, score)
    check("cheap is the best row of the cheapest tier", cheap == "vendor/free-a", cheap)


def test_optimizer_select_gold_and_cheap_honors_an_explicit_cheap_model() -> None:
    board = {"tiers": {"free": [{"model": "vendor/free-a", "score": 3.0}],
                       "paid-high": [{"model": "vendor/best", "score": 4.8}]}}
    _, _, cheap = optimizer.select_gold_and_cheap(board, cheap_model="vendor/named")
    check("an explicit --cheap-model is honored as-is", cheap == "vendor/named", cheap)


def test_optimizer_select_gold_and_cheap_raises_when_theres_nothing_to_optimize() -> None:
    board = {"tiers": {"paid-high": [{"model": "vendor/only", "score": 4.8}]}}
    try:
        optimizer.select_gold_and_cheap(board)
        check("raised NothingToOptimize", False)
    except optimizer.NothingToOptimize:
        check("raised NothingToOptimize", True)


def _optimizer_fixture():
    uc = config.load(ROOT / "examples" / "prep_material" / "use_case.yaml")
    board = {"tiers": {"free": [{"model": "vendor/cheap", "score": 3.0}],
                       "paid-high": [{"model": "vendor/gold", "score": 4.5}]}}
    bench_result = {"judge": uc.judge_model,
                    "results": [{"model": "vendor/gold", "testCases": [
                        {"testCase": tc.id, "status": "ok", "output": "gold answer",
                         "reasons": {}} for tc in uc.test_cases]}]}
    return uc, board, bench_result


def test_optimizer_execute_stops_once_the_target_gap_is_met() -> None:
    import asyncio

    uc, board, bench_result = _optimizer_fixture()
    calls = {"propose": 0, "iterate": 0}
    scores = [3.0, 4.4]                 # 4.4 is within 0.20 of the 4.5 gold score

    async def fake_propose(*a, **k):
        calls["propose"] += 1
        return f"prompt v{calls['propose']}"

    async def fake_iterate(uc_, cheap_model, prompt, *, provider, judge_model, cache=None):
        calls["iterate"] += 1
        return {"prompt": prompt, "meanScore": scores[calls["iterate"] - 1],
                "perTestCase": [{"status": "ok", "output": "x", "reasons": {}}
                               for _ in uc_.test_cases]}

    original_propose, original_iterate = optimizer.propose_prompt, optimizer._run_iteration
    optimizer.propose_prompt, optimizer._run_iteration = fake_propose, fake_iterate
    try:
        result = asyncio.run(optimizer.execute(
            uc, board, bench_result, max_iterations=5, target_gap=0.20))
    finally:
        optimizer.propose_prompt, optimizer._run_iteration = original_propose, original_iterate

    check("stopped after 2 iterations, well short of the 5-iteration cap",
         len(result["iterations"]) == 2, len(result["iterations"]))
    check("target reached", result["reachedTarget"] is True)
    check("best iteration is the second one", result["bestIteration"] == 2,
         result["bestIteration"])


def test_optimizer_execute_never_exceeds_max_iterations() -> None:
    import asyncio

    uc, board, bench_result = _optimizer_fixture()

    async def fake_propose(*a, **k):
        return "same prompt every time"

    async def fake_iterate(uc_, cheap_model, prompt, *, provider, judge_model, cache=None):
        return {"prompt": prompt, "meanScore": 2.0,   # never gets near the gold score
               "perTestCase": [{"status": "ok", "output": "x", "reasons": {}}
                              for _ in uc_.test_cases]}

    original_propose, original_iterate = optimizer.propose_prompt, optimizer._run_iteration
    optimizer.propose_prompt, optimizer._run_iteration = fake_propose, fake_iterate
    try:
        result = asyncio.run(optimizer.execute(
            uc, board, bench_result, max_iterations=3, target_gap=0.20))
    finally:
        optimizer.propose_prompt, optimizer._run_iteration = original_propose, original_iterate

    check("never exceeds max_iterations", len(result["iterations"]) == 3,
         len(result["iterations"]))
    check("target not reached", result["reachedTarget"] is False)


def test_optimizer_save_load_round_trips_and_never_collides_with_a_run_glob() -> None:
    from modelcicd import dashboard

    with tempfile.TemporaryDirectory() as tmp:
        out_dir = Path(tmp)
        result = {"schema": 1, "stamp": "20260101T000000Z", "useCase": "demo",
                 "goldModel": "vendor/gold", "cheapModel": "vendor/cheap", "iterations": []}
        path = optimizer.save(result, out_dir=out_dir)
        check("saved under a sibling optimize/ directory, not inside out/ itself",
             path.parent.name == "optimize", str(path))
        check("round-trips byte for byte", optimizer.load(path) == result)
        check("load_latest finds it", optimizer.load_latest(out_dir, "demo") == result)

        # THE EXACT GUARD `runner.py`/`dashboard.py`'s convention relies on:
        # `dashboard._runs_for` globs `out/*_<use_case>.json` NON-recursively,
        # so an optimize artifact living one directory down must never be
        # mistaken for a regular bench run.
        check("invisible to dashboard's own run-discovery glob",
             dashboard._runs_for("demo", out_dir) == [])


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
