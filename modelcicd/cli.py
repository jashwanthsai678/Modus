"""Model CICD — the whole loop, from the command line.

    python -m modelcicd.cli init                              # write a template
    python -m modelcicd.cli project create --name "My App"     # connect an application
    python -m modelcicd.cli wizard --project my-app             # define an AI feature interactively
    python -m modelcicd.cli catalogue                          # poll, free
    python -m modelcicd.cli shortlist --use-case examples/prep_material/use_case.yaml
    python -m modelcicd.cli run       --use-case examples/prep_material/use_case.yaml
    python -m modelcicd.cli status    --use-case examples/prep_material/use_case.yaml
    python -m modelcicd.cli approve   --use-case examples/prep_material/use_case.yaml
    python -m modelcicd.cli scheduler run-due                  # run whatever is due, once
    python -m modelcicd.cli scheduler serve                    # ... and keep checking, forever
    python -m modelcicd.cli ui                                  # local dashboard

ONLY `run` (AND A SCHEDULED `scheduler run-due`/`serve`) SPENDS MONEY. `run`
says the cost and asks before it does, unless `--yes` is passed; a scheduled
run always behaves as if `--yes` was passed, because nobody is there to ask.

`apply-code-patch` is the one command that writes into a CONNECTED APP's own
source (only if that project has `--repo-path` and the feature has a
`codeTarget`). It always previews the change and asks before writing, and is
never triggered automatically by `approve` — approving only ever changes
what `resolve()` returns.
"""
import argparse
import asyncio
import getpass
import json
import sys
from pathlib import Path
from typing import Optional

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# Windows consoles are cp1252; one character outside it raises mid-write and
# loses the leaderboard rather than the character.
for stream in (sys.stdout, sys.stderr):
    try:
        stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):            # pragma: no cover
        pass

try:
    from dotenv import load_dotenv
    if (_ROOT / ".env").exists():
        load_dotenv(_ROOT / ".env")
except ImportError:
    pass

from modelcicd import (cache as cache_module, catalogue as catalogue_module,     # noqa: E402
                       config as config_module, guardrails as guardrails_module,
                       health as health_module, onboarding as onboarding_module,
                       project as project_module, runner as runner_module,
                       state as state_module)

OUT = _ROOT / "out"
CATALOGUE_PATH = OUT / "catalogue.json"
TEMPLATE = _ROOT / "examples" / "prep_material" / "use_case.yaml"

# Kept as plain constants, not imported from dashboard.py, so `run` and
# `status` can print a dashboard link without requiring Flask to be
# installed just to answer "what URL would this be at".
DASHBOARD_HOST = "127.0.0.1"
DASHBOARD_PORT = 5000


def _load_catalogue(refresh: bool) -> dict:
    if not refresh and CATALOGUE_PATH.exists():
        cat = catalogue_module.load_or_poll(CATALOGUE_PATH, refresh=False)
        print(f"catalogue: {len(cat.get('models') or {})} model(s) from "
              f"{str(cat.get('fetchedAt'))[:19]}  (--refresh to re-poll)")
        return cat
    print("polling platforms…")
    cat = catalogue_module.load_or_poll(CATALOGUE_PATH, refresh=True)
    print(f"  {cat['counts']} -> {len(cat['models'])} logical model(s)")
    return cat


def _state_root(args) -> Optional[Path]:
    project = getattr(args, "project", None)
    return project_module.state_dir(project) if project else None


def _out_dir(args) -> Path:
    project = getattr(args, "project", None)
    return project_module.out_dir(project) if project else OUT


def cmd_init(args) -> int:
    dest = Path(args.out)
    if dest.exists() and not args.force:
        print(f"{dest} already exists — pass --force to overwrite.")
        return 2
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(TEMPLATE.read_text(encoding="utf-8"), encoding="utf-8")
    print(f"wrote a starting use case to {dest}")
    print("Edit the systemPrompt and testCases for your own application, then:")
    print(f"    python -m modelcicd.cli shortlist --use-case {dest}")
    return 0


def cmd_onboarding(args) -> int:
    print(onboarding_module.text(dashboard_host=DASHBOARD_HOST, dashboard_port=DASHBOARD_PORT))
    return 0


def cmd_set_key(args) -> int:
    from . import secrets as secrets_module

    value = args.key or getpass.getpass(f"{args.provider} API key (input hidden): ")
    if not value.strip():
        print("no key given — nothing saved.")
        return 2
    try:
        secrets_module.set_key(args.provider, value.strip())
    except ValueError as exc:
        print(str(exc))
        return 2
    print(f"{args.provider}: key saved to .env")
    return 0


def cmd_keys(args) -> int:
    from . import secrets as secrets_module
    for provider, configured in secrets_module.status().items():
        print(f"{provider:<12} {'configured' if configured else 'not set'}")
    return 0


def cmd_project_create(args) -> int:
    providers = [p.strip() for p in (args.providers or "").split(",") if p.strip()] or None
    if args.repo_url:
        print(f"cloning {args.repo_url}…")
    try:
        proj = project_module.create(args.name, description=args.description or "",
                                     notify_email=args.notify_email,
                                     repo_path=args.repo_path, repo_url=args.repo_url,
                                     providers=providers)
    except ValueError as exc:
        print(str(exc))
        return 2
    print(f"connected {proj.name!r} -> project {proj.slug!r}")
    print(f"providers   {', '.join(proj.providers)}")
    if proj.repo_path:
        print(f"repo        {proj.repo_path}  (read-only reference in the wizard; "
             f"never scanned or written to except via `apply-code-patch`)")
        print(f"scan it for AI features already in the code:")
        print(f"    python -m modelcicd.cli scan-repo --project {proj.slug}")
    print("add an AI feature to it:")
    print(f"    python -m modelcicd.cli wizard --project {proj.slug}")
    return 0


def cmd_project_set_defaults(args) -> int:
    """The shared rubric/price/judge every NEW feature in this project starts
    from — same `project.set_defaults` the dashboard's Project Defaults page
    calls, so the two can never disagree about what a project's template is.
    Only overrides the fields actually passed; everything else keeps its
    current value."""
    try:
        proj = project_module.load(args.project)
    except FileNotFoundError as exc:
        print(str(exc))
        return 2
    current = dict(proj.defaults)
    if args.max_price_in is not None:
        current["maxPriceIn"] = args.max_price_in
    if args.max_price_out is not None:
        current["maxPriceOut"] = args.max_price_out
    if args.min_context is not None:
        current["minContext"] = args.min_context
    if args.tiers is not None:
        current["tiers"] = [t.strip() for t in args.tiers.split(",") if t.strip()]
    if args.judge_model is not None:
        current["judgeModel"] = args.judge_model
    if args.generation_model is not None:
        current["generationModel"] = args.generation_model
    if args.max_tokens is not None:
        current["maxTokens"] = args.max_tokens
    if args.min_improvement is not None:
        current["minImprovement"] = args.min_improvement
    if args.rubric is not None:
        rubric = []
        for item in args.rubric.split(";"):
            parts = [p.strip() for p in item.split(":")]
            if len(parts) < 2 or not parts[0] or not parts[1]:
                continue
            weight = float(parts[2]) if len(parts) > 2 and parts[2] else 1.0
            rubric.append({"id": parts[0], "description": parts[1], "weight": weight})
        current["rubric"] = rubric

    updated = project_module.set_defaults(args.project, current)
    d = updated.defaults
    print(f"defaults for {args.project!r}:")
    print(f"  rubric           {len(d.get('rubric') or [])} criterion(criteria) — "
         f"{', '.join(c['id'] for c in d.get('rubric') or []) or '(none set)'}")
    print(f"  price ceiling    in <= ${d['maxPriceIn']:.2f}/M, out <= ${d['maxPriceOut']:.2f}/M")
    print(f"  min context      {d['minContext']:,}")
    print(f"  tiers            {', '.join(d['tiers'])}")
    print(f"  judge model      {d.get('judgeModel') or '(uses the global default)'}")
    print(f"  generation model {d.get('generationModel') or '(reuses the judge model)'}")
    print(f"  min improvement  {d['minImprovement']}")
    print(f"\nApplies to features created after this point — nothing already saved changes.")
    return 0


def cmd_project_list(args) -> int:
    projects = project_module.list_all()
    if not projects:
        print("no projects connected yet.")
        print('    python -m modelcicd.cli project create --name "My App"')
        return 0
    for p in projects:
        n = len(project_module.list_use_cases(p.slug))
        print(f"{p.slug:<24} {p.name}  ({n} AI feature(s))")
    return 0


def _scan_results_path(slug: str) -> Path:
    return project_module.DEFAULT_DIR / slug / "scan_results.json"


def cmd_wizard(args) -> int:
    from . import wizard as wizard_module

    proj = None
    if args.project:
        try:
            proj = project_module.load(args.project)
        except FileNotFoundError as exc:
            print(str(exc))
            return 2

    prefill = None
    if args.from_scan is not None:
        if not args.project:
            print("--from-scan needs --project")
            return 2
        results_path = _scan_results_path(args.project)
        if not results_path.exists():
            print(f"no scan results for {args.project!r} — run "
                 f"`scan-repo --project {args.project}` first.")
            return 2
        candidates = json.loads(results_path.read_text(encoding="utf-8")).get("candidates", [])
        if not 0 <= args.from_scan < len(candidates):
            print(f"--from-scan must be between 0 and {len(candidates) - 1}.")
            return 2
        prefill = candidates[args.from_scan]

    repo_slug = args.project if (proj and proj.repo_path) else None
    fields = wizard_module.collect_cli(repo_slug=repo_slug, prefill=prefill)
    errors = wizard_module.validate(fields)
    if errors:
        print("\ncould not save — fix the following:")
        for e in errors:
            print(f"  - {e}")
        return 2

    dest_dir = project_module.use_cases_dir(args.project) if args.project else Path("use_cases")
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / f"{fields.name.strip()}.yaml"
    dest.write_text(wizard_module.to_yaml(fields), encoding="utf-8")

    print(f"\nwrote {dest}")
    project_flag = f" --project {args.project}" if args.project else ""
    print("next:")
    print(f"    python -m modelcicd.cli shortlist --use-case {dest}{project_flag}")
    return 0


def cmd_scan_repo(args) -> int:
    from . import code_scan as code_scan_module

    try:
        proj = project_module.load(args.project)
    except FileNotFoundError as exc:
        print(str(exc))
        return 2
    if not proj.repo_path:
        print(f"project {args.project!r} has no connected repo path.")
        return 2

    files = code_scan_module.candidate_files(proj.repo_path, max_files=args.max_files)
    if not files:
        print("no candidate files found — nothing looked like it might call an LLM.")
        return 0

    print(f"candidate files : {len(files)}")
    print(f"cost            : {len(files)} model call(s), scanned with {args.scan_model}")
    for f in files[:12]:
        print(f"   {f}")
    if len(files) > 12:
        print(f"   … and {len(files) - 12} more")

    if not args.yes:
        try:
            reply = input("\nproceed? [y/N] ").strip().lower()
        except EOFError:
            reply = "n"
        if reply not in ("y", "yes"):
            print("nothing was spent.")
            return 1

    results = asyncio.run(code_scan_module.scan_repo(
        proj.repo_path, scan_model=args.scan_model, max_files=args.max_files))
    found, errors = code_scan_module.split_results(results)
    counts = code_scan_module.summary(found)

    results_path = _scan_results_path(args.project)
    results_path.parent.mkdir(parents=True, exist_ok=True)
    results_path.write_text(
        json.dumps({"candidates": found, "errors": errors}, indent=2, ensure_ascii=False),
        encoding="utf-8")

    print()
    if errors:
        print(f"{len(errors)} of {len(results) if results else len(found) + len(errors)} "
             f"file(s) could not be scanned — this is a REAL failure, not \"no LLM calls "
             f"here\":")
        for e in errors:
            print(f"   {e['file']}: {e['error']}")
        print()
    if not found:
        if not errors:
            print("no likely LLM call sites found. This is expected for code that builds its "
                 "prompts dynamically or through an agent/tool-calling framework — the scanner "
                 "reads for a literal call site, not for that kind of indirection. Define the "
                 "feature directly instead:")
        else:
            print("no candidates to show — every file that could be scanned either errored "
                 "(above) or genuinely had no LLM call. Fix the errors and scan again, or "
                 "define the feature directly:")
        print(f"    python -m modelcicd.cli wizard --project {args.project}")
        return 0

    print(f"found {counts['total']} candidate(s) — {counts['high']} high confidence, "
         f"{counts['low']} low")
    for i, c in enumerate(found):
        prompt_preview = (c.get("prompt") or "")[:80].replace("\n", " ")
        flag = "" if c.get("confidence") == "high" else "  [low confidence — verify before using]"
        print(f"[{i}] {c['file']}  model={c.get('model')}  "
             f"confidence={c.get('confidence')}{flag}")
        print(f"     {prompt_preview}")
        if c.get("inputStructure"):
            print(f"     input:  {c['inputStructure'][:100]}")
        if c.get("outputStructure"):
            print(f"     output: {c['outputStructure'][:100]}")
    if counts["high"] == 0:
        print("\nnothing here was high-confidence — worth double-checking against the file "
             "yourself, or just defining the feature directly:")
        print(f"    python -m modelcicd.cli wizard --project {args.project}")

    print(f"\nsaved to {results_path}")
    print("use one as a starting point:")
    print(f"    python -m modelcicd.cli wizard --project {args.project} --from-scan <index>")
    return 0


def cmd_catalogue(args) -> int:
    _load_catalogue(refresh=True)
    print(f"saved to {CATALOGUE_PATH}")
    return 0


def cmd_shortlist(args) -> int:
    uc = config_module.load(args.use_case)
    cat = _load_catalogue(args.refresh)
    print()
    print(config_module.describe(uc))
    result = guardrails_module.apply(cat, uc.guardrails)
    print()
    print(guardrails_module.summarise(result))
    if args.project:
        providers = project_module.load(args.project).providers
        in_scope = [m for m in result["passed"]
                   if set(m.get("providers") or []) & set(providers)]
        print(f"\nof those, {len(in_scope)} are on this project's chosen "
             f"provider(s): {', '.join(providers)}")
    return 0


def cmd_run(args) -> int:
    uc = config_module.load(args.use_case)
    cat = _load_catalogue(args.refresh)
    providers = project_module.load(args.project).providers if args.project else None
    candidates = runner_module.select_candidates(
        uc, cat, providers=providers, tier=args.tier, models=args.models, limit=args.limit)
    if not candidates:
        print("no candidates — widen the use case's guardrails or --tier.")
        return 2

    print()
    print(config_module.describe(uc))
    print(f"\ncandidates : {len(candidates)}")
    print(f"cost       : {len(candidates) * len(uc.test_cases)} generation "
          f"call(s) + {len(candidates) * len(uc.test_cases)} judge call(s)")
    if args.cache:
        print("           — MINUS whatever is already cached from an earlier run "
              "with this same prompt, input, and rubric. Answers reused this way "
              "are a REPLAY, not a fresh measurement; the saved run file records "
              "how many were.")
    if not args.no_health_check:
        print(f"           + {len(candidates) + 1} tiny probe call(s) first "
              f"(each candidate, plus the judge) to drop ids the platform has "
              f"deprecated before spending a full run discovering them. "
              f"--no-health-check to skip.")
    for c in candidates[:12]:
        print(f"   {c}")
    if len(candidates) > 12:
        print(f"   … and {len(candidates) - 12} more")
    if uc.endpoint:
        print(f"   + your live endpoint ({uc.endpoint.url}) as a baseline comparison")
    if args.resample_shortlist:
        print("           + if a tie-zone shortlist forms, each shortlisted candidate "
             "is re-generated a couple more times too (extra generation + judge calls "
             "— exact count depends on how many end up in the shortlist)")

    if not args.yes:
        try:
            reply = input("\nproceed? [y/N] ").strip().lower()
        except EOFError:
            reply = "n"
        if reply not in ("y", "yes"):
            print("nothing was spent.")
            return 1

    try:
        result = asyncio.run(runner_module.execute(
            uc, cat, candidates, state_root=_state_root(args), out_dir=_out_dir(args),
            resample_candidates=args.resample_shortlist, use_cache=args.cache,
            health_check=not args.no_health_check))
    except Exception as exc:                        # noqa: BLE001
        print(f"\nrefused or failed: {exc}")
        return 3

    health = result.get("health") or []
    if health:
        print(f"\nhealth: {health_module.summary(health)}")
        # NAMED, NOT COUNTED. A candidate dropped before the bench never
        # reaches the leaderboard, so this is the only place it's visible.
        for h in result.get("excluded") or []:
            print(f"   dropped {h['model']} — {h['detail']}")
        for h in health:
            if h["status"] in ("rate_limited", "unknown"):
                print(f"   kept {h['model']} despite a {h['status']} probe "
                      f"({h['detail'][:80]}) — not proof it's gone, so it still ran")

    print()
    print(result["report"])

    changed = result.get("measurement_changes")
    if changed:
        print(f"\nNOT COMPARABLE TO THE PREVIOUS RUN: {', '.join(changed)} changed "
              f"since then, so this score was produced by a different measuring "
              f"stick. The trend line breaks here rather than joining the two.")

    stats = result.get("cache_stats")
    if stats and stats.get("hits"):
        print(f"\ncache: {stats['hits']} of {stats['hits'] + stats['misses']} call(s) "
              f"were REPLAYED from an earlier run, not measured now "
              f"({stats['hitRate']:.0%} hit rate).")

    pending = result["state"].get("pending")
    if pending:
        print(f"\nPENDING: {pending['model']} scores {pending['score']:.2f}, "
              f"beating the approved model by the use case's threshold.")
    else:
        print("\nNo candidate beat the approved model by enough to page anyone.")

    link = (f"/projects/{args.project}/run/{result['out_path'].name}" if args.project
           else f"/run/{result['out_path'].name}")
    print(f"\nview this leaderboard : http://{DASHBOARD_HOST}:{DASHBOARD_PORT}{link}")
    print(f"                        (run `python -m modelcicd.cli ui` if it "
          f"isn't already running)")
    return 0


def _use_case_name(args) -> Optional[str]:
    """The name state.py files under, resolved consistently everywhere.

    THE BUG THIS FUNCTION FIXES. `run` names state after `uc.name` — the
    `useCase:` field INSIDE the yaml — while a first version of `status` and
    `approve` fell back to the FILENAME's stem when `--use-case-name` was not
    given. `use_case.yaml` loads as use case `prep_material_writer`, so `status`
    was silently reading and reporting on a state file nothing ever wrote to,
    and showing "(none yet)" right after a run that found and recorded a
    pending candidate. Wrong by construction, and it would never raise — every
    call would just report an empty, plausible-looking status forever.
    """
    if args.use_case_name:
        return args.use_case_name
    if args.use_case:
        return config_module.load(args.use_case).name
    return None


def cmd_status(args) -> int:
    name = _use_case_name(args)
    if not name:
        print("pass --use-case-name or --use-case")
        return 2
    state = state_module.load(name, _state_root(args))
    print(f"use case       {state['useCase']}")
    print(f"approved       {state.get('approvedModel') or '(none yet)'}")
    if state.get("approvedScore") is not None:
        print(f"  score        {state['approvedScore']}")
    print(f"  since        {state.get('approvedAt') or '-'}")
    if state.get("pending"):
        p = state["pending"]
        print(f"pending        {p['model']}  (score {p['score']}, found {p['foundAt']})")
    else:
        print("pending        (nothing)")
    print(f"history        {len(state.get('history') or [])} run(s)")
    link = (f"/projects/{args.project}/usecase/{name}" if args.project
           else f"/usecase/{name}")
    print(f"dashboard      http://{DASHBOARD_HOST}:{DASHBOARD_PORT}{link}")
    return 0


def cmd_approve(args) -> int:
    name = _use_case_name(args)
    if not name:
        print("pass --use-case-name or --use-case")
        return 2
    try:
        state = state_module.approve(name, args.model, root=_state_root(args))
    except ValueError as exc:
        print(str(exc))
        return 2
    print(f"{name}: approved {state['approvedModel']}")

    if args.project:
        uc_path = project_module.find_use_case(args.project, name)
        if uc_path:
            uc = config_module.load(uc_path)
            if uc.code_target:
                print(f"\nthis feature also has a code target ({uc.code_target.file}) "
                     f"— nothing in your connected repo has changed yet. Apply it:")
                print(f"    python -m modelcicd.cli apply-code-patch "
                     f"--use-case-name {name} --project {args.project}")
    return 0


def cmd_reject(args) -> int:
    name = _use_case_name(args)
    if not name:
        print("pass --use-case-name or --use-case")
        return 2
    try:
        state = state_module.reject(name, root=_state_root(args))
    except ValueError as exc:
        print(str(exc))
        return 2
    print(f"{name}: dismissed. resolve() is unaffected — this only clears the pending "
         f"suggestion, it never approves or changes anything.")
    if state.get("rejected"):
        print(f"  rejected  {state['rejected'][-1]['model']}")
    return 0


def cmd_pending(args) -> int:
    items = state_module.all_pending()
    if not items:
        print("nothing pending anywhere — every project is caught up.")
        return 0
    for item in items:
        where = f"{item['project']}/{item['useCase']}" if item["project"] else item["useCase"]
        p = item["pending"]
        print(f"{where:<40} {p['model']:<40} score {p['score']}  found {p['foundAt']}")
    print(f"\n{len(items)} pending. Approve or dismiss each:")
    print("    python -m modelcicd.cli approve --use-case-name <name> [--project <slug>]")
    print("    python -m modelcicd.cli reject  --use-case-name <name> [--project <slug>]")
    return 0


def cmd_apply_code_patch(args) -> int:
    from . import code_patch as code_patch_module

    try:
        proj = project_module.load(args.project)
    except FileNotFoundError as exc:
        print(str(exc))
        return 2
    if not proj.repo_path:
        print(f"project {args.project!r} has no connected repo path.")
        return 2

    uc_path = project_module.find_use_case(args.project, args.use_case_name)
    if not uc_path:
        print(f"no AI feature named {args.use_case_name!r} in project {args.project!r}.")
        return 2
    uc = config_module.load(uc_path)
    if not uc.code_target:
        print(f"{uc.name!r} has no code target — nothing to patch.")
        return 2

    state = state_module.load(uc.name, project_module.state_dir(args.project))
    new_model = state.get("approvedModel")
    if not new_model:
        print(f"nothing is approved yet for {uc.name!r} — approve a candidate first.")
        return 2
    old_model = uc.code_target.current_model
    if not old_model:
        print(f"{uc.name!r}'s codeTarget has no currentModel recorded — set it in "
             f"{uc_path} first, so there's something to search for.")
        return 2
    if old_model == new_model:
        print(f"{uc.code_target.file} already tracks {new_model!r} — nothing to do.")
        return 0

    try:
        prev = code_patch_module.preview(proj.repo_path, uc.code_target.file, old_model, new_model)
    except (ValueError, FileNotFoundError) as exc:
        print(str(exc))
        return 3

    print(f"file          {prev['file']}")
    print(f"occurrences   {prev['occurrences']}")
    print(f"change        {old_model!r} -> {new_model!r}")
    if not args.yes:
        try:
            reply = input("\napply this change to your repo? [y/N] ").strip().lower()
        except EOFError:
            reply = "n"
        if reply not in ("y", "yes"):
            print("nothing was changed.")
            return 1

    count = code_patch_module.apply(proj.repo_path, uc.code_target.file, old_model, new_model)
    code_patch_module.update_tracked_model(uc_path, new_model)
    print(f"\nreplaced {count} occurrence(s) in {uc.code_target.file}.")
    return 0


def cmd_cache(args) -> int:
    """What's cached, and how to reclaim it.

    --clear-stale IS THE ONE TO REACH FOR. It drops only entries past their
    TTL — the ones a `--cache` run would refuse to use anyway — so it costs
    nothing. --clear-all throws away answers a rubric edit is about to
    reuse, which is exactly the money the cache exists to save, so it says
    what it's about to delete and asks."""
    if args.clear_stale and args.clear_all:
        print("pick one: --clear-stale or --clear-all.")
        return 2

    info = cache_module.describe()
    print(f"\nroot     {info['root']}")
    print(f"entries  {info['entries']}")
    print(f"stale    {info['stale']}  (older than {cache_module.DEFAULT_TTL_DAYS} days — "
          f"a --cache run treats these as a miss anyway)")
    print(f"size     {info['bytes'] / 1024:.1f} KiB")

    if args.clear_stale:
        removed = cache_module.clear(stale_only=True)
        print(f"\nremoved {removed} stale entry(ies). Nothing reusable was touched.")
        return 0

    if args.clear_all:
        if not info["entries"]:
            print("\nnothing to clear.")
            return 0
        print(f"\n--clear-all deletes all {info['entries']} entry(ies), including "
              f"{info['entries'] - info['stale']} still reusable. The next --cache run "
              f"re-buys those answers with real calls.")
        try:
            reply = input("proceed? [y/N] ").strip().lower()
        except EOFError:
            reply = "n"
        if reply not in ("y", "yes"):
            print("nothing was cleared.")
            return 1
        print(f"removed {cache_module.clear()} entry(ies).")
        return 0

    print("\n--clear-stale to reclaim the expired ones, --clear-all to drop everything.")
    return 0


def cmd_scheduler_run_due(args) -> int:
    from . import scheduler as scheduler_module
    ran = scheduler_module.run_due()
    if not ran:
        print("nothing due.")
        return 0
    for r in ran:
        outcome = r.get("pending") or r.get("skipped") or "no pending candidate"
        print(f"{r['project']}/{r['useCase']}: {outcome}")
    return 0


def cmd_scheduler_serve(args) -> int:
    from . import scheduler as scheduler_module
    scheduler_module.serve(interval_minutes=args.interval_minutes)
    return 0


def cmd_ui(args) -> int:
    from . import dashboard as dashboard_module
    dashboard_module.serve(host=args.host, port=args.port)
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m modelcicd.cli",
                                description="Continuous model discovery, sandboxing, "
                                           "and human-approved promotion.")
    sub = p.add_subparsers(dest="command", required=True)

    i = sub.add_parser("init", help="write a starting use-case template")
    i.add_argument("--out", default="use_case.yaml")
    i.add_argument("--force", action="store_true")

    sub.add_parser("onboarding", help="print the guided quickstart")

    sk = sub.add_parser("set-key", help="save a platform's API key to .env")
    sk.add_argument("--provider", required=True, choices=["openrouter", "groq", "fireworks"])
    sk.add_argument("--key", default=None, help="omit to be prompted (input hidden)")

    sub.add_parser("keys", help="which platforms have a key configured")

    proj = sub.add_parser("project", help="connect / list applications")
    proj_sub = proj.add_subparsers(dest="project_command", required=True)
    pc = proj_sub.add_parser("create", help="register a new project")
    pc.add_argument("--name", required=True)
    pc.add_argument("--description", default="")
    pc.add_argument("--notify-email", required=True,
                    help="required — every AI feature in this project defaults to it")
    pc.add_argument("--repo-path", default=None,
                    help="local path to the application's repo (must exist) — "
                         "read-only reference in the wizard, and the target for "
                         "`apply-code-patch`")
    pc.add_argument("--repo-url", default=None,
                    help="a git URL to clone instead (shallow, read-only) — "
                         "mutually exclusive with --repo-path")
    pc.add_argument("--providers", default=None,
                    help="comma-separated marketplaces to search for this project's "
                         "candidates, e.g. openrouter,groq (default: openrouter)")
    proj_sub.add_parser("list", help="list connected projects")

    pd = proj_sub.add_parser("set-defaults",
                             help="set the shared rubric/price/judge every NEW AI feature "
                                  "in this project starts from — only overrides what you pass")
    pd.add_argument("--project", required=True)
    pd.add_argument("--max-price-in", type=float, default=None)
    pd.add_argument("--max-price-out", type=float, default=None)
    pd.add_argument("--min-context", type=int, default=None)
    pd.add_argument("--tiers", default=None, help="comma-separated, e.g. free,paid-low")
    pd.add_argument("--judge-model", default=None)
    pd.add_argument("--generation-model", default=None,
                    help="drafts a test case's reference answer when bulk-creating from a "
                         "scan — a separate role from the judge, worth a stronger model even "
                         "if the judge stays cheap. Blank/omitted reuses the judge model.")
    pd.add_argument("--max-tokens", type=int, default=None)
    pd.add_argument("--min-improvement", type=float, default=None)
    pd.add_argument("--rubric", default=None,
                    help="semicolon-separated criteria, each 'id:description[:weight]', "
                         "e.g. 'clarity:Is it clear?:1.0;tone:Is the tone right?:0.5' "
                         "— replaces the whole rubric, not additive")

    w = sub.add_parser("wizard", help="interactively define a new AI feature — no YAML to hand-write")
    w.add_argument("--project", default=None, help="write into this project's use_cases/ (omit for unscoped)")
    w.add_argument("--from-scan", type=int, default=None,
                  help="pre-fill from candidate <index> of this project's last scan-repo run")

    sub.add_parser("catalogue", help="poll the platforms (free)")

    sr = sub.add_parser("scan-repo",
                        help="read the connected repo with a model to find likely LLM call sites")
    sr.add_argument("--project", required=True)
    sr.add_argument("--scan-model", default="nvidia/nemotron-3-ultra-550b-a55b:free")
    sr.add_argument("--max-files", type=int, default=60)
    sr.add_argument("--yes", action="store_true", help="skip the cost confirmation")

    def uc_arg(sp):
        sp.add_argument("--use-case", "--feature", dest="use_case", required=True,
                        help="path to a use_case.yaml — same thing the dashboard calls "
                             "an 'AI feature'; --feature is accepted as an alias")
        sp.add_argument("--refresh", action="store_true")
        return sp

    sl = uc_arg(sub.add_parser("shortlist", help="apply the use case's guardrails (free)"))
    sl.add_argument("--project", default=None, help="also show how many pass this project's chosen provider(s)")

    r = uc_arg(sub.add_parser("run", help="sandbox + judge + rank the shortlist"))
    r.add_argument("--tier", choices=["free", "paid-low", "paid-mid", "paid-high"])
    r.add_argument("--models", default=None, help="comma-separated ids, explicit")
    r.add_argument("--limit", type=int, default=None)
    r.add_argument("--yes", action="store_true", help="skip the cost confirmation")
    r.add_argument("--resample-shortlist", action="store_true",
                   help="if a tie-zone shortlist forms, re-GENERATE each shortlisted "
                        "candidate's answers a couple more times each (not just "
                        "re-score the same answer) to check candidate-side variance, "
                        "not just judge-side. Spends real extra generation calls — "
                        "off by default.")
    r.add_argument("--cache", action="store_true",
                   help="reuse answers already bought for this exact prompt, input, "
                        "and rubric instead of re-buying them — what makes iterating "
                        "on a rubric nearly free. Off by default: a replayed answer "
                        "is not a fresh measurement of what that model does today. "
                        "The saved run file records how many calls were replays.")
    r.add_argument("--no-health-check", action="store_true",
                   help="skip the tiny probe call per candidate (and the judge) "
                        "that drops ids the platform has deprecated. The probe "
                        "costs a few tokens each and prevents spending a whole "
                        "run discovering a dead model — or, if the JUDGE is dead, "
                        "paying for every answer and scoring none of them.")
    r.add_argument("--project", default=None,
                  help="scope state/out and candidate providers to this connected project")

    ca = sub.add_parser("cache", help="inspect or clear the cached model responses")
    ca.add_argument("--clear-stale", action="store_true",
                    help=f"delete entries older than {cache_module.DEFAULT_TTL_DAYS} days")
    ca.add_argument("--clear-all", action="store_true",
                    help="delete every cached response (the next --cache run re-buys them)")

    s = sub.add_parser("status", help="what is approved and what is pending")
    s.add_argument("--use-case", "--feature", dest="use_case", default=None)
    s.add_argument("--use-case-name", "--feature-name", dest="use_case_name", default=None)
    s.add_argument("--project", default=None)

    a = sub.add_parser("approve", help="promote a model — this changes resolve()")
    a.add_argument("--use-case", "--feature", dest="use_case", default=None)
    a.add_argument("--use-case-name", "--feature-name", dest="use_case_name", default=None)
    a.add_argument("--model", default=None, help="defaults to whatever is pending")
    a.add_argument("--project", default=None)

    rj = sub.add_parser("reject", help="dismiss the pending candidate — never changes resolve()")
    rj.add_argument("--use-case", "--feature", dest="use_case", default=None)
    rj.add_argument("--use-case-name", "--feature-name", dest="use_case_name", default=None)
    rj.add_argument("--project", default=None)

    sub.add_parser("pending", help="everything waiting for review, across every project")

    cp = sub.add_parser("apply-code-patch",
                        help="write an approved model into its codeTarget file — previewed, confirmed")
    cp.add_argument("--use-case-name", "--feature-name", dest="use_case_name", required=True)
    cp.add_argument("--project", required=True, help="code targets only exist within a connected project")
    cp.add_argument("--yes", action="store_true", help="skip the confirmation")

    sch = sub.add_parser("scheduler", help="re-run whatever is due, on its own")
    sch_sub = sch.add_subparsers(dest="scheduler_command", required=True)
    sch_sub.add_parser("run-due", help="check every project once, run what's due, exit")
    srv = sch_sub.add_parser("serve", help="loop forever, checking on an interval")
    srv.add_argument("--interval-minutes", type=int, default=60)

    u = sub.add_parser("ui", help="launch the local dashboard")
    u.add_argument("--host", default="127.0.0.1")
    u.add_argument("--port", type=int, default=5000)
    return p


def main() -> int:
    args = build_parser().parse_args()
    if args.command == "project":
        return {"create": cmd_project_create, "list": cmd_project_list,
                "set-defaults": cmd_project_set_defaults}[args.project_command](args)
    if args.command == "scheduler":
        return {"run-due": cmd_scheduler_run_due, "serve": cmd_scheduler_serve}[args.scheduler_command](args)
    return {"init": cmd_init, "onboarding": cmd_onboarding, "set-key": cmd_set_key,
            "keys": cmd_keys, "catalogue": cmd_catalogue,
            "wizard": cmd_wizard, "scan-repo": cmd_scan_repo, "shortlist": cmd_shortlist,
            "run": cmd_run, "status": cmd_status, "approve": cmd_approve,
            "reject": cmd_reject, "pending": cmd_pending,
            "apply-code-patch": cmd_apply_code_patch, "cache": cmd_cache,
            "ui": cmd_ui}[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
