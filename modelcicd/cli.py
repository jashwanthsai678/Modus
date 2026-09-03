"""Model CICD — the whole loop, from the command line.

    python -m modelcicd.cli init                              # write a template
    python -m modelcicd.cli catalogue                          # poll, free
    python -m modelcicd.cli shortlist --use-case examples/prep_material/use_case.yaml
    python -m modelcicd.cli run       --use-case examples/prep_material/use_case.yaml
    python -m modelcicd.cli status    --use-case examples/prep_material/use_case.yaml
    python -m modelcicd.cli approve   --use-case examples/prep_material/use_case.yaml

ONLY `run` SPENDS MONEY, and it says the cost and asks before it does, unless
`--yes` is passed for use in a scheduled job.
"""
import argparse
import asyncio
import json
import sys
from datetime import datetime, timezone
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

from modelcicd import (bench as bench_module, catalogue as catalogue_module,   # noqa: E402
                       config as config_module, guardrails as guardrails_module,
                       notify as notify_module, rank as rank_module,
                       state as state_module)

OUT = _ROOT / "out"
CATALOGUE_PATH = OUT / "catalogue.json"
TEMPLATE = _ROOT / "examples" / "prep_material" / "use_case.yaml"


def _load_catalogue(refresh: bool) -> dict:
    if not refresh and CATALOGUE_PATH.exists():
        cat = catalogue_module.load(CATALOGUE_PATH)
        print(f"catalogue: {len(cat.get('models') or {})} model(s) from "
              f"{str(cat.get('fetchedAt'))[:19]}  (--refresh to re-poll)")
        return cat
    print("polling platforms…")
    cat = catalogue_module.poll()
    catalogue_module.save(cat, CATALOGUE_PATH)
    print(f"  {cat['counts']} -> {len(cat['models'])} logical model(s)")
    return cat


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
    return 0


def _candidates(args, uc, cat: dict) -> list:
    if args.models:
        return [m.strip() for m in args.models.split(",") if m.strip()]
    result = guardrails_module.apply(cat, uc.guardrails)
    passed = guardrails_module.within_tiers(
        result["passed"], [args.tier] if args.tier else uc.guardrails.tiers)
    if args.limit:
        passed = passed[:args.limit]
    return [cid for m in passed if (cid := catalogue_module.cheapest_host_id(m))]


def cmd_run(args) -> int:
    uc = config_module.load(args.use_case)
    cat = _load_catalogue(args.refresh)
    candidates = _candidates(args, uc, cat)
    if not candidates:
        print("no candidates — widen the use case's guardrails or --tier.")
        return 2

    print()
    print(config_module.describe(uc))
    print(f"\ncandidates : {len(candidates)}")
    print(f"cost       : {len(candidates) * len(uc.test_cases)} generation "
          f"call(s) + {len(candidates) * len(uc.test_cases)} judge call(s)")
    for c in candidates[:12]:
        print(f"   {c}")
    if len(candidates) > 12:
        print(f"   … and {len(candidates) - 12} more")

    if not args.yes:
        try:
            reply = input("\nproceed? [y/N] ").strip().lower()
        except EOFError:
            reply = "n"
        if reply not in ("y", "yes"):
            print("nothing was spent.")
            return 1

    try:
        result = asyncio.run(bench_module.run(candidates, uc))
    except Exception as exc:                        # noqa: BLE001
        print(f"\nrefused or failed: {exc}")
        return 3

    state = state_module.load(uc.name)
    board = rank_module.build(result, cat, approved_model=state.get("approvedModel"))
    text = rank_module.report(board)
    print()
    print(text)

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{stamp}_{uc.name}.json").write_text(
        json.dumps({"bench": result, "board": board}, indent=2, ensure_ascii=False),
        encoding="utf-8")

    new_state = state_module.record_run(uc.name, board)
    if new_state.get("pending"):
        p = new_state["pending"]
        print(f"\nPENDING: {p['model']} scores {p['score']:.2f}, beating the "
              f"approved model by the use case's threshold.")
        notify_module.send_pending(uc.name, new_state, text, to=uc.notify.email)
    else:
        print("\nNo candidate beat the approved model by enough to page anyone.")
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
    state = state_module.load(name)
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
    return 0


def cmd_approve(args) -> int:
    name = _use_case_name(args)
    if not name:
        print("pass --use-case-name or --use-case")
        return 2
    try:
        state = state_module.approve(name, args.model)
    except ValueError as exc:
        print(str(exc))
        return 2
    print(f"{name}: approved {state['approvedModel']}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m modelcicd.cli",
                                description="Continuous model discovery, sandboxing, "
                                           "and human-approved promotion.")
    sub = p.add_subparsers(dest="command", required=True)

    i = sub.add_parser("init", help="write a starting use-case template")
    i.add_argument("--out", default="use_case.yaml")
    i.add_argument("--force", action="store_true")

    sub.add_parser("catalogue", help="poll the platforms (free)")

    def uc_arg(sp):
        sp.add_argument("--use-case", required=True, help="path to a use_case.yaml")
        sp.add_argument("--refresh", action="store_true")
        return sp

    uc_arg(sub.add_parser("shortlist", help="apply the use case's guardrails (free)"))

    r = uc_arg(sub.add_parser("run", help="sandbox + judge + rank the shortlist"))
    r.add_argument("--tier", choices=["free", "paid-low", "paid-mid", "paid-high"])
    r.add_argument("--models", default=None, help="comma-separated ids, explicit")
    r.add_argument("--limit", type=int, default=None)
    r.add_argument("--yes", action="store_true", help="skip the cost confirmation")

    s = sub.add_parser("status", help="what is approved and what is pending")
    s.add_argument("--use-case", default=None)
    s.add_argument("--use-case-name", default=None)

    a = sub.add_parser("approve", help="promote a model — this changes resolve()")
    a.add_argument("--use-case", default=None)
    a.add_argument("--use-case-name", default=None)
    a.add_argument("--model", default=None, help="defaults to whatever is pending")
    return p


def main() -> int:
    args = build_parser().parse_args()
    return {"init": cmd_init, "catalogue": cmd_catalogue, "shortlist": cmd_shortlist,
            "run": cmd_run, "status": cmd_status, "approve": cmd_approve}[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
