"""A browser view over the files the CLI already writes.

MIRRORS `mlflow ui`. State (`state/*.json`) and run results (`out/*.json`) are
produced by `cli.py`'s own `run` command; this module adds no new source of
truth for that data, it only renders what is already on disk.

THE ONE EXCEPTION: APPROVING. The dashboard's "Approve" button calls
`state.approve()` directly — the exact function `cli.py approve` calls, not a
second implementation of it. There is still only one thing in this project
that can change what `resolver.resolve()` returns; the browser is just a
second door into the same room, POST-only and confirmed client-side before it
opens.
"""
import json
from pathlib import Path
from typing import Optional

from flask import Flask, abort, redirect, render_template, request, url_for

from . import state as state_module

ROOT = Path(__file__).resolve().parent.parent
STATE_DIR = ROOT / "state"
OUT_DIR = ROOT / "out"


def _all_use_cases() -> list:
    """Every use case with a state file on disk, alphabetical."""
    if not STATE_DIR.exists():
        return []
    cases = []
    for p in sorted(STATE_DIR.glob("*.json")):
        try:
            cases.append(json.loads(p.read_text(encoding="utf-8")))
        except (json.JSONDecodeError, OSError):
            continue
    return cases


def _runs_for(use_case: str) -> list:
    """Every saved run file for one use case, newest first.

    Matched by filename suffix (`cli.py` writes `<stamp>_<uc.name>.json`)
    rather than re-deriving `uc.name`'s sanitization — the raw name is
    embedded in the filename exactly as `run` wrote it.
    """
    if not OUT_DIR.exists():
        return []
    suffix = f"_{use_case}.json"
    files = [p for p in OUT_DIR.glob("*.json") if p.name.endswith(suffix)]
    return sorted(files, key=lambda p: p.name, reverse=True)


def create_app() -> Flask:
    app = Flask(__name__)

    @app.route("/")
    def index():
        return render_template("index.html", cases=_all_use_cases())

    @app.route("/usecase/<path:name>")
    def usecase(name):
        state = state_module.load(name)
        known = bool(state.get("approvedModel") or state.get("pending")
                     or state.get("history"))
        if not known:
            abort(404, f"no state recorded yet for use case {name!r}")
        runs = [{"filename": p.name, "stamp": p.name.split("_", 1)[0]}
                for p in _runs_for(name)]
        return render_template("usecase.html", state=state, runs=runs,
                              approved_flash=request.args.get("approved"),
                              error_flash=request.args.get("error"))

    @app.route("/usecase/<path:name>/approve", methods=["POST"])
    def approve(name):
        """Promote the pending candidate. Calls `state.approve()` — the same
        function `cli.py approve` calls — so there is exactly one place a
        model actually gets promoted, however it was triggered."""
        # 303, not Flask's default 302: a POST->302 is ambiguous about whether
        # the redirect should be re-fetched with POST or GET (browsers guess
        # GET; other clients, e.g. curl -L, may resend the POST). 303 makes
        # "fetch this next one with GET" unambiguous for every client.
        pending = state_module.load(name).get("pending")
        if not pending:
            return redirect(url_for("usecase", name=name,
                                    error="nothing is pending — nothing to approve"),
                            code=303)
        try:
            state_module.approve(name, pending["model"])
        except ValueError as exc:
            return redirect(url_for("usecase", name=name, error=str(exc)), code=303)
        return redirect(url_for("usecase", name=name, approved=pending["model"]),
                        code=303)

    @app.route("/run/<path:filename>")
    def run_detail(filename):
        # `.name` strips any directory component a crafted URL might smuggle
        # in — this must never resolve outside OUT_DIR.
        p = OUT_DIR / Path(filename).name
        if not p.exists():
            abort(404, f"no run file named {filename!r}")
        data = json.loads(p.read_text(encoding="utf-8"))
        board = data.get("board") or {}
        return render_template("run.html", board=board, filename=p.name)

    return app


def serve(host: str = "127.0.0.1", port: int = 5000) -> None:
    app = create_app()
    print(f"Model CICD dashboard: http://{host}:{port}  (Ctrl+C to stop)")
    app.run(host=host, port=port)
