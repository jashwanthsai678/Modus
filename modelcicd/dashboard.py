"""A browser view over the files the CLI already writes — and, now, a second
door into creating them too.

MIRRORS `mlflow ui`, PLUS THE WIZARD. State (`state/*.json` or
`projects/<slug>/state/*.json`) and run results are produced by the same
`runner.execute` the CLI's `run` command calls; this module adds no second
implementation of running a bench. The two places it DOES write something new
— connecting a project, and defining an AI feature — call the exact same
`project.create` / `wizard.to_yaml` functions the CLI's `project create` and
`wizard` commands call, for the same reason `state.approve()` is the one
function both doors call to promote a model: one implementation, however it
was triggered.

PROJECT-SCOPED ROUTES ARE OPTIONAL, NOT A REWRITE. Every route that existed
before (`/usecase/<name>`, its `/approve`, `/run/<file>`) still works exactly
as it did, reading the repo-root `state/`/`out/` directories. Each gains a
`/projects/<slug>/...` sibling, registered against the SAME view function, so
there is still exactly one implementation of "render a use case" or "approve
a candidate" — just two URLs that can reach it.
"""
import json
from pathlib import Path
from typing import Optional

from flask import Flask, abort, redirect, render_template, request, url_for

from . import code_patch as code_patch_module
from . import config as config_module
from . import project as project_module
from . import state as state_module
from . import wizard as wizard_module

ROOT = Path(__file__).resolve().parent.parent
STATE_DIR = ROOT / "state"
OUT_DIR = ROOT / "out"


def _best_per_run(history: list) -> list:
    """The best score any candidate reached, per past run, oldest first — the
    ceiling that was available each time, not necessarily what got approved.
    A run where nothing could be scored contributes no point rather than a
    fabricated zero."""
    points = []
    for run in history[-12:]:
        scores = [row.get("score") for band in (run.get("tiers") or {}).values()
                  for row in band if row.get("score") is not None]
        if scores:
            points.append(max(scores))
    return points


def _sparkline(points: list, *, width: int = 90, height: int = 26, pad: int = 4
              ) -> Optional[dict]:
    """Coordinates for a minimal trend sparkline — the dataviz method's
    stat-tile 'trend': a de-emphasis line with the latest point in the
    accent. None when there are fewer than two runs to show a trend across."""
    if len(points) < 2:
        return None
    lo, hi = min(points), max(points)
    span = (hi - lo) or 1.0
    n = len(points)
    xs = [pad + i * (width - 2 * pad) / (n - 1) for i in range(n)]
    ys = [height - pad - (v - lo) * (height - 2 * pad) / span for v in points]
    return {"width": width, "height": height,
            "line": " ".join(f"{x:.1f},{y:.1f}" for x, y in zip(xs, ys)),
            "last_x": xs[-1], "last_y": ys[-1],
            "title": " → ".join(f"{v:.2f}" for v in points)}


def _use_cases_in(state_dir: Path) -> list:
    """Every use case with a state file under one directory, alphabetical,
    with a ready-to-render trend sparkline attached."""
    if not state_dir.exists():
        return []
    cases = []
    for p in sorted(state_dir.glob("*.json")):
        try:
            case = json.loads(p.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        case["sparkline"] = _sparkline(_best_per_run(case.get("history") or []))
        cases.append(case)
    return cases


def _runs_for(use_case: str, out_dir: Path) -> list:
    """Every saved run file for one use case under one directory, newest
    first. Matched by filename suffix (`runner.py` writes
    `<stamp>_<uc.name>.json`) rather than re-deriving the name's
    sanitization — the raw name is embedded in the filename exactly as it
    was written."""
    if not out_dir.exists():
        return []
    suffix = f"_{use_case}.json"
    files = [p for p in out_dir.glob("*.json") if p.name.endswith(suffix)]
    return sorted(files, key=lambda p: p.name, reverse=True)


def _scan_results_path(slug: str) -> Path:
    return project_module.DEFAULT_DIR / slug / "scan_results.json"


def _load_scan_results(slug: str) -> list:
    p = _scan_results_path(slug)
    if not p.exists():
        return []
    return json.loads(p.read_text(encoding="utf-8"))


def _fields_from_scan_candidate(proj, candidate: dict) -> "wizard_module.WizardFields":
    """The same pre-fill `cli.py wizard --from-scan` builds interactively,
    built once for the GET render instead — the person still edits/confirms
    everything in the form before it's ever saved."""
    code_file = candidate.get("file") or ""
    if proj.repo_path:
        try:
            code_file = str(Path(code_file).relative_to(Path(proj.repo_path)))
        except ValueError:
            pass
    return wizard_module.WizardFields(
        system_prompt=candidate.get("prompt") or "",
        code_file=code_file or None, code_current_model=candidate.get("model"))


def create_app() -> Flask:
    app = Flask(__name__)

    @app.template_global()
    def scoped_url(endpoint, project=None, **kwargs):
        """Builds a URL for a route that has both an unscoped and a
        project-scoped rule, without every template having to branch."""
        return url_for(endpoint, project=project, **kwargs) if project \
            else url_for(endpoint, **kwargs)

    # ── Projects landing / onboarding ───────────────────────────────────────

    @app.route("/")
    def index():
        projects = project_module.list_all()
        legacy = _use_cases_in(STATE_DIR)
        for p in projects:
            p.use_case_count = len(project_module.list_use_cases(p.slug))
        return render_template("index.html", projects=projects, legacy_cases=legacy,
                              error_flash=request.args.get("error"))

    @app.route("/projects", methods=["POST"])
    def create_project():
        name = (request.form.get("name") or "").strip()
        providers = request.form.getlist("providers") or None
        try:
            proj = project_module.create(
                name, description=(request.form.get("description") or "").strip(),
                notify_email=(request.form.get("notify_email") or "").strip() or None,
                repo_path=(request.form.get("repo_path") or "").strip() or None,
                repo_url=(request.form.get("repo_url") or "").strip() or None,
                providers=providers)
        except ValueError as exc:
            return redirect(url_for("index", error=str(exc)), code=303)
        return redirect(url_for("project_detail", slug=proj.slug), code=303)

    @app.route("/projects/<slug>")
    def project_detail(slug):
        try:
            proj = project_module.load(slug)
        except FileNotFoundError:
            abort(404, f"no project at {slug!r}")
        cases = _use_cases_in(project_module.state_dir(slug))
        return render_template("project.html", project=proj, cases=cases)

    # ── The wizard ───────────────────────────────────────────────────────────

    @app.route("/projects/<slug>/use_cases/new", methods=["GET", "POST"])
    def new_use_case(slug):
        try:
            proj = project_module.load(slug)
        except FileNotFoundError:
            abort(404, f"no project at {slug!r}")

        if request.method == "POST":
            fields = wizard_module.from_form(request.form.getlist, request.form.get)
            errors = wizard_module.validate(fields)
            if errors:
                return render_template("wizard.html", project=proj, errors=errors,
                                      fields=fields)
            dest = project_module.use_cases_dir(slug) / f"{fields.name.strip()}.yaml"
            dest.write_text(wizard_module.to_yaml(fields), encoding="utf-8")
            return redirect(url_for("project_detail", slug=slug), code=303)

        fields = None
        from_scan = request.args.get("from_scan", type=int)
        if from_scan is not None:
            candidates = _load_scan_results(slug)
            if candidates and 0 <= from_scan < len(candidates):
                fields = _fields_from_scan_candidate(proj, candidates[from_scan])
        return render_template("wizard.html", project=proj, errors=[], fields=fields)

    @app.route("/projects/<slug>/repo")
    def repo_file(slug):
        """Read-only view of one file from the project's connected repo, for
        reference while filling in the wizard — opened in a new tab, never
        the file's content itself submitted anywhere."""
        try:
            proj = project_module.load(slug)
        except FileNotFoundError:
            abort(404, f"no project at {slug!r}")
        rel = request.args.get("path", "")
        content, error = None, None
        if rel:
            try:
                content = project_module.read_repo_file(slug, rel)
            except (ValueError, FileNotFoundError) as exc:
                error = str(exc)
        return render_template("repo_file.html", project=proj, rel=rel,
                              content=content, error=error)

    # ── Repo scan — an LLM reads the code, this SPENDS MONEY, previewed ─────
    #
    # Same "show the cost, ask first" pattern as `run`. A scan only ever
    # proposes candidates into `scan_results.json`; turning one into a real
    # AI feature still goes through the wizard above, unchanged.

    @app.route("/projects/<slug>/scan")
    def scan_repo_form(slug):
        from . import code_scan as code_scan_module
        try:
            proj = project_module.load(slug)
        except FileNotFoundError:
            abort(404, f"no project at {slug!r}")
        if not proj.repo_path:
            abort(404, f"project {slug!r} has no connected repo path.")
        files = code_scan_module.candidate_files(proj.repo_path)
        return render_template("scan.html", project=proj, file_count=len(files))

    @app.route("/projects/<slug>/scan", methods=["POST"])
    def scan_repo_run(slug):
        import asyncio
        from . import code_scan as code_scan_module
        proj = project_module.load(slug)
        results = asyncio.run(code_scan_module.scan_repo(proj.repo_path))
        found = [r for r in results if not r.get("error")]
        path = _scan_results_path(slug)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(found, indent=2, ensure_ascii=False), encoding="utf-8")
        return redirect(url_for("scan_results", slug=slug), code=303)

    @app.route("/projects/<slug>/scan/results")
    def scan_results(slug):
        try:
            proj = project_module.load(slug)
        except FileNotFoundError:
            abort(404, f"no project at {slug!r}")
        candidates = _load_scan_results(slug)
        return render_template("scan_results.html", project=proj, candidates=candidates)

    # ── Use case detail / approve / run detail — scoped and unscoped ────────

    def _roots(project):
        if project:
            return project_module.state_dir(project), project_module.out_dir(project)
        return STATE_DIR, OUT_DIR

    def _code_target_info(project, name):
        """The use case's codeTarget, if this is project-scoped and it has
        one — for the "code target" card and its preview/apply routes.
        Unscoped use cases have no fixed yaml location to look this up
        from, so they never show this card."""
        if not project:
            return None, None
        uc_path = project_module.find_use_case(project, name)
        if not uc_path:
            return None, None
        uc = config_module.load(uc_path)
        return uc_path, uc.code_target

    def usecase(name, project=None):
        state_root, out_dir = _roots(project)
        state = state_module.load(name, state_root)
        known = bool(state.get("approvedModel") or state.get("pending")
                     or state.get("history"))
        if not known:
            abort(404, f"no state recorded yet for use case {name!r}")
        runs = [{"filename": p.name, "stamp": p.name.split("_", 1)[0]}
                for p in _runs_for(name, out_dir)]
        _, code_target = _code_target_info(project, name)
        return render_template("usecase.html", state=state, runs=runs, project=project,
                              code_target=code_target,
                              approved_flash=request.args.get("approved"),
                              error_flash=request.args.get("error"))

    app.add_url_rule("/usecase/<path:name>", "usecase", usecase)
    app.add_url_rule("/projects/<project>/usecase/<path:name>", "usecase", usecase)

    def approve(name, project=None):
        """Promotes the pending candidate. Calls `state.approve()` — the same
        function `cli.py approve` calls — so there is exactly one place a
        model actually gets promoted, however it was triggered."""
        # 303, not Flask's default 302: a POST->302 is ambiguous about whether
        # the redirect should be re-fetched with POST or GET (browsers guess
        # GET; other clients, e.g. curl -L, may resend the POST). 303 makes
        # "fetch this next one with GET" unambiguous for every client.
        state_root, _ = _roots(project)
        pending = state_module.load(name, state_root).get("pending")
        if not pending:
            return redirect(scoped_url("usecase", project=project, name=name,
                                       error="nothing is pending — nothing to approve"),
                            code=303)
        try:
            state_module.approve(name, pending["model"], root=state_root)
        except ValueError as exc:
            return redirect(scoped_url("usecase", project=project, name=name,
                                       error=str(exc)), code=303)
        return redirect(scoped_url("usecase", project=project, name=name,
                                   approved=pending["model"]), code=303)

    app.add_url_rule("/usecase/<path:name>/approve", "approve", approve, methods=["POST"])
    app.add_url_rule("/projects/<project>/usecase/<path:name>/approve", "approve",
                     approve, methods=["POST"])

    def run_detail(filename, project=None):
        # `.name` strips any directory component a crafted URL might smuggle
        # in — this must never resolve outside the run directory.
        _, out_dir = _roots(project)
        p = out_dir / Path(filename).name
        if not p.exists():
            abort(404, f"no run file named {filename!r}")
        data = json.loads(p.read_text(encoding="utf-8"))
        board = data.get("board") or {}
        return render_template("run.html", board=board, filename=p.name, project=project)

    app.add_url_rule("/run/<path:filename>", "run_detail", run_detail)
    app.add_url_rule("/projects/<project>/run/<path:filename>", "run_detail", run_detail)

    # ── Code patch — a SEPARATE, second confirmation from Approve ───────────
    #
    # Project-scoped only: a code target needs `project.repo_path`, which
    # only projects have. Deliberately its own click, its own confirm()
    # dialog — approving a model (state.approve, above) never on its own
    # edits a byte of the connected application's source.

    @app.route("/projects/<project>/usecase/<path:name>/code-patch")
    def code_patch_preview(project, name):
        proj = project_module.load(project)
        uc_path, code_target = _code_target_info(project, name)
        if not code_target:
            abort(404, f"{name!r} has no code target.")
        if not proj.repo_path:
            abort(404, f"project {project!r} has no connected repo path.")
        state = state_module.load(name, project_module.state_dir(project))
        new_model = state.get("approvedModel")
        preview, error = None, None
        if not new_model:
            error = "nothing is approved yet — approve a candidate first."
        elif not code_target.current_model:
            error = "this feature's codeTarget has no currentModel recorded."
        elif code_target.current_model == new_model:
            error = f"{code_target.file} already tracks {new_model!r} — nothing to do."
        else:
            try:
                preview = code_patch_module.preview(
                    proj.repo_path, code_target.file, code_target.current_model, new_model)
            except (ValueError, FileNotFoundError) as exc:
                error = str(exc)
        return render_template("code_patch.html", project=proj, name=name,
                              code_target=code_target, preview=preview, error=error)

    @app.route("/projects/<project>/usecase/<path:name>/code-patch", methods=["POST"])
    def code_patch_apply_route(project, name):
        proj = project_module.load(project)
        uc_path, code_target = _code_target_info(project, name)
        if not code_target or not proj.repo_path:
            abort(404)
        state = state_module.load(name, project_module.state_dir(project))
        new_model = state.get("approvedModel")
        old_model = code_target.current_model
        if not new_model or not old_model or old_model == new_model:
            return redirect(scoped_url("usecase", project=project, name=name,
                                       error="nothing to apply"), code=303)
        try:
            code_patch_module.preview(proj.repo_path, code_target.file, old_model, new_model)
        except (ValueError, FileNotFoundError) as exc:
            return redirect(scoped_url("usecase", project=project, name=name,
                                       error=str(exc)), code=303)
        code_patch_module.apply(proj.repo_path, code_target.file, old_model, new_model)
        code_patch_module.update_tracked_model(uc_path, new_model)
        return redirect(scoped_url("usecase", project=project, name=name,
                                   approved=f"code patched: {code_target.file}"), code=303)

    return app


def serve(host: str = "127.0.0.1", port: int = 5000) -> None:
    app = create_app()
    print(f"Model CICD dashboard: http://{host}:{port}  (Ctrl+C to stop)")
    app.run(host=host, port=port)
