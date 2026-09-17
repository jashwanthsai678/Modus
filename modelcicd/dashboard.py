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

`/api/resolve/<name>` AND `/api/status/<name>` ARE THE ONE DOOR TO OTHER
LANGUAGES. `resolver.resolve()` is Python-only; these expose exactly that
answer over HTTP, GET-only, read-only — see the section above `_roots` for
why that boundary is load-bearing, not incidental.

EVERY ROUTE IS OPEN BY DEFAULT — THAT'S THE LOCALHOST-ONLY ASSUMPTION THIS
PROJECT HAS ALWAYS MADE, MADE EXPLICIT. Set `MODELCICD_API_KEY` (see
`auth.py`) and every route here — the API and every dashboard page, POST
routes included — starts requiring it. Leave it unset and nothing about
this file's behavior changes, including every existing test that calls
`create_app()` with no key configured.
"""
import json
from pathlib import Path
from typing import Optional

from flask import Flask, abort, redirect, render_template, request, url_for

from . import auth as auth_module
from . import code_patch as code_patch_module
from . import config as config_module
from . import health as health_module
from . import onboarding as onboarding_module
from . import project as project_module
from . import resolver as resolver_module
from . import secrets as secrets_module
from . import state as state_module
from . import wizard as wizard_module

ROOT = Path(__file__).resolve().parent.parent
STATE_DIR = ROOT / "state"
OUT_DIR = ROOT / "out"


def _best_per_run(history: list) -> list:
    """The best score any candidate reached, per past run, oldest first — the
    ceiling that was available each time, not necessarily what got approved.
    A run where nothing could be scored contributes no point rather than a
    fabricated zero.

    Each point carries the fingerprint of what was being measured when it
    was recorded (see `config.measurement`), so `_sparkline` can refuse to
    connect two scores produced by different rubrics."""
    points = []
    for run in history[-12:]:
        scores = [row.get("score") for band in (run.get("tiers") or {}).values()
                  for row in band if row.get("score") is not None]
        if scores:
            # The whole fingerprint, not just the combined hash — the
            # scheme version is part of deciding comparability, and
            # `config.comparable` is the single place that decides it.
            points.append({"score": max(scores),
                           "fingerprint": run.get("measurement")})
    return points


def _sparkline(points: list, *, width: int = 90, height: int = 26, pad: int = 4
              ) -> Optional[dict]:
    """Coordinates for a minimal trend sparkline — the dataviz method's
    stat-tile 'trend': a de-emphasis line with the latest point in the
    accent. None when there are fewer than two runs to show a trend across.

    THREE STATES BETWEEN ADJACENT RUNS, NOT TWO — and getting this to two
    was my first mistake here, caught by looking at real history rather
    than a test:

        same fingerprint       comparable.  SOLID line.
        different fingerprint  provably not comparable.  GAP.
        either one absent      unknown.  DASHED line.

    The third case is the one worth being careful about. A gap asserts "the
    measurement changed"; a solid line asserts "it didn't". For a run
    recorded before fingerprints existed, neither is true — so it gets a
    dashed line, which asserts nothing and says "drawn, but unverified".
    Rendering those as gaps looked rigorous and was actually just wrong: it
    shattered every existing history into isolated dots, which reads as
    noise and trains people to ignore breaks entirely. "Unknown is never
    resolved in the flattering direction" does not mean unknown gets
    resolved in the hostile one; it means it doesn't get resolved.
    """
    if len(points) < 2:
        return None
    values = [p["score"] for p in points]
    lo, hi = min(values), max(values)
    span = (hi - lo) or 1.0
    n = len(points)
    xs = [pad + i * (width - 2 * pad) / (n - 1) for i in range(n)]
    ys = [height - pad - (v - lo) * (height - 2 * pad) / span for v in values]

    # Modelled as EDGES, not runs of points — one state per adjacent pair,
    # which is where comparability actually lives. Segments built from runs
    # of points can't express "dashed here, solid there" without splitting
    # on the same boundary twice.
    # `config.comparable` returns True / False / None, and each maps to one
    # of the three renderings. Read from there rather than compared here, so
    # this and the run page's warning can't disagree.
    verdicts = [config_module.comparable(points[i]["fingerprint"],
                                         points[i - 1]["fingerprint"])
                for i in range(1, n)]
    edges, linked = [], set()
    for i, verdict in enumerate(verdicts, start=1):
        if verdict is False:                        # provably a different scale
            continue
        edges.append({"points": f"{xs[i-1]:.1f},{ys[i-1]:.1f} {xs[i]:.1f},{ys[i]:.1f}",
                     "verified": verdict is True})
        linked.update((i - 1, i))
    breaks = sum(1 for v in verdicts if v is False)
    unverified = sum(1 for v in verdicts if v is None)

    note = ""
    if breaks:
        note += (f"  ({breaks} break{'s' if breaks != 1 else ''} — the rubric, checks, "
                f"test cases or judge changed, so those scores are not on the same scale)")
    if unverified:
        note += (f"  ({unverified} dashed — those runs carry no fingerprint, or "
                f"one from an older scheme, so comparability is unknown, not "
                f"confirmed)")

    return {"width": width, "height": height,
            "edges": edges,
            "dots": [{"x": xs[i], "y": ys[i]} for i in range(n) if i not in linked],
            "breaks": breaks, "unverified": unverified,
            "last_x": xs[-1], "last_y": ys[-1],
            "title": " → ".join(f"{v:.2f}" for v in values) + note}


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
    return json.loads(p.read_text(encoding="utf-8")).get("candidates", [])


def _load_scan_errors(slug: str) -> list:
    p = _scan_results_path(slug)
    if not p.exists():
        return []
    return json.loads(p.read_text(encoding="utf-8")).get("errors", [])


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
        code_file=code_file or None, code_current_model=candidate.get("model"),
        input_structure=candidate.get("inputStructure"),
        output_structure=candidate.get("outputStructure"))


def create_app() -> Flask:
    app = Flask(__name__)

    @app.template_global()
    def scoped_url(endpoint, project=None, **kwargs):
        """Builds a URL for a route that has both an unscoped and a
        project-scoped rule, without every template having to branch."""
        return url_for(endpoint, project=project, **kwargs) if project \
            else url_for(endpoint, **kwargs)

    # ── Auth gate — off unless MODELCICD_API_KEY is set ─────────────────────
    #
    # RUNS BEFORE EVERY ROUTE BELOW, so this is the one place "is this
    # request allowed" is decided — not a decorator someone has to remember
    # to add to each new route, which is exactly the kind of thing that gets
    # forgotten on route number twelve. See `auth.py` for the reasoning
    # behind the shape of this.

    # ONE YEAR, roughly — a browser signed into their own self-hosted
    # dashboard shouldn't need to re-paste the key weekly. Rotating the key
    # (via `/keys` or `set-dashboard-key`) invalidates every existing cookie
    # immediately regardless of this duration — see `auth.session_token`.
    _COOKIE_MAX_AGE = 365 * 24 * 3600

    def _signed_in() -> bool:
        return auth_module.session_cookie_valid(request.cookies.get(auth_module.COOKIE_NAME))

    def _safe_next(raw: Optional[str]) -> str:
        # A same-site relative path only — never redirect wherever a crafted
        # `next=` value points. `//evil.example` has no scheme and LOOKS
        # relative, but a browser treats a leading `//` as protocol-relative
        # to another host entirely, so that's rejected too, not just an
        # absolute `http(s)://` URL.
        if raw and raw.startswith("/") and not raw.startswith("//"):
            return raw
        return url_for("index")

    @app.before_request
    def _require_auth():
        if not auth_module.configured():
            return None                                  # legacy: auth is off
        if request.endpoint in ("login", "static"):
            return None

        token = auth_module.bearer_token(request.headers.get("Authorization"))
        if token and auth_module.check(token):
            return None
        if _signed_in():
            return None

        # An API caller gets a JSON body it can branch on; a browser gets
        # sent to the one page that doesn't require being already logged in.
        if request.path.startswith("/api/"):
            return {"error": "unauthorized — send Authorization: Bearer <key>"}, 401
        # `full_path` always trails with "?" even with no query string —
        # stripped so a plain page doesn't round-trip through login with a
        # stray empty query string tacked on.
        return redirect(url_for("login", next=request.full_path.rstrip("?")))

    @app.route("/login", methods=["GET", "POST"])
    def login():
        if not auth_module.configured():
            # Nothing to log into — there's no key to check a guess against.
            return redirect(url_for("index"))
        error = None
        if request.method == "POST":
            if auth_module.check(request.form.get("key")):
                resp = redirect(_safe_next(request.form.get("next")))
                resp.set_cookie(auth_module.COOKIE_NAME, auth_module.session_token(),
                               max_age=_COOKIE_MAX_AGE, httponly=True, samesite="Lax")
                return resp
            error = "that key doesn't match."
        return render_template("login.html", error=error,
                              next=request.args.get("next", ""))

    @app.route("/logout", methods=["POST"])
    def logout():
        resp = redirect(url_for("login"))
        resp.delete_cookie(auth_module.COOKIE_NAME)
        return resp

    @app.context_processor
    def _auth_status():
        # So `base.html` can show a Sign out link only when there is
        # something meaningful to sign out OF — auth configured, and this
        # browser actually holds a valid cookie (not a bearer-token caller,
        # which has no cookie to clear).
        return {"auth_enabled": auth_module.configured(), "signed_in": _signed_in()}

    # ── Home: title, the "new project" CTA, existing projects, the pitch ────

    @app.route("/")
    def index():
        projects = project_module.list_all()
        legacy = _use_cases_in(STATE_DIR)
        for p in projects:
            p.use_case_count = len(project_module.list_use_cases(p.slug))
        return render_template("index.html", projects=projects, legacy_cases=legacy,
                              quickstart=onboarding_module.text(),
                              pending_count=len(state_module.all_pending()))

    @app.route("/projects/new")
    def connect_project_form():
        return render_template("connect.html", error=request.args.get("error"),
                              key_status=secrets_module.status())

    @app.route("/pick-folder")
    def pick_folder():
        """Opens a REAL native folder-browser dialog and returns the real
        path the person picked. This only works because the dashboard's
        server and the browser viewing it are the same machine — a webpage
        can never learn a file's true path itself, that's a browser security
        rule, not something this tool can work around. If tkinter isn't
        available (not every Python install has it), this fails gracefully
        — the caller falls back to letting the path be typed by hand."""
        try:
            import tkinter
            from tkinter import filedialog
            root_window = tkinter.Tk()
            root_window.withdraw()
            root_window.attributes("-topmost", True)
            path = filedialog.askdirectory()
            root_window.destroy()
            return {"path": path or ""}
        except Exception as exc:                     # noqa: BLE001
            return {"path": "", "error": str(exc)[:200]}

    @app.route("/keys")
    def keys_form():
        return render_template("keys.html", key_status=secrets_module.status(),
                              dashboard_key_set=auth_module.configured(),
                              saved=request.args.get("saved"),
                              dashboard_error=request.args.get("dashboard_error"))

    @app.route("/keys", methods=["POST"])
    def keys_save():
        saved = []
        for provider in secrets_module.status():
            value = (request.form.get(f"{provider}_key") or "").strip()
            if value:
                secrets_module.set_key(provider, value)
                saved.append(provider)

        dashboard_value = (request.form.get("dashboard_key") or "").strip()
        resp = redirect(url_for("keys_form"), code=303)  # target filled in below
        if dashboard_value:
            try:
                secrets_module.set_dashboard_key(dashboard_value)
            except ValueError as exc:
                return redirect(url_for("keys_form", dashboard_error=str(exc)), code=303)
            saved.append("dashboard")
            # SETTING THIS THE FIRST TIME MUST NOT LOCK OUT THE BROWSER THAT
            # JUST DID IT. `session_token()` reads `MODELCICD_API_KEY` fresh
            # from `os.environ`, which `set_dashboard_key` just updated —
            # so the cookie set on THIS response already matches what the
            # very next request will check. Nothing to get the ordering of.
            resp.set_cookie(auth_module.COOKIE_NAME, auth_module.session_token(),
                           max_age=_COOKIE_MAX_AGE, httponly=True, samesite="Lax")

        resp.headers["Location"] = url_for("keys_form", saved=",".join(saved))
        return resp

    @app.route("/unscoped")
    def unscoped_use_cases():
        """The full unscoped-use-case table, always reachable — home only
        shows it inline when there are no projects yet; once you have real
        projects it's a link instead, so it stops competing for attention."""
        return render_template("legacy.html", legacy_cases=_use_cases_in(STATE_DIR))

    @app.route("/projects", methods=["POST"])
    def create_project():
        name = (request.form.get("name") or "").strip()
        providers = request.form.getlist("providers") or None
        for provider in secrets_module.status():
            value = (request.form.get(f"{provider}_key") or "").strip()
            if value:
                secrets_module.set_key(provider, value)
        try:
            proj = project_module.create(
                name, description=(request.form.get("description") or "").strip(),
                notify_email=(request.form.get("notify_email") or "").strip() or None,
                repo_path=(request.form.get("repo_path") or "").strip() or None,
                repo_url=(request.form.get("repo_url") or "").strip() or None,
                providers=providers)
        except ValueError as exc:
            return redirect(url_for("connect_project_form", error=str(exc)), code=303)
        return redirect(url_for("project_detail", slug=proj.slug), code=303)

    @app.route("/projects/<slug>")
    def project_detail(slug):
        try:
            proj = project_module.load(slug)
        except FileNotFoundError:
            abort(404, f"no project at {slug!r}")
        cases = _use_cases_in(project_module.state_dir(slug))
        known = {c.get("useCase") for c in cases}
        defined_not_run = []
        for p in project_module.list_use_cases(slug):
            if p.stem not in known:
                defined_not_run.append(p.stem)
        has_scan_results = _scan_results_path(slug).exists()
        steps = {
            "connected": True,
            "scanned": has_scan_results or not proj.repo_path,
            "defined": bool(cases or defined_not_run),
            "benched": bool(cases),
            "approved": any(c.get("approvedModel") for c in cases),
        }
        return render_template("project.html", project=proj, cases=cases,
                              defined_not_run=sorted(defined_not_run),
                              has_scan_results=has_scan_results, steps=steps,
                              created_flash=request.args.get("created"))

    # ── Overview — everything about a project's features, in one place ──────
    #
    # A PURE READ, NOTHING NEW COMPUTED. Every value shown here already lives
    # in a use_case.yaml, a state file, or a saved run file — this route just
    # gathers them into one page instead of requiring five separate visits
    # (the project page, the use case page, a run page, Project Defaults) to
    # see the full picture of one project.

    def _top_tier_snapshot(board):
        """The single most relevant tier's top 3 rows, for a compact
        per-feature snapshot — the tier holding the approved model if
        there is one, else whichever tier has the single best score.
        Keeps an overview page readable with many features on it, instead
        of dumping every tier from every run inline."""
        if not board or not board.get("tiers"):
            return None, []
        for tier, rows in board["tiers"].items():
            if any(r.get("isApproved") for r in rows):
                return tier, rows[:3]
        best_tier, best_score = None, None
        for tier, rows in board["tiers"].items():
            scored = [r for r in rows if r.get("score") is not None]
            if scored and (best_score is None or scored[0]["score"] > best_score):
                best_tier, best_score = tier, scored[0]["score"]
        if best_tier is None:
            best_tier = next(iter(board["tiers"]))
        return best_tier, board["tiers"][best_tier][:3]

    @app.route("/projects/<slug>/overview")
    def project_overview(slug):
        from . import config as config_module
        try:
            proj = project_module.load(slug)
        except FileNotFoundError:
            abort(404, f"no project at {slug!r}")

        rows = []
        for p in project_module.list_use_cases(slug):
            try:
                uc = config_module.load(p)
            except (ValueError, FileNotFoundError):
                continue
            state = state_module.load(uc.name, project_module.state_dir(slug))
            run_files = _runs_for(uc.name, project_module.out_dir(slug))
            latest_board = None
            if run_files:
                try:
                    saved = json.loads(run_files[0].read_text(encoding="utf-8"))
                    latest_board = saved.get("board")
                except (json.JSONDecodeError, OSError):
                    latest_board = None
            snapshot_tier, snapshot_rows = _top_tier_snapshot(latest_board)
            rows.append({
                "uc": uc, "state": state, "latest_board": latest_board,
                "latest_run_filename": run_files[0].name if run_files else None,
                "snapshot_tier": snapshot_tier, "snapshot_rows": snapshot_rows,
                "sparkline": _sparkline(_best_per_run(state.get("history") or [])),
            })
        return render_template("project_overview.html", project=proj, rows=rows)

    # ── Project defaults — the shared template every NEW feature starts from ─

    @app.route("/projects/<slug>/defaults", methods=["GET", "POST"])
    def project_defaults(slug):
        from . import config as config_module
        try:
            proj = project_module.load(slug)
        except FileNotFoundError:
            abort(404, f"no project at {slug!r}")

        if request.method == "POST":
            fields = wizard_module.WizardFields()
            fields.rubric = []
            c_ids = request.form.getlist("rubric_id")
            c_descs = request.form.getlist("rubric_description")
            c_weights = request.form.getlist("rubric_weight")
            for i, c_id in enumerate(c_ids):
                c_desc = c_descs[i] if i < len(c_descs) else ""
                c_weight = c_weights[i] if i < len(c_weights) else ""
                if c_id.strip() and c_desc.strip():
                    fields.rubric.append({"id": c_id.strip(), "description": c_desc,
                                          "weight": float(c_weight) if c_weight else 1.0})
            fields.assertions = wizard_module.assertions_from_form(request.form.getlist)
            # Validated before saving, so a bad regex is refused here rather
            # than written into every feature created from these defaults
            # and then failing at load.
            assertion_errors = wizard_module.validate_assertions(fields.assertions)
            if assertion_errors:
                f = wizard_module.defaults_from_project(proj.defaults)
                f.assertions = fields.assertions
                return render_template(
                    "project_defaults.html", project=proj, fields=f,
                    default_judge_model=config_module.DEFAULT_JUDGE_MODEL,
                    saved=None, errors=assertion_errors), 400
            defaults = {
                "rubric": fields.rubric,
                "assertions": fields.assertions,
                "maxPriceIn": float(request.form.get("max_price_in") or 0.50),
                "maxPriceOut": float(request.form.get("max_price_out") or 3.00),
                "minContext": int(request.form.get("min_context") or 32_000),
                "requireJson": request.form.get("require_json") == "on",
                "allowFree": request.form.get("allow_free") == "on",
                "tiers": request.form.getlist("tiers") or list(wizard_module.DEFAULT_TIERS),
                "judgeModel": request.form.get("judge_model", "").strip() or None,
                "generationModel": request.form.get("generation_model", "").strip() or None,
                "maxTokens": int(request.form.get("max_tokens") or 1200),
                "minImprovement": float(request.form.get("min_improvement") or 0.20),
            }
            project_module.set_defaults(slug, defaults)
            return redirect(url_for("project_defaults", slug=slug, saved="1"), code=303)

        f = wizard_module.defaults_from_project(proj.defaults)
        return render_template("project_defaults.html", project=proj, fields=f,
                              default_judge_model=config_module.DEFAULT_JUDGE_MODEL,
                              saved=request.args.get("saved"))

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
            wizard_module.write_yaml(dest, fields)
            return redirect(url_for("project_detail", slug=slug), code=303)

        fields = None
        from_scan = request.args.get("from_scan", type=int)
        if from_scan is not None:
            candidates = _load_scan_results(slug)
            if candidates and 0 <= from_scan < len(candidates):
                fields = _fields_from_scan_candidate(proj, candidates[from_scan])
        return render_template("wizard.html", project=proj, errors=[], fields=fields)

    # ── Editing a feature that already exists ────────────────────────────────
    #
    # THE SAME FORM, NOT A SECOND ONE. Creating and editing a feature ask for
    # exactly the same things, so `wizard.html` serves both and there is one
    # place to keep correct. Until this existed, changing a saved rubric or
    # adding a deterministic check meant hand-editing YAML — the last part of
    # the loop with no door but a text editor.
    #
    # THE NAME IS NOT EDITABLE HERE. State, run history, the approved model
    # and every saved run file are keyed on the feature's name; renaming
    # would orphan all of it while looking like a rename succeeded. Refused
    # server-side too, not just disabled in the form.

    @app.route("/projects/<project>/usecase/<path:name>/edit", methods=["GET", "POST"])
    def edit_use_case(project, name):
        from . import config as config_module
        try:
            proj = project_module.load(project)
        except FileNotFoundError:
            abort(404, f"no project at {project!r}")
        uc_path = project_module.find_use_case(project, name)
        if not uc_path:
            abort(404, f"no use_case.yaml found for {name!r} in project {project!r}.")

        existing = wizard_module.from_yaml(uc_path)

        if request.method == "POST":
            fields = wizard_module.from_form(request.form.getlist, request.form.get)
            # The form submits the name read-only; a mismatch means it was
            # edited anyway (or the page was stale), and silently writing to
            # a new file would leave two features and orphan the history.
            fields.name = existing.name
            # Per-test-case rubric/assertion overrides have no form inputs —
            # carried across so editing anything else can't delete them.
            wizard_module.carry_over_overrides(fields, existing)

            errors = wizard_module.validate(fields)
            if errors:
                return render_template("wizard.html", project=proj, errors=errors,
                                      fields=fields, editing=name), 400
            # VALIDATED BY THE REAL LOADER BEFORE REPLACING A WORKING FILE.
            # `wizard.validate` checks what a person can get wrong in the
            # form; `config.load` is what every run actually uses, and an
            # edit overwrites something that already worked. Rendered to a
            # temp file, loaded, and only then written into place.
            probe = Path(str(uc_path) + ".probe.tmp")
            try:
                probe.write_text(wizard_module.to_yaml(fields), encoding="utf-8")
                config_module.load(probe)
            except (ValueError, FileNotFoundError) as exc:
                return render_template("wizard.html", project=proj, fields=fields,
                                      editing=name,
                                      errors=[f"the saved file would not load: {exc}"]), 400
            finally:
                try:
                    probe.unlink()
                except OSError:
                    pass

            wizard_module.write_yaml(uc_path, fields)
            return redirect(scoped_url("usecase", project=project, name=name,
                                       ran="saved — the next run uses these settings"),
                            code=303)

        # What editing this will cost in comparability, stated before the
        # edit rather than discovered as a broken trend line afterwards.
        state = state_module.load(name, project_module.state_dir(project))
        return render_template("wizard.html", project=proj, errors=[], fields=existing,
                              editing=name, run_count=len((state.get("history") or [])))

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
        found, errors = code_scan_module.split_results(results)
        path = _scan_results_path(slug)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"candidates": found, "errors": errors},
                                   indent=2, ensure_ascii=False), encoding="utf-8")
        return redirect(url_for("scan_results", slug=slug), code=303)

    @app.route("/projects/<slug>/scan/results")
    def scan_results(slug):
        from . import code_scan as code_scan_module
        try:
            proj = project_module.load(slug)
        except FileNotFoundError:
            abort(404, f"no project at {slug!r}")
        candidates = _load_scan_results(slug)
        errors = _load_scan_errors(slug)
        return render_template("scan_results.html", project=proj, candidates=candidates,
                              errors=errors, summary=code_scan_module.summary(candidates))

    # ── Bulk-create — every scan candidate at once, sharing the project's
    # defaults, test cases drafted by a model — never saved without review.
    #
    # TWO STEPS, LIKE SCAN ITSELF. Step 1 shows the cost and asks (this
    # spends real money: one generation call per candidate). Step 2 shows
    # every draft — name, prompt, generated test cases — fully editable,
    # and ONLY the explicit "Create N AI features" submit on THAT page
    # writes anything. Nothing here ever saves a file a person hasn't seen.

    @app.route("/projects/<slug>/scan/bulk-create")
    def bulk_create_form(slug):
        from . import config as config_module
        try:
            proj = project_module.load(slug)
        except FileNotFoundError:
            abort(404, f"no project at {slug!r}")
        candidates = _load_scan_results(slug)
        if not candidates:
            return redirect(url_for("scan_results", slug=slug), code=303)
        judge_model = proj.defaults.get("judgeModel") or config_module.DEFAULT_JUDGE_MODEL
        generation_model = proj.defaults.get("generationModel") or judge_model
        # CHECKED BEFORE ANY MONEY IS SPENT, not after. A feature can't be
        # saved without a rubric, so with an empty project rubric EVERY draft
        # below would be generated (one paid call each) and then refused at
        # the last step — which is exactly what happened to a real person
        # before this check existed: drafts vanished with no explanation.
        return render_template("bulk_create.html", project=proj, candidates=candidates,
                              generation_model=generation_model,
                              rubric_missing=not (proj.defaults.get("rubric") or []))

    @app.route("/projects/<slug>/scan/bulk-create", methods=["POST"])
    def bulk_create_generate(slug):
        import asyncio
        from . import code_scan as code_scan_module
        from . import config as config_module
        proj = project_module.load(slug)
        candidates = _load_scan_results(slug)
        judge_model = proj.defaults.get("judgeModel") or config_module.DEFAULT_JUDGE_MODEL
        generation_model = proj.defaults.get("generationModel") or judge_model
        existing = {p.stem for p in project_module.list_use_cases(slug)}

        async def draft(candidate: dict) -> dict:
            cases = await code_scan_module.generate_test_cases(
                candidate.get("prompt") or "", input_structure=candidate.get("inputStructure"),
                output_structure=candidate.get("outputStructure"), model=generation_model)
            return {"candidate": candidate, "cases": cases}

        async def draft_all() -> list:
            return await asyncio.gather(*(draft(c) for c in candidates))

        drafts = asyncio.run(draft_all())

        fields_list = []
        used_names = set(existing)
        for d in drafts:
            candidate = d["candidate"]
            base = code_scan_module.suggest_name(candidate.get("file") or "ai_feature")
            name = base
            n = 2
            while name in used_names:
                name = f"{base}_{n}"
                n += 1
            used_names.add(name)

            code_file = candidate.get("file") or ""
            if proj.repo_path:
                try:
                    code_file = str(Path(code_file).relative_to(Path(proj.repo_path)))
                except ValueError:
                    pass

            f = wizard_module.defaults_from_project(proj.defaults)
            f.name = name
            f.system_prompt = candidate.get("prompt") or ""
            f.test_cases = [{"id": c["id"], "input": c["input"], "reference": c.get("reference")}
                            for c in (d["cases"] or [])]
            f.input_structure = candidate.get("inputStructure")
            f.output_structure = candidate.get("outputStructure")
            if candidate.get("model"):
                f.code_file = code_file or None
                f.code_current_model = candidate.get("model")
            fields_list.append(f)

        return render_template("bulk_create_review.html", project=proj,
                              fields_list=fields_list, generation_model=generation_model,
                              errors_list=[[] for _ in fields_list],
                              skipped_list=[False for _ in fields_list],
                              blocked_count=0,
                              rubric_missing=not (proj.defaults.get("rubric") or []))

    @app.route("/projects/<slug>/scan/bulk-create/save", methods=["POST"])
    def bulk_create_save(slug):
        try:
            proj = project_module.load(slug)
        except FileNotFoundError:
            abort(404, f"no project at {slug!r}")
        from . import config as config_module
        count = int(request.form.get("feature_count") or 0)
        # ALL-OR-NOTHING, AND NEVER SILENT. This route used to `continue`
        # past any feature that failed validation and then redirect as if
        # everything worked — so a project with an empty rubric saved
        # nothing at all and said nothing about it. Now: every feature is
        # validated FIRST, and if any one of them fails, none are written
        # and the review page comes back with the person's own edits intact
        # and the reason on the offending card.
        entries = []
        for i in range(count):
            prefix = f"f{i}_"
            f = wizard_module.defaults_from_project(proj.defaults)
            f.name = request.form.get(f"{prefix}name", "").strip()
            f.system_prompt = request.form.get(f"{prefix}system_prompt", "")
            ids = request.form.getlist(f"{prefix}tc_id")
            inputs = request.form.getlist(f"{prefix}tc_input")
            refs = request.form.getlist(f"{prefix}tc_reference")
            for j, tc_id in enumerate(ids):
                tc_input = inputs[j] if j < len(inputs) else ""
                tc_ref = refs[j] if j < len(refs) else ""
                if tc_id.strip() and tc_input.strip():
                    f.test_cases.append({"id": tc_id.strip(), "input": tc_input,
                                         "reference": tc_ref.strip() or None})
            code_file = request.form.get(f"{prefix}code_file", "").strip()
            code_model = request.form.get(f"{prefix}code_current_model", "").strip()
            if code_file and code_model:
                f.code_file, f.code_current_model = code_file, code_model
            f.input_structure = request.form.get(f"{prefix}input_structure", "").strip() or None
            f.output_structure = request.form.get(f"{prefix}output_structure", "").strip() or None
            skipped = request.form.get(f"{prefix}skip") == "on"
            entries.append({"fields": f, "skipped": skipped,
                            "errors": [] if skipped else wizard_module.validate(f)})

        # Name collisions are a validation failure too, not an overwrite. A
        # drafted name matching a feature that already exists would silently
        # replace someone's hand-edited use_case.yaml.
        seen = set()
        for e in entries:
            if e["skipped"] or e["errors"]:
                continue
            name = e["fields"].name
            if name in seen:
                e["errors"].append(f"another feature in this batch is also named {name!r} — "
                                   "rename one of them.")
            elif (project_module.use_cases_dir(slug) / f"{name}.yaml").exists():
                e["errors"].append(f"a feature named {name!r} already exists in this project — "
                                   "rename this one, or skip it to keep the existing one.")
            else:
                seen.add(name)

        blocked = sum(1 for e in entries if e["errors"])
        if blocked:
            judge_model = proj.defaults.get("judgeModel") or config_module.DEFAULT_JUDGE_MODEL
            return render_template(
                "bulk_create_review.html", project=proj,
                fields_list=[e["fields"] for e in entries],
                errors_list=[e["errors"] for e in entries],
                skipped_list=[e["skipped"] for e in entries],
                generation_model=proj.defaults.get("generationModel") or judge_model,
                blocked_count=blocked,
                rubric_missing=not (proj.defaults.get("rubric") or [])), 400

        written = []
        for e in entries:
            if e["skipped"]:
                continue
            f = e["fields"]
            dest = project_module.use_cases_dir(slug) / f"{f.name}.yaml"
            wizard_module.write_yaml(dest, f)
            written.append(f.name)
        if not written:
            return redirect(url_for("bulk_create_form", slug=slug), code=303)
        created = f"created {len(written)} AI feature(s): " + ", ".join(written)
        return redirect(url_for("project_detail", slug=slug, created=created), code=303)

    # ── Use case detail / approve / run detail — scoped and unscoped ────────

    def _roots(project):
        if project:
            return project_module.state_dir(project), project_module.out_dir(project)
        return STATE_DIR, OUT_DIR

    # ── Read-only JSON API — the one door this project opens to OTHER
    #    languages ────────────────────────────────────────────────────────
    #
    # THE ENTIRE INTEGRATION SURFACE, OVER HTTP INSTEAD OF AN IMPORT.
    # `resolver.resolve()` is deliberately the only thing an application
    # needs to call from Python — this exposes exactly that, and nothing
    # else, so a non-Python app doesn't need its own port of the state-file
    # format to get the same one answer. Nothing here writes anything:
    # approving, rejecting, and spending money on a run all still require a
    # human at the CLI or the dashboard, which is the two-gate design this
    # whole project is built around. Widening this to accept a POST would be
    # widening WHO can approve or spend, and that is a different, much
    # bigger decision than "let another language read what's approved" —
    # these routes are GET-only and there is no plan to make them anything
    # else.
    #
    # SAFE BY CONSTRUCTION, NOT BY A PERMISSIONS CHECK. This project has no
    # auth story at all — self-hosted, single-tenant, local-only by default
    # — so "safe" here means "read-only", never "authenticated". If this
    # dashboard is ever exposed beyond localhost, that is an operator's
    # decision to put a reverse proxy in front of it, the same as any other
    # local dev server; this file doesn't pretend to solve that.

    def _api_error(message: str, *, status: int, **extra):
        # A CLEAR JSON BODY, NOT AN HTML ABORT PAGE. `abort()` renders
        # Flask's default HTML error page, which is fine for a person in a
        # browser and useless for a caller parsing a response over HTTP.
        return {"error": message, **extra}, status

    def api_resolve(name, project=None):
        if project and not project_module.exists(project):
            return _api_error(f"no project at {project!r}", status=404, project=project)
        state_root, _ = _roots(project)
        fallback = request.args.get("fallback") or None
        try:
            model = resolver_module.resolve(name, fallback=fallback, root=state_root)
        except LookupError as exc:
            # THE SAME FAILURE MODE `resolver.resolve()` HAS ALWAYS HAD, put
            # into a status code instead of a raised exception — a caller
            # integrating over HTTP needs something to branch on, not a
            # traceback. Still fails loudly: 404, not a 200 with a null
            # model that a careless caller might use anyway.
            return _api_error(str(exc), status=404, useCase=name, project=project)
        state = state_module.load(name, state_root)
        return {"model": model, "useCase": name, "project": project,
                # WHICH ANSWER THIS WAS, so a caller can tell "a human
                # approved this" from "nothing is approved and you're
                # seeing your own fallback echoed back" without having to
                # separately call /api/status.
                "source": "approved" if state.get("approvedModel") else "fallback"}

    app.add_url_rule("/api/resolve/<path:name>", "api_resolve", api_resolve)
    app.add_url_rule("/projects/<project>/api/resolve/<path:name>", "api_resolve", api_resolve)

    def api_status(name, project=None):
        if project and not project_module.exists(project):
            return _api_error(f"no project at {project!r}", status=404, project=project)
        state_root, _ = _roots(project)
        result = resolver_module.status(name, root=state_root)
        return {**result, "project": project}

    app.add_url_rule("/api/status/<path:name>", "api_status", api_status)
    app.add_url_rule("/projects/<project>/api/status/<path:name>", "api_status", api_status)

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
                     or state.get("history") or state.get("rejected"))
        if not known:
            abort(404, f"no state recorded yet for use case {name!r}")
        runs = [{"filename": p.name, "stamp": p.name.split("_", 1)[0]}
                for p in _runs_for(name, out_dir)]
        _, code_target = _code_target_info(project, name)
        return render_template("usecase.html", state=state, runs=runs, project=project,
                              code_target=code_target,
                              approved_flash=request.args.get("approved"),
                              dismissed_flash=request.args.get("dismissed"),
                              error_flash=request.args.get("error"),
                              ran_flash=request.args.get("ran"))

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

    def reject(name, project=None):
        """Dismisses the pending candidate. Calls `state.reject()` — never
        `state.approve()` — so this can never be the thing that changes what
        `resolve()` returns, only what's left waiting for review."""
        state_root, _ = _roots(project)
        try:
            state_module.reject(name, root=state_root)
        except ValueError as exc:
            return redirect(scoped_url("usecase", project=project, name=name,
                                       error=str(exc)), code=303)
        return redirect(scoped_url("usecase", project=project, name=name,
                                   dismissed="1"), code=303)

    app.add_url_rule("/usecase/<path:name>/reject", "reject", reject, methods=["POST"])
    app.add_url_rule("/projects/<project>/usecase/<path:name>/reject", "reject",
                     reject, methods=["POST"])

    @app.route("/pending")
    def pending_list():
        return render_template("pending.html", items=state_module.all_pending())

    def run_detail(filename, project=None):
        from . import config as config_module
        from . import rank as rank_module
        # `.name` strips any directory component a crafted URL might smuggle
        # in — this must never resolve outside the run directory.
        _, out_dir = _roots(project)
        p = out_dir / Path(filename).name
        if not p.exists():
            abort(404, f"no run file named {filename!r}")
        data = json.loads(p.read_text(encoding="utf-8"))
        board = data.get("board") or {}
        bench = data.get("bench") or {}

        uc = None
        if project:
            uc_path = project_module.find_use_case(project, board.get("useCase", ""))
            if uc_path:
                try:
                    uc = config_module.load(uc_path)
                except (ValueError, FileNotFoundError):
                    uc = None

        # THE ACTUAL PROOF, NOT JUST THE NUMBER. A score means nothing to a
        # reader who can't see what earned it — this is every candidate's
        # real answer to every real test case, exactly what the judge saw,
        # already sitting in the saved run file and never shown until now.
        tc_inputs = {tc.id: tc.input for tc in uc.test_cases} if uc else {}
        detail_by_model = {r["model"]: r.get("testCases") or [] for r in bench.get("results") or []}

        price_sensitivity = request.args.get("price_sensitivity", type=float) or 0.0
        tiers = board.get("tiers") or {}
        if price_sensitivity:
            max_price_out = uc.guardrails.max_price_out if uc else None
            if max_price_out is None:
                # Unscoped run, or the use case's yaml couldn't be found —
                # fall back to the highest price actually seen in this run
                # rather than fail; still self-consistent, just normalized
                # against this run's own field instead of a stated ceiling.
                observed = [r.get("price_out") for band in tiers.values() for r in band
                           if r.get("price_out") is not None]
                max_price_out = max(observed) if observed else 0.0
            tiers = rank_module.reorder_by_preference(
                board, price_sensitivity=price_sensitivity, max_price_out=max_price_out)

        # WHAT WAS MEASURED, AND WHETHER IT MATCHES THE RUN BEFORE THIS ONE.
        # Read from the use case's own history rather than by opening the
        # adjacent run file, so this still works when older runs have been
        # cleaned up — history keeps the fingerprint even after the run
        # file is gone.
        measurement = board.get("measurement")
        drifted_from_previous = []
        state_root, _ = _roots(project)
        history = state_module.load(board.get("useCase", ""), state_root).get("history") or []
        for i, entry in enumerate(history):
            if entry.get("ranAt") == board.get("ranAt") and i > 0:
                drifted_from_previous = config_module.measurement_changes(
                    measurement, history[i - 1].get("measurement"))
                break
        # A pre-fingerprint run has nothing to show and nothing to compare —
        # reported as unknown, never as "unchanged".
        # Candidates the health probe dropped before the bench — they have no
        # leaderboard row, so this is the only place they appear at all.
        health = data.get("health") or []
        judge_id = board.get("judge")
        excluded = [h for h in health
                    if h.get("status") == health_module.FATAL and h.get("model") != judge_id]
        kept_despite = [h for h in health
                        if h.get("status") in ("rate_limited", "unknown")]
        return render_template("run.html", board=board, tiers=tiers, filename=p.name,
                              project=project, price_sensitivity=price_sensitivity,
                              tc_inputs=tc_inputs, detail_by_model=detail_by_model,
                              excluded=excluded, kept_despite=kept_despite,
                              measurement=measurement,
                              measurement_labels=config_module.MEASUREMENT_LABELS,
                              drifted_from_previous=drifted_from_previous)

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

    # ── Run a bench — the one action that spends real money, previewed ──────
    #
    # SAME COST-PREVIEW-AND-CONFIRM PATTERN AS SCAN AND BULK-CREATE. GET shows
    # how many candidates the CURRENT tier/limit selection resolves to (free —
    # only reads the cached catalogue) and the resulting cost; only the POST
    # on that page spends anything. Calls the exact same `runner.execute` the
    # CLI's `run` command calls — one implementation, this is just a second
    # door into it. Project-scoped only: an unscoped use case has no
    # remembered path back to its own use_case.yaml to re-run.

    def _catalogue_path() -> Path:
        return OUT_DIR / "catalogue.json"

    def _run_candidates(uc, proj, *, tier, limit):
        from . import catalogue as catalogue_module
        from . import runner as runner_module
        cat = catalogue_module.load_or_poll(_catalogue_path(), refresh=False)
        candidates = runner_module.select_candidates(
            uc, cat, providers=proj.providers, tier=tier or None, limit=limit)
        return cat, candidates

    @app.route("/projects/<project>/usecase/<path:name>/run")
    def run_form(project, name):
        from . import config as config_module
        proj = project_module.load(project)
        uc_path = project_module.find_use_case(project, name)
        if not uc_path:
            abort(404, f"no use_case.yaml found for {name!r} in project {project!r}.")
        uc = config_module.load(uc_path)
        tier = request.args.get("tier") or "free"
        limit = request.args.get("limit", type=int)
        cat, candidates = _run_candidates(uc, proj, tier=tier, limit=limit)
        # WHAT THE CACHE WOULD ACTUALLY SAVE, counted from local files — no
        # call, nothing spent. A cost preview that quotes the full price
        # while half of it is already paid for is worse than no preview.
        from . import cache as cache_module
        from . import catalogue as catalogue_module
        from . import sandbox as sandbox_module
        already_cached = sandbox_module.cached_count(
            candidates, uc, cache_module.Cache(),
            provider_by_model=catalogue_module.provider_map(cat))
        return render_template("run_form.html", project=proj, name=name, uc=uc,
                              candidates=candidates, tier=tier, limit=limit,
                              tiers=uc.guardrails.tiers, error=request.args.get("error"),
                              already_cached=already_cached)

    @app.route("/projects/<project>/usecase/<path:name>/run", methods=["POST"])
    def run_execute(project, name):
        import asyncio
        from . import config as config_module
        from . import runner as runner_module
        proj = project_module.load(project)
        uc_path = project_module.find_use_case(project, name)
        if not uc_path:
            abort(404, f"no use_case.yaml found for {name!r} in project {project!r}.")
        uc = config_module.load(uc_path)
        tier = request.form.get("tier") or "free"
        limit = request.form.get("limit", type=int)
        cat, candidates = _run_candidates(uc, proj, tier=tier, limit=limit)
        if not candidates:
            return redirect(scoped_url("usecase", project=project, name=name,
                                       error="no candidates survived the guardrails "
                                             "for that tier/limit — nothing to run"),
                            code=303)
        use_cache = request.form.get("use_cache") == "on"
        health_check = request.form.get("health_check") == "on"
        try:
            result = asyncio.run(runner_module.execute(
                uc, cat, candidates, state_root=project_module.state_dir(project),
                out_dir=project_module.out_dir(project), use_cache=use_cache,
                health_check=health_check))
        except Exception as exc:                        # noqa: BLE001
            return redirect(url_for("run_form", project=project, name=name,
                                    tier=tier, limit=limit, error=str(exc)), code=303)
        pending = (result.get("state") or {}).get("pending")
        excluded = result.get("excluded") or []
        flash = f"benched {len(candidates) - len(excluded)} candidate(s)"
        if excluded:
            # NAMED IN THE FLASH. A candidate dropped before the bench isn't
            # on the leaderboard, so without this it just isn't there.
            flash += (f" — dropped {len(excluded)} the platform no longer has: "
                      f"{', '.join(h['model'] for h in excluded)}")
        changed = result.get("measurement_changes")
        if changed:
            # SAID AT THE MOMENT IT HAPPENS. Someone who just edited a rubric
            # and re-ran is the one person who can still remember why the
            # number moved; telling them here is worth more than a broken
            # line they find three runs later.
            flash += (f" — NOT comparable to the previous run: "
                      f"{', '.join(changed)} changed since then")
        stats = result.get("cache_stats")
        if stats and stats.get("hits"):
            # SAID OUT LOUD, not buried in the run file. A person who ticked
            # "reuse cached answers" should be told how much of the result
            # they're looking at was replayed rather than measured.
            flash += (f" ({stats['hits']} of {stats['hits'] + stats['misses']} call(s) "
                      f"replayed from cache, not measured now)")
        if pending:
            flash += f" — {pending['model']} is now pending review"
        return redirect(scoped_url("usecase", project=project, name=name, ran=flash), code=303)

    return app


def serve(host: str = "127.0.0.1", port: int = 5000) -> None:
    app = create_app()
    print(f"Modus dashboard: http://{host}:{port}  (Ctrl+C to stop)")
    app.run(host=host, port=port)
