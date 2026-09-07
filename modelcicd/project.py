"""A connected application — a named registration, plus an optional read-only
window into its actual code.

A PROJECT IS OTHERWISE DELIBERATELY LIGHTWEIGHT. Connecting an application
does not clone a repo or scan any code, and most of its fields (name,
description, notify email) are never validated — that would add setup
friction for data nothing else depends on. `repo_path` IS validated, because
it is the one field this module actually reads from (and, via
`code_patch.py`, eventually writes into): a project with a bad path would
fail confusingly later, on someone else's schedule, instead of clearly now.
Every AI feature's actual data (system prompt, test cases, rubric, guardrails,
its own optional live endpoint, its own optional code target) is entered
through the wizard afterward and lives in that feature's own `use_case.yaml`,
exactly as it already did before projects existed.

ONE DIRECTORY PER PROJECT, FLAT FILES UNDERNEATH — same philosophy as the rest
of this project (see `state.py`, `config.py`): no database, because there is
one operator, not many tenants to isolate.

EVERY FUNCTION TAKES AN OPTIONAL `root`, LIKE `state.py` DOES. Not a style
nicety — it is what lets the test suite exercise `create`/`load`/`list_all`
against a tempdir instead of writing real project directories into this
repo's own `projects/`.

`repo_path` READS ARE CONTAINMENT-CHECKED, ALWAYS. `read_repo_file` resolves
the requested path and refuses anything that lands outside `repo_path` —
a project's repo is a WINDOW, not a way to read arbitrary files on disk.
"""
import re
import shutil
import subprocess
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import yaml

DEFAULT_DIR = Path(__file__).resolve().parent.parent / "projects"
DEFAULT_PROVIDERS = ["openrouter"]

# Every field here is one a person would otherwise have to re-type in the
# wizard for EVERY feature in the same project — the rubric, the price
# ceiling, the judge, how big a win has to be to notify anyone. Deliberately
# NOT here: name, description, system prompt, test cases, endpoint, schedule,
# code target — anything that is inherently specific to one feature, not
# shared across a project. Set once via `set_defaults`, applied by
# `wizard.defaults_from_project`, always still editable per feature.
DEFAULT_PROJECT_DEFAULTS = {
    "rubric": [],
    "maxPriceIn": 0.50, "maxPriceOut": 3.00, "minContext": 32_000,
    "requireJson": True, "allowFree": True, "tiers": ["free", "paid-low", "paid-mid"],
    "judgeModel": None,   # None here means "use config.DEFAULT_JUDGE_MODEL"
    "maxTokens": 1200, "minImprovement": 0.20,
}


@dataclass
class Project:
    slug: str
    name: str
    description: str = ""
    notify_email: Optional[str] = None
    repo_path: Optional[str] = None
    providers: list = field(default_factory=lambda: list(DEFAULT_PROVIDERS))
    defaults: dict = field(default_factory=lambda: dict(DEFAULT_PROJECT_DEFAULTS))
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


def slugify(name: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", name.strip().lower()).strip("-")
    return slug or "project"


def _dir(slug: str, root: Optional[Path] = None) -> Path:
    return Path(root or DEFAULT_DIR) / slug


def _yaml_path(slug: str, root: Optional[Path] = None) -> Path:
    return _dir(slug, root) / "project.yaml"


def use_cases_dir(slug: str, root: Optional[Path] = None) -> Path:
    return _dir(slug, root) / "use_cases"


def state_dir(slug: str, root: Optional[Path] = None) -> Path:
    return _dir(slug, root) / "state"


def out_dir(slug: str, root: Optional[Path] = None) -> Path:
    return _dir(slug, root) / "out"


def list_use_cases(slug: str, root: Optional[Path] = None) -> list:
    """Every use_case.yaml registered under this project, alphabetical."""
    d = use_cases_dir(slug, root)
    return sorted(d.glob("*.yaml")) if d.exists() else []


def exists(slug: str, root: Optional[Path] = None) -> bool:
    return _yaml_path(slug, root).exists()


def _repo_dir(slug: str, root: Optional[Path] = None) -> Path:
    return _dir(slug, root) / "repo"


def _clone(url: str, dest: Path) -> str:
    """Shallow, read-only clone of `url` into `dest`. Relies entirely on
    whatever the local git/OS credential helper already provides — no
    prompting, no token storage here. Raises with git's own stderr on
    failure, and never leaves a partial clone behind."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        ["git", "clone", "--depth", "1", url, str(dest)],
        capture_output=True, text=True, timeout=300)
    if result.returncode != 0:
        if dest.exists():
            shutil.rmtree(dest, ignore_errors=True)
        raise ValueError(f"git clone failed: {result.stderr.strip()[:500]}")
    return str(dest.resolve())


def create(name: str, *, description: str = "", notify_email: Optional[str] = None,
          repo_path: Optional[str] = None, repo_url: Optional[str] = None,
          providers: Optional[list] = None, slug: Optional[str] = None,
          root: Optional[Path] = None) -> Project:
    """Registers a new project. Fails loudly if the slug is already taken —
    silently overwriting another project's registration would orphan its
    use cases from the landing page without deleting anything, which is a
    worse failure than a clear error up front. Fails loudly, too, if
    `repo_path` doesn't exist — the one field here that actually gets read
    from later, so a typo is worth catching now, not on some future run.

    `notify_email` IS REQUIRED — every AI feature in this project defaults
    to it for pending-candidate notifications, and a project with nowhere
    for that to go means the notify step silently degrades to printing at
    a terminal nobody's watching. Enforced here, not just in the dashboard
    form, so the CLI and the GUI can never disagree about it.

    `repo_path` and `repo_url` are mutually exclusive: a local folder you
    already have, or a URL modelcicd clones itself (shallow) — never both."""
    if not name or not name.strip():
        raise ValueError("a project needs a name.")
    if not notify_email or not notify_email.strip():
        raise ValueError("a project needs a notify email — every AI feature in it "
                         "defaults to this for pending-candidate notifications.")
    if repo_path and repo_url:
        raise ValueError("give either --repo-path or --repo-url, not both.")
    s = slug or slugify(name)
    if exists(s, root):
        raise ValueError(f"a project already exists at slug {s!r} — "
                         f"pick a different name, or use it as-is.")

    resolved_repo = None
    if repo_path:
        resolved_repo = Path(repo_path)
        if not resolved_repo.is_dir():
            raise ValueError(f"repo path {repo_path!r} is not an existing directory.")
        resolved_repo = str(resolved_repo.resolve())
    elif repo_url:
        resolved_repo = _clone(repo_url, _repo_dir(s, root))

    proj = Project(slug=s, name=name.strip(), description=description or "",
                   notify_email=notify_email or None, repo_path=resolved_repo,
                   providers=list(providers) if providers else list(DEFAULT_PROVIDERS))
    for d in (use_cases_dir(s, root), state_dir(s, root), out_dir(s, root)):
        d.mkdir(parents=True, exist_ok=True)
    _save(proj, root)
    return proj


def _save(proj: Project, root: Optional[Path] = None) -> None:
    _yaml_path(proj.slug, root).write_text(yaml.safe_dump({
        "slug": proj.slug, "name": proj.name, "description": proj.description,
        "notifyEmail": proj.notify_email, "repoPath": proj.repo_path,
        "providers": proj.providers, "defaults": proj.defaults,
        "createdAt": proj.created_at,
    }, sort_keys=False), encoding="utf-8")


def load(slug: str, root: Optional[Path] = None) -> Project:
    p = _yaml_path(slug, root)
    if not p.exists():
        raise FileNotFoundError(f"no project registered at slug {slug!r}.")
    raw = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    defaults = dict(DEFAULT_PROJECT_DEFAULTS)
    defaults.update(raw.get("defaults") or {})
    return Project(slug=raw.get("slug", slug), name=raw.get("name", slug),
                   description=raw.get("description", ""),
                   notify_email=raw.get("notifyEmail"),
                   repo_path=raw.get("repoPath"),
                   providers=raw.get("providers") or list(DEFAULT_PROVIDERS),
                   defaults=defaults,
                   created_at=raw.get("createdAt", ""))


def set_defaults(slug: str, defaults: dict, *, root: Optional[Path] = None) -> Project:
    """Overwrites this project's shared feature defaults — the rubric,
    price ceilings, judge model, and notify threshold every NEW feature in
    it starts from. Never touches any `use_case.yaml` already written; a
    feature only picks these up at the moment it's created, exactly like
    editing a template doesn't rewrite documents already made from it."""
    proj = load(slug, root)
    merged = dict(DEFAULT_PROJECT_DEFAULTS)
    merged.update(defaults)
    proj.defaults = merged
    _save(proj, root)
    return proj


def list_all(root: Optional[Path] = None) -> list:
    """Every registered project, alphabetical by slug."""
    d = Path(root or DEFAULT_DIR)
    if not d.exists():
        return []
    return [load(p.parent.name, root) for p in sorted(d.glob("*/project.yaml"))]


def read_repo_file(slug: str, relative_path: str, *, root: Optional[Path] = None,
                   max_bytes: int = 200_000) -> str:
    """Reads one file out of a project's connected repo, for reference —
    e.g. while writing a system prompt in the wizard. Read-only, and the
    result can never resolve outside `repo_path`: the repo is a WINDOW, not a
    way to read arbitrary files on this machine."""
    proj = load(slug, root)
    if not proj.repo_path:
        raise ValueError(f"project {slug!r} has no connected repo path.")
    repo_root = Path(proj.repo_path).resolve()
    target = (repo_root / relative_path).resolve()
    try:
        target.relative_to(repo_root)
    except ValueError:
        raise ValueError(f"{relative_path!r} is outside the connected repo.") from None
    if not target.is_file():
        raise FileNotFoundError(f"no file at {relative_path!r} in the connected repo.")
    data = target.read_bytes()
    text = data[:max_bytes].decode("utf-8", errors="replace")
    if len(data) > max_bytes:
        text += f"\n… truncated ({len(data):,} bytes total)"
    return text


def find_use_case(slug: str, name: str, *, root: Optional[Path] = None) -> Optional[Path]:
    """The yaml file for one AI feature by its `useCase:` name — tries the
    wizard's own naming convention first (fast path), falls back to scanning
    every file in case it was renamed or hand-written."""
    direct = use_cases_dir(slug, root) / f"{name}.yaml"
    if direct.exists():
        return direct
    from . import config as config_module
    for p in list_use_cases(slug, root):
        try:
            if config_module.load(p).name == name:
                return p
        except (ValueError, FileNotFoundError):
            continue
    return None
