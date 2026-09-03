# Progress log

What exists in this repo, in the order it was built, and what's still open.
Read this before starting a new session of work so nothing gets re-decided or
re-built by accident. User-facing docs live in `README.md`; this file is for
whoever (human or AI) picks development back up.

---

## Baseline — the core engine (pre-existing)

Before any of the dashboard work below, the project already had a complete,
tested, working CLI tool:

- `modelcicd/catalogue.py` — polls OpenRouter's model list (free)
- `modelcicd/guardrails.py` — filters by price/context/JSON-mode/modality (free)
- `modelcicd/config.py` — loads and validates `use_case.yaml`
- `modelcicd/client.py` — the one place a model is actually called
- `modelcicd/sandbox.py` — runs one candidate against a use case's test cases
- `modelcicd/judge.py` — blind, pinned-judge scoring against the rubric
- `modelcicd/bench.py` — orchestrates sandbox + judge across all candidates
- `modelcicd/rank.py` — per-tier leaderboard, ties within noise
- `modelcicd/state.py` — approved model + history per use case (flat JSON files)
- `modelcicd/resolver.py` — `resolve(use_case)` — the integration point
- `modelcicd/notify.py` — emails a human when a candidate is pending
- `modelcicd/cli.py` — `init` / `catalogue` / `shortlist` / `run` / `status` / `approve`
- `modelcicd/tests/test_modelcicd.py` — 32 dependency-free checks, all passing
- `examples/prep_material/use_case.yaml` — a worked example use case

At the point this log starts, `state/prep_material_writer.json` already had a
real approved model (`google/gemini-2.5-flash`) from a real prior bench run —
proof the whole loop worked end to end before any dashboard code existed.

---

## What we added this session: a local dashboard, MLflow-style

Goal: a `mlflow ui`-style local web view over the same files the CLI already
writes, built in small steps so the core engine is never at risk. Every step
below was verified against real (or, for write actions, disposable
throwaway) data before moving on — never assumed from reading the code alone.

### Step 1 — Read-only dashboard
**Files:** `modelcicd/dashboard.py` (new), `modelcicd/templates/{base,index,usecase,run}.html` (new), `modelcicd/cli.py` (`ui` subcommand), `requirements.txt` (+Flask)

`python -m modelcicd.cli ui` starts a local Flask server that reads
`state/*.json` and `out/*.json` — never writes at this step. Three pages:
use-case list, one use case's history, one run's full leaderboard.
Path-traversal guarded on `/run/<filename>` (`Path(filename).name` only).

### Step 2 — CLI prints the dashboard link
**Files:** `modelcicd/cli.py`

`run` and `status` now print the relevant `http://127.0.0.1:5000/...` URL
after finishing, like MLflow's "View run at ...". Shared `DASHBOARD_HOST` /
`DASHBOARD_PORT` constants added near the top of `cli.py`.

### Step 3 — Visual leaderboard (bar chart)
**Files:** `modelcicd/templates/base.html` (chart CSS + tokens), `modelcicd/templates/run.html` (chart markup)

Loaded the `dataviz` skill before writing any chart code. Horizontal bar
chart per tier: score is a magnitude comparison across models, so one
sequential hue (`#3987e5`), not a categorical rainbow. "Approved" gets the
reserved status-green (`#0ca30c`) plus an icon+label badge, never color
alone. Both colors were run through the skill's validator
(`validate_palette.js`) against this dashboard's actual card surface
(`#171a21`) — both pass every check. A model with a `null` score (a failed
candidate) is correctly excluded from the chart but still shown in the table
below it, which remains as the chart's required table-view twin.

### Step 4 — Approve from the browser
**Files:** `modelcicd/dashboard.py` (`POST /usecase/<name>/approve`), `modelcicd/templates/usecase.html` (button + flash banners), `modelcicd/templates/base.html` (styles)

The only write path added. Calls `state.approve()` directly — the exact
function `cli.py approve` calls — so there is still only one place a model
actually gets promoted. Client-side `confirm()` before the form submits.
Redirects use **303**, not Flask's default 302 (302 is ambiguous about
whether a POST should be re-sent on redirect; found this the hard way via a
curl test that returned 405 — real browsers happened to guess right, but 303
makes it unambiguous for every client). Verified against a disposable fake
use-case state file, never against real state; real state was untouched and
confirmed so afterward.

### Step 5 — Multi-use-case overview + trend
**Files:** `modelcicd/dashboard.py` (`_best_per_run`, `_sparkline`), `modelcicd/templates/index.html` (Trend column)

Home page gained a per-use-case sparkline: the best score found by each past
run, oldest to newest (skips runs where nothing could be scored — no
fabricated zeros). Rendered as inline SVG per the dataviz skill's stat-tile
"trend" spec — de-emphasis gray line, latest point in the validated accent
blue. Shows "not enough runs yet" instead of a misleading single-point line
when a use case has fewer than 2 runs.

### Step 6 — Packaging
**Files:** `pyproject.toml` (new), `.gitignore` (+`*.egg-info/`, `build/`, `dist/`), `README.md`

`pip install -e .` now works and gives a `modelcicd` console command
(`modelcicd status`, `modelcicd ui`, etc.) as an alternative to
`python -m modelcicd.cli`. Verified the console script resolves templates
correctly through the installed package, not just via `-m`.

Also written this session: the top-level `README.md` (project overview,
quickstart, CLI reference, use-case format, dashboard section, design notes).

---

## How verification was done (so it isn't redone unnecessarily)

- Every free/read command (`catalogue`, `shortlist`, `status`, `ui` GET
  routes) was run for real against the live OpenRouter API or real local
  files.
- **No paid command (`run`) has been executed during any of the dashboard
  work** — no API tokens were spent building or testing Steps 1–6.
- Any test that needed to *mutate* state (the Approve button) used a
  disposable fake use-case file (`state/_dashboard_test_*.json`), created,
  exercised over real HTTP, and deleted immediately after — real state
  (`state/prep_material_writer.json`) was never touched by testing.
- `python -m modelcicd.tests.test_modelcicd` (32 checks) was re-run after
  every step and stayed green throughout.

---

## Open decisions / not yet done

- **No license chosen.** `pyproject.toml` deliberately omits `license =`
  and there's no `LICENSE` file — this is a real legal decision for the repo
  owner, not something to default silently. Pick one (MIT is the common
  default for a project like this) before treating `ModelVigil` as properly
  open-source.
- **Approve is only wired up from the use-case page**, using whatever is
  currently `pending`. There's no way yet to approve an arbitrary model
  straight from a run's leaderboard page — would need a small form per row
  if that's wanted.
- **Dashboard has no auth.** Fine bound to `127.0.0.1` (the default); if it's
  ever run with `--host 0.0.0.0`, anyone on the network could hit the
  Approve endpoint. Not addressed because the tool is meant to stay
  self-hosted and single-user — flag this again if that assumption changes.
- **Notification is still email-only** (`notify.py`) — Slack/webhook
  channels were discussed as a possible future step but not built.
- Repo remote: `https://github.com/jashwanthsai678/ModelVigil.git`, author
  identity configured locally as `jashwanthsai678` / `jashwanthsai678@gmail.com`.
