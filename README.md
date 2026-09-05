# Model CICD

**The AI model behind every feature in your app goes stale the moment
someone stops watching it. This project watches it for you — and never
swaps one in without your sign-off.**

## What it is

Model CICD is a self-hosted tool that continuously discovers, benchmarks,
and lets you approve which LLM your application actually uses — separately,
for each specific feature that calls one. It doesn't look for "the best
model" in general; it finds the best model **for one job**, tested against
that job's own test cases and scored against a rubric written for that job,
and it never changes what your app uses in production until a human
explicitly says yes.

```python
from modelcicd.resolver import resolve

model = resolve("support_bot_reply", fallback="gpt-4o-mini")
# call your existing LLM client with `model`, exactly as before
```

That's the entire integration surface. Everything else in this repo exists
to answer one question well: which model id should that line return?

## Who needs it

Any developer or team whose application calls an LLM in one or more places,
and who has ever had to answer (or been unable to answer) questions like:
"why are we using this particular model here," "are we overpaying for it,"
"would a newer or cheaper model do just as well," or "who approved this and
when." If your app has exactly one throwaway LLM call that nobody would
notice if it got slightly worse or slightly more expensive, you probably
don't need this yet. The moment even one LLM call matters enough that its
cost or quality is worth tracking, this is for you.

## Why they need it

Because a model choice, left alone, decays:

- **New, cheaper, or better models ship every few weeks.** Without a
  standing process to re-check, you either keep overpaying for a model a
  newer one now beats, or never find out a cheaper option would do the same
  job.
- **A quality regression from a model swap is invisible until a user hits
  it.** Nobody manually re-tests forty candidates by hand every time a new
  model drops — so most teams simply don't, and find out something got worse
  from a support ticket instead of a benchmark.
- **"Which model, exactly?" is usually answered from memory**, not
  evidence — no record of what was tried, what it scored, or why it was
  chosen, so the decision can't be revisited or defended later.
- **A silent model swap in production is a real risk**, on par with an
  unreviewed dependency upgrade — it should never happen without a human
  deciding it should.

## What value it adds

- **Lower cost without guessing.** Every candidate is filtered by your
  price ceiling before anything is spent, and ranked within its own price
  tier, so "cheaper" is a fact you can see, not a hope.
- **Quality protected, not just cost.** Nothing gets proposed as a
  replacement without first being scored, blind, against the rubric that
  feature's own owner wrote — a model can't win purely by being cheap.
- **A record that survives memory.** Every run, every candidate, every
  score, and every approval is saved to disk with a timestamp — "why this
  model" is always answerable later, by anyone, not just whoever remembers.
- **A single, low-friction decision point.** A human sees the evidence and
  clicks Approve (CLI or dashboard) — nothing changes silently, and nothing
  requires re-deriving a process from scratch each time.
- **This, repeated, per feature.** The same loop runs independently for
  every place your app calls an LLM, so a five-feature application ends up
  with five independently tracked, independently approved model choices,
  not one blunt global setting.

---

## The design constraints behind it

- **New models ship constantly, and prices move.** Re-checking the field by
  hand does not scale past one or two use cases.
- **A generic leaderboard lies to you.** A model that tops a public
  benchmark can still write unusable output for a task nobody else tested it
  on. "Correctness" means something different for a support bot than for a
  codegen assistant.
- **An LLM grading its own family's writing is not a benchmark.** Judging
  needs to be blind and done by a model that isn't also a candidate.
- **A model swap in production is a real decision.** It should require the
  same kind of sign-off a dependency upgrade would get — not happen silently
  because a scheduled job found a slightly higher number.

Model CICD is built around those four constraints. It's a flat-file,
self-hosted, single-tenant tool — no server, no database, no signup — meant
to be dropped into an existing repository.

---

## How it works

In plain terms, the loop is:

1. **Keep checking for new or updated models** — what exists, what it costs,
   right now, not whatever was true when you last looked.
2. **See how each one actually performs for this use case** — not a generic
   benchmark, your own test cases, run for real against each candidate.
3. **Note all of it down** — every candidate, every test case, every answer,
   every score, saved to disk, never silently discarded.
4. **Measure with your own metric** — the rubric *you* wrote for this
   specific feature, scored by a separate judge model, blind.
5. **Surface analytics per response** — a score, a would-ship verdict, and
   *why*, per criterion, per candidate, per test case, plus price and a
   trend of how the best available score has moved over time.
6. **Connect to your application** — one `resolve("use_case_name")` call
   returns the model your app should use for that feature.
7. **The model name changes — but only when you say so.** Steps 1–5 run
   automatically, on whatever schedule you choose. Step 6 does **not**
   auto-switch to whatever scored highest: a better candidate is flagged as
   *pending*, and `resolve()` keeps returning the old model until a human
   explicitly approves the new one, from the CLI or the dashboard.

That last point is the one deliberate exception to "automatic": discovery,
testing, measuring, and analytics all happen with zero human involvement —
only the moment your application's actual behavior changes requires someone
to say yes.

```
 catalogue.py          guardrails.py         sandbox.py        judge.py        rank.py         state.py
┌───────────────┐    ┌────────────────┐    ┌────────────┐    ┌───────────┐   ┌───────────┐   ┌──────────────┐
│ poll every     │ →  │ filter by      │ →  │ call every  │ →  │ blind,    │ → │ leaderboard│ → │ approve()    │
│ model + price  │    │ price/context/ │    │ candidate   │    │ pinned    │   │ per price  │   │ is the ONLY  │
│ (free, no LLM  │    │ JSON-mode/     │    │ with your   │    │ judge     │   │ tier, ties │   │ thing that   │
│ call)          │    │ modality       │    │ test cases  │    │ scores    │   │ within     │   │ changes what │
│                │    │ (free)         │    │ ($)         │    │ each      │   │ noise      │   │ resolve()    │
│                │    │                │    │             │    │ answer($) │   │            │   │ returns      │
└───────────────┘    └────────────────┘    └────────────┘    └───────────┘   └───────────┘   └──────────────┘
```

1. **Catalogue** — polls OpenRouter's model list (id, price, context length,
   JSON-mode support). Pure `GET` request; no model is ever called here.
2. **Guardrails** — filters that catalogue against one use case's price
   ceiling, minimum context, JSON-mode requirement, and modality. Every
   rejection is named with the exact rule and value that failed it. Still
   free.
3. **Sandbox** — runs every surviving candidate against that use case's own
   test cases, one call per test case. A model that fails is recorded with
   its error, not silently dropped from the leaderboard.
4. **Judge** — a separate, pinned model scores every answer against the use
   case's own rubric, blind (one answer at a time, never told which model
   wrote it). It refuses to run if the judge is also a candidate.
5. **Rank** — builds a leaderboard **within each price tier** (free /
   paid-low / paid-mid / paid-high), because a global ranking always
   concludes the expensive model won. Scores within a small margin are
   shown as a tie, not a winner — one sample per test case is confident at
   the extremes, not enough to crown a narrow "best."
6. **State + notify** — a candidate that clearly beats the currently
   approved model is recorded as **pending** and, if configured, emailed to
   a human. Nothing in production changes yet.
7. **Approve** — the one action that promotes a model, from the CLI or the
   dashboard. Only after this does `resolve()` return the new id.

---

## Quickstart

```bash
# 1. Install
python -m venv .venv
.venv\Scripts\activate            # Windows
source .venv/bin/activate         # macOS/Linux
pip install -r requirements.txt
# or: pip install -e .            # also gives you the `modelcicd` command,
#                                  # so every command below can drop `python -m`

# 2. Configure
cp .env.example .env
# edit .env and set OPENROUTER_API_KEY

# 3. Write a use case (or copy the example under examples/prep_material/)
python -m modelcicd.cli init --out use_cases/my_feature.yaml

# 4. See what candidates survive your guardrails — free, no model calls
python -m modelcicd.cli shortlist --use-case use_cases/my_feature.yaml

# 5. Run the bench — this is the only command that spends money,
#    and it shows the cost and asks before it does
python -m modelcicd.cli run --use-case use_cases/my_feature.yaml

# 6. Check what's approved vs. pending
python -m modelcicd.cli status --use-case use_cases/my_feature.yaml

# 7. Promote a candidate once you're satisfied — from the CLI...
python -m modelcicd.cli approve --use-case use_cases/my_feature.yaml
# ...or from the dashboard:
python -m modelcicd.cli ui
```

### Or: connect an application and use the guided flow instead

You don't have to hand-write `use_case.yaml`. Connect an application (a
**project** — just a name, nothing is read from it) and define each of its AI
features through a wizard — identical from the CLI and the dashboard:

```bash
python -m modelcicd.cli project create --name "My App"
python -m modelcicd.cli wizard --project my-app        # asks for the system
                                                          # prompt, test cases,
                                                          # rubric, and price/
                                                          # model preferences
python -m modelcicd.cli onboarding                      # the full guided flow, any time
```

See [Projects, the wizard, and a live endpoint](#projects-the-wizard-and-a-live-endpoint) below.

---

## CLI reference

| Command | Cost | What it does |
|---|---|---|
| `init` | free | Writes a starting `use_case.yaml` template |
| `onboarding` | free | Prints the guided quickstart |
| `set-key --provider {openrouter,groq,fireworks} [--key <value>]` | free | Saves a platform's API key to `.env` (prompts, hidden, if `--key` omitted) |
| `keys` | free | Which platforms have a key configured |
| `project create --name <n> [--description] [--notify-email] [--repo-path <dir> \| --repo-url <url>] [--providers <list>]` | free | Connects a new application |
| `project list` | free | Lists connected applications |
| `scan-repo --project <slug> [--scan-model <id>] [--max-files <n>] [--yes]` | **$** | Reads the connected repo with a model to find likely LLM call sites |
| `wizard [--project <slug>] [--from-scan <index>]` | free | Interactively defines an AI feature — no YAML to hand-write |
| `apply-code-patch --use-case-name <n> --project <slug> [--yes]` | free | Writes an approved model into its `codeTarget` file — previewed, confirmed |
| `catalogue` | free | Polls OpenRouter and saves `out/catalogue.json` |
| `shortlist --use-case <path>` | free | Applies one use case's guardrails to the catalogue |
| `run --use-case <path> [--project <slug>]` | **$** | Sandboxes + judges + ranks the shortlist (+ the live endpoint, if configured) |
| `status [--use-case <path> \| --use-case-name <name>] [--project <slug>]` | free | Shows the approved model and any pending candidate |
| `approve [--use-case <path> \| --use-case-name <name>] [--model <id>] [--project <slug>]` | free | Promotes a model — the only thing that changes `resolve()` |
| `reject [--use-case <path> \| --use-case-name <name>] [--project <slug>]` | free | Dismisses the pending candidate — never changes `resolve()` |
| `pending` | free | Everything waiting for review, across every project |
| `scheduler run-due` | **$** | Runs every AI feature that's due right now, once, and exits |
| `scheduler serve [--interval-minutes <n>]` | **$** | Loops forever, re-checking every AI feature's schedule |
| `ui [--host <addr>] [--port <n>]` | free | Launches the local dashboard at `http://127.0.0.1:5000` |

Useful `run` flags: `--tier {free,paid-low,paid-mid,paid-high}` to restrict
which price band is benched, `--models a,b,c` to bench an explicit list
instead of the guardrail-filtered shortlist, `--limit N` to cap candidate
count, `--yes` to skip the interactive cost confirmation (for scheduled
jobs — a `scheduler` run always behaves as if `--yes` was passed, since
nobody is there to answer). If installed with `pip install -e .`, every
command above also works as `modelcicd <command>` instead of
`python -m modelcicd.cli <command>`.

---

## Dashboard

`python -m modelcicd.cli ui` starts a local, read-only-by-default web view
over the same files the CLI already writes — nothing it displays is computed
anywhere else, and nothing about it is required to use the CLI. Think
`mlflow ui`, pointed at `state/` and `out/` instead of `mlruns/`.

- **Home page** — every use case at a glance: approved model, pending
  candidate, run count, and a trend sparkline of the best score each past run
  found (so a slow drift or a newly competitive cheap model is visible
  without opening a single JSON file).
- **Use case page** — the full approval history and a link to every past
  run's leaderboard. If a candidate is pending, an **Approve** button sits
  right next to it — behind a confirmation dialog, and wired to the exact
  same `state.approve()` function the CLI's `approve` command calls, so
  there's still only one thing in this project that can change what
  `resolve()` returns.
- **Run page** — the leaderboard as a per-tier bar chart (score, price,
  would-ship rate on hover) with the full data table underneath it, so
  nothing is chart-only.

`run` and `status` print the relevant dashboard link after they finish, so
you don't have to know the URL scheme by heart.

---

## API keys — asked for in the interface, and actually used where they're set

Each marketplace needs its own key. Instead of only hand-editing `.env`, the
interface asks for it — the connect form has a masked field under each
platform's checkbox, and `/keys` (also reachable from the CLI) lets you set
or update one any time, not just when connecting a project:

```bash
python -m modelcicd.cli set-key --provider groq      # prompts, input hidden
python -m modelcicd.cli keys                          # which platforms are configured
```

Either way it's written to the same `.env` a key always lived in — no new
secret storage, and a key is never displayed back once saved, only whether
one is set.

**Setting a key is what makes selecting that platform real, not cosmetic.**
Every candidate is now called through the platform it was actually
discovered on — a Groq-sourced candidate hits Groq's own endpoint with
`GROQ_API_KEY`, a Fireworks one hits Fireworks' with `FIREWORKS_API_KEY`,
OpenRouter candidates work exactly as before. Before this, checking "Groq"
only changed what showed up while *browsing* the catalogue; the actual bench
call still always went to OpenRouter regardless. The judge is the one
exception — it always calls through OpenRouter, regardless of which
platform a candidate came from, since a use case's `judgeModel` is normally
pinned as an OpenRouter-style id either way.

---

## Projects, the wizard, and a live endpoint

A **project** is a connected application — a name, a description, a
**required** notify email (every AI feature in it defaults to this; a
project with nowhere for a pending-candidate notification to go just means
that step silently degrades to a console message nobody's watching, so it's
asked for up front, in the CLI and the dashboard both), a marketplace list
(`--providers openrouter,groq,fireworks` — default: OpenRouter alone), and
optionally the application's own code: a local `--repo-path` (must already
exist) or a `--repo-url` modelcicd clones itself (shallow, read-only —
relies entirely on whatever the local git/OS credential helper already
provides; no credentials handled here). Without either, nothing is read
from or written to the connected application at all — a project exists so
several applications' AI features stay grouped, each benched and approved on
its own. Its AI features live under `projects/<slug>/use_cases/`, with their
own `state/` and `out/`, exactly like the top-level `state/`/`out/`
directories already used for an unscoped use case.

```bash
python -m modelcicd.cli project create --name "Support Portal" \
  --notify-email you@example.com \
  --repo-path D:\repos\support-portal \
  --providers openrouter,groq          # optional — enables the features below
python -m modelcicd.cli wizard --project support-portal
```

The dashboard's connect form has the same fields plus one thing the CLI
doesn't: a **"Browse…" button** next to the repo path that opens your
computer's own real folder picker instead of making you type a path — this
works only because the dashboard's server and the browser viewing it are the
same machine, so the server can pop the dialog and hand back the real
location. It quietly falls back to a plain typed path if that's not
available. The repo path and git URL fields are behind a dropdown — pick
which one applies (or neither, if you're not connecting code yet), only the
matching field shows.

The wizard — identical from the CLI's interactive prompts and the dashboard's
"Add an AI feature" form, both going through the same `modelcicd/wizard.py` —
asks for everything a `use_case.yaml` needs: the system prompt, test cases,
a rubric, price/context guardrails, the judge model, and who to notify. It
writes the same YAML schema described below; nothing about how a use case is
*read* changes because it was written this way.

### Scanning the repo for AI features already in the code

With a repo connected, `scan-repo` reads candidate files with a model to find
likely LLM call sites — the literal model string and prompt, where present —
across *any* language or framework, because it's reading for meaning, not
matching a fixed list of SDK patterns:

```bash
python -m modelcicd.cli scan-repo --project support-portal
# candidate files : 14
# cost            : 14 model call(s)
# proceed? [y/N]
#
# [0] app/bot.py  model=gpt-4o-mini  confidence=high
#      You are a support agent. Read the ticket and draft a reply...

python -m modelcicd.cli wizard --project support-portal --from-scan 0
```

**This spends a small amount of money** — one model call per candidate file
— shown and confirmed first, exactly like `run`. A scan only ever *proposes*:
picking a candidate pre-fills the wizard's system prompt and code target,
both still shown and editable before anything is saved, and test cases are
always still typed in by hand — a scan can identify a prompt, not invent a
good test case. The same flow exists in the dashboard: a "Scan repo for AI
features" button on the project page, a results list, "Use this" into the
same wizard form.

**Read the confidence, don't just read the result.** Every candidate carries
a `confidence: high|low` — a low-confidence one is flagged distinctly
("verify before using") rather than presented the same as a clean hit, and a
scan that finds nothing or only low-confidence candidates points straight at
`wizard --project <slug>` (no `--from-scan`) as the reliable fallback. This
matters most exactly where it's weakest: code that builds a prompt
dynamically or through an agent/tool-calling framework is real code the
scanner is likely to miss or misread — it reads for a literal call site, not
that kind of indirection. Don't read a quiet scan as "there's nothing here."

### A live endpoint, for a real baseline

Every AI feature can optionally name its own live endpoint — hosted or
`http://localhost:...`, both are just a URL:

```yaml
endpoint:
  url: http://localhost:8000/reply
  method: POST
  inputField: input           # request body = {"input": "<test case text>"}
  responseField: output       # dot-path into the JSON response, e.g. data.reply
  headers:
    Authorization: "Bearer ${SUPPORT_API_KEY}"   # ${VAR} expands from the
                                                  # environment, never committed
  timeoutSeconds: 30
```

When set, every `run` also calls this endpoint once per test case — the same
input every candidate model gets — and judges the answer through the same
rubric, blind, exactly like a candidate. The leaderboard then shows a
**"current (your endpoint)"** row alongside every candidate, so approving a
model is a real "this beats what you have" decision, not just "this beats
other candidates." The endpoint is only ever called to capture that one
comparison row — no candidate is ever routed through it.

### A code target, for an app that isn't integrated with `resolve()` yet

If a project has `--repo-path`, the wizard can read files from it for
reference (a "Browse a file" link — read-only, never written to except via
the patch below), and an AI feature can optionally name where its model is
hardcoded:

```yaml
codeTarget:
  file: app/bot.py            # relative to the project's repo_path
  currentModel: gpt-4o-mini   # what's hardcoded there right now
```

After you approve a candidate, `apply-code-patch` — from the CLI, or a
"Preview code change" button on the use case page — shows exactly what would
change (file, occurrence count, old model → new model) and asks for its own,
separate confirmation before writing anything:

```bash
python -m modelcicd.cli apply-code-patch --use-case-name support_bot_reply --project support-portal
```

This is deliberately never automatic. `state.approve()` — the only function
that changes what `resolve()` returns — stays a plain state-file write with
no filesystem side effects. Patching real source is a second, separate,
always-previewed action: an exact literal string replace (never a regex,
never AST-aware), refusing rather than guessing if the recorded model string
isn't found verbatim anymore. modelcicd performs no git operations on your
repo — committing or reviewing the change is your own normal git workflow.
If your application already calls `resolve("feature_name")` instead of
hardcoding a model, you don't need a code target at all: approving already
changes what it uses, with nothing to patch.

---

## Scheduler

Add a `schedule` block to a use case (by hand, or the wizard's last step) and
it becomes eligible to run on its own:

```yaml
schedule:
  intervalDays: 7
```

```bash
python -m modelcicd.cli scheduler serve            # leave this running — checks on an interval
python -m modelcicd.cli scheduler run-due           # or trigger a single check from cron / Task Scheduler
```

Either way, being due only ever runs the same `catalogue → guardrails →
sandbox → judge → rank → notify` loop `run` does — landing, at most, a new
**pending** candidate. Nothing here calls `approve()`. A model still only
reaches production when a human clicks Approve, however often the platform
re-checks.

**A scheduled run only emails you when the pending candidate actually
changes.** The first time a candidate is found, you're notified once; if
next week's scheduled run finds the exact same still-unreviewed candidate,
it does **not** send the same email again — repeating an unread suggestion
every week is how notifications get ignored. The candidate itself is
unaffected either way — it stays visible in the dashboard, `status`, and the
aggregate view below until you act on it.

```bash
python -m modelcicd.cli pending                        # everything waiting for review, everywhere
python -m modelcicd.cli reject --use-case-name <name>   # dismiss one — never changes resolve()
```

`pending` lists every candidate awaiting a decision across every project, so
a scheduler left running for months doesn't quietly bury suggestions inside
individual feature pages. `reject` is the other side of `approve` — it
clears the suggestion **without promoting anything**, and is recorded (not
silently discarded) so a dismissal is still answerable later. If that exact
model resurfaces as the best candidate in some future run, you're notified
about it again — a dismissal isn't permanent silence, it's "not this time."
The dashboard has the same two actions as buttons on the use case page:
Approve and Dismiss, each with its own confirmation.

---

## Rate-limit visibility — observed, never guessed

Free-tier models (OpenRouter's especially) hit shared rate limits fast in
real use. There's no public catalogue of per-model rate limits to check
that against, so this isn't another guardrail filter — it's evidence,
gathered the same way the score is: by actually calling the candidate.

```yaml
usage:
  callsPerDay: 2000   # optional, per AI feature — the wizard asks for this
```

If a candidate returns 429 (Too Many Requests) during the bench — even a
light one, a handful of test cases — that's recorded as its own outcome,
distinct from a generic failure, and shown on the leaderboard next to your
usage estimate if you gave one:

```
-  vendor/free-model   could not be scored — rate limited during this bench
                        (3 call(s)) — a light test already hit it; your
                        ~2,000/day estimate is unlikely to be sustainable here
```

The candidate still appears on the leaderboard either way — this never
disqualifies anything automatically. It's a warning attached to real
evidence, for the human who's about to approve something.

---

## Judge noise, not just candidate noise

Every score here comes from ONE judged sample — cheap by design, but that
means a "tie" between two close candidates could be candidate noise (they're
genuinely similar) or **judge noise** (the judge itself wasn't consistent
about it), and a single sample can't tell those apart. Left alone, a tie
band that only protects against the first kind can make a shaky ranking look
confident for the wrong reason.

So a `run` deepens on exactly the candidates that made a tie-band shortlist:
their already-generated answers (no candidate is re-run) get re-scored by
the judge two more times, and the spread across those looks is shown right
next to the score:

```
*  1  vendor/model-b   4.60  (judge spread ±0.70)
   2  vendor/model-a   4.25
```

A small spread means the judge agreed with itself; a wide one means this
"tie" is judge noise, not a real result — worth reading before approving a
close call. A candidate outside the tie zone was never in question, so it's
never re-judged — same principle as everywhere else here: deepen on the
shortlist, never the whole field.

---

## One use case per feature

Every place your app calls an LLM gets its own `use_case.yaml` — its own
system prompt, test cases, rubric, price ceiling, and notification target.
Nothing else needs a database; the file **is** the onboarding.

```yaml
useCase: support_bot_reply
description: Drafts a first-line reply to an incoming support ticket.

systemPrompt: |
  You are a support agent. Read the ticket and draft a reply. Return
  exactly: {"reply": "the message to send"}

testCases:
  - id: angry_refund_request
    input: "Customer: This is the third time my order arrived broken.
            I want a refund NOW."
    reference: null

rubric:
  - id: tone
    description: Calm, empathetic, never defensive.
    weight: 2
  - id: resolves
    description: Actually proposes a concrete next step, not just an apology.
    weight: 1.5
  - id: format
    description: Returns exactly the one required JSON field.
    weight: 1

guardrails:
  maxPriceIn: 0.50
  maxPriceOut: 3.00
  minContext: 32000
  requireJson: true
  allowFree: true
  tiers: [free, paid-low, paid-mid]

judgeModel: openai/gpt-4o
maxTokens: 700

notify:
  email: null          # or set MODELCICD_NOTIFY_EMAIL in the environment
  minImprovement: 0.20
```

Each use case's state (`state/<name>.json`) and every bench run's full
results (`out/<timestamp>_<name>.json`) are tracked independently — running
or approving one use case never touches another. So a real application ends
up with several use cases, each on its own bench-and-approve cadence:

```
use_cases/
  support_bot_reply.yaml
  ticket_summarizer.yaml
  codegen_assistant.yaml
```

Grouped under a project instead, the same independence holds one level up —
one connected application, several use cases, each still benched and
approved on its own:

```
projects/support-portal/
  project.yaml
  use_cases/
    support_bot_reply.yaml
    ticket_summarizer.yaml
  state/
    support_bot_reply.json
    ticket_summarizer.json
  out/
    20260310T090000Z_support_bot_reply.json
```

```python
from modelcicd.resolver import resolve

reply_model   = resolve("support_bot_reply",   fallback="gpt-4o-mini")
summary_model = resolve("ticket_summarizer",   fallback="gpt-4o-mini")
codegen_model = resolve("codegen_assistant",   fallback="claude-sonnet")
```

---

## Configuration

Set these in `.env` (see `.env.example`) — or use `set-key` / `/keys`, which
write to this same file:

| Variable | Required | Purpose |
|---|---|---|
| `OPENROUTER_API_KEY` | yes, for `run` | Every OpenRouter-sourced candidate and the judge are called through OpenRouter |
| `GROQ_API_KEY` | only if a project searches Groq | Groq-sourced candidates are called through Groq directly |
| `FIREWORKS_API_KEY` | only if a project searches Fireworks | Fireworks-sourced candidates are called through Fireworks directly |
| `MODELCICD_NOTIFY_EMAIL` | no | Default recipient for pending-candidate emails, if not set per use case |
| `SMTP_HOST` / `SMTP_PORT` / `SMTP_USER` / `SMTP_PASSWORD` / `SMTP_FROM` / `SMTP_STARTTLS` | no | If unset, the CLI prints the approve command instead of emailing it |

API keys never belong in a `use_case.yaml` — that file is meant to be
committed to your repo, and a config with a key in it is a leaked key.

---

## Project layout

```
modelcicd/
  catalogue.py       fetch + normalize the model catalogue (free); provider_map traces a host to its platform
  guardrails.py      filter candidates by price/context/JSON/modality (free)
  config.py          load and validate a use_case.yaml (incl. endpoint/schedule/codeTarget blocks)
  client.py          the one place this project calls a CANDIDATE model — standalone, endpoint/key are parameters
  secrets.py         the one place this project writes an API key (into .env, nowhere new)
  endpoint_client.py the one place this project calls a use case's OWN live endpoint
  sandbox.py         run one candidate (through ITS OWN platform) or the live endpoint, against the test cases
  judge.py           blind, pinned-judge scoring against the use case's rubric
  bench.py           orchestrates sandbox + judge across all candidates + the endpoint
  rank.py            builds the per-tier leaderboard
  state.py           what's approved per use case, its history, its pending queue
  resolver.py        resolve(use_case) -> the approved model id (the integration point)
  notify.py          emails a human when a candidate is pending
  project.py         connects an application — a name, an optional repo, its providers, its AI features
  code_scan.py       reads a connected repo with a model to find likely LLM call sites
  wizard.py          turns answers (CLI or dashboard form) into a valid use_case.yaml
  code_patch.py      the one place this project writes into a connected app's OWN source
  runner.py          the reusable core of one bench run, shared by `cli run` and the scheduler
  scheduler.py       re-runs whatever is due, on its own schedule
  cli.py             the `python -m modelcicd.cli ...` / `modelcicd ...` entrypoint
  dashboard.py       the local web view (`cli ui`) — same functionality as the CLI
  templates/         the dashboard's HTML (Jinja2)
  tests/             a dependency-free test module: python -m modelcicd.tests.test_modelcicd
examples/
  prep_material/use_case.yaml   a worked example use case
pyproject.toml    lets you `pip install -e .` for the `modelcicd` command
```

## Testing

```bash
python -m modelcicd.tests.test_modelcicd
```

No `pytest` dependency — a small, self-contained runner that checks the
specific edge cases the design relies on (negative-price sentinels, the
judge never being a candidate, tier ranking, stale-state bugs, and so on).

---

## Design notes worth knowing before you extend this

- **Unknown is not free.** A model with no published price is excluded by
  the cost guardrail, never assumed to cost $0.
- **A logical model keeps its hosting options rather than merging them.**
  The same weights on two platforms can have different quantization,
  context windows, or JSON-mode support — collapsing them into one row
  throws away the thing worth measuring.
- **Ranking is always per price tier**, never global — otherwise the
  expensive model always "wins."
- **One sample per test case** is deliberate: a screening pass over forty
  candidates costs 40 calls this way and 120+ if repeated for statistical
  confidence. Deepen sampling on the shortlist a candidate makes it to,
  never on the whole field.
- **`approve()` is the only function that changes what `resolve()`
  returns.** Every other step in the pipeline only ever proposes — including
  a scheduled run finding a candidate on its own.
- **A live endpoint captures a baseline; it never runs a candidate.** The
  leaderboard's "current (your endpoint)" row exists so a human compares a
  candidate against what they actually have, not just against other
  candidates — but nothing routes a candidate model through someone else's
  application.
- **The CLI and the dashboard write through the same functions, always.**
  `wizard.to_yaml`, `project.create`, and `state.approve` are each called
  from exactly one place in both `cli.py` and `dashboard.py` — never
  reimplemented for the browser — so the two can never quietly drift apart.
- **Approving a model and patching your source are two separate decisions.**
  `state.approve()` never touches a filesystem outside `state/`; a
  `codeTarget` patch always gets its own preview and its own confirmation
  (`apply-code-patch`, or its own button on the use case page) — writing into
  someone else's real application is a bigger, harder-to-reverse action than
  writing a JSON file, and it is never bundled into a click meant to do the
  smaller one.
- **A scan proposes; it never writes.** `scan-repo` reads a connected repo
  with a model to find likely LLM call sites, but the result is only ever a
  pre-fill — a person still confirms (and can edit) the prompt and test
  cases through the same wizard every other AI feature goes through. Like
  `run`, it shows its real cost (one model call per candidate file) before
  spending anything.
- **A provider's price/JSON-mode support is only ever what its own API
  publishes.** Groq's and Fireworks' `/models` endpoints don't guarantee
  either — those fields come back unknown rather than fabricated, and the
  existing "unknown is not free" guardrail excludes such models the same
  way it excludes any other unpriced one. No provider gets a hardcoded price
  table that could go stale or wrong.
- **Rate-limit risk is observed, never fabricated.** There's no public
  catalogue of per-model rate limits the way there is for price, so this
  isn't a guardrail — a 429 during the bench is real evidence, recorded as
  its own outcome and shown next to the use case's own usage estimate. It
  never disqualifies a candidate automatically; it's context for the human
  who's about to approve one.
- **`reject()` is not `approve()`.** It's the only other function allowed to
  clear `pending`, and it's structurally incapable of touching
  `approvedModel` — a dismissal is recorded (`state["rejected"]`), never
  silently discarded, and never promotes anything.
- **A scheduled run notifies once per candidate, not once per check.**
  `state.notifiedModel` tracks what you were last emailed about; the same
  still-pending candidate doesn't re-trigger the email on every scheduled
  run, only a genuinely different candidate does.
- **Judge noise gets the same "deepen only the shortlist" treatment as
  everything else.** Re-scoring is never applied to the whole field — only
  to candidates that already made a tie-band shortlist, and only against
  answers already generated, so it costs extra judge calls and zero extra
  candidate calls.
- **`client.py` stays standalone; it never learns what a "provider" is.**
  It's meant to be dropped into another repo on its own, so it can't import
  `catalogue.py`. Multi-platform routing is `sandbox.py`'s job — it resolves
  a candidate's endpoint and key env var and passes them in as plain
  parameters, which default to exactly OpenRouter, exactly as `call_json`
  always behaved before providers existed.
- **A key is never displayed back once saved.** `secrets.status()` reports
  whether one is set, never its value — the connect form and `/keys` show a
  configured/not-set badge, not the key itself.

## Contributing

Bug reports, ideas, and pull requests are welcome. See
[CONTRIBUTING.md](CONTRIBUTING.md) for how to get set up, run the test
suite, and the conventions this codebase follows.

## License

Apache 2.0 — see [LICENSE](LICENSE).
