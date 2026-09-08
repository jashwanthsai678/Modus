<h1 align="center">Model CICD</h1>

<p align="center">
  <strong>Continuous model discovery, sandboxed benchmarking, and human-approved
  promotion — for every place your application calls an LLM.</strong>
</p>

<p align="center">
  <a href="LICENSE"><img alt="License: Apache 2.0" src="https://img.shields.io/badge/license-Apache%202.0-blue.svg"></a>
  <a href=".github/workflows/tests.yml"><img alt="Tests" src="https://github.com/jashwanthsai678/ModelVigil/actions/workflows/tests.yml/badge.svg"></a>
  <img alt="Python 3.9+" src="https://img.shields.io/badge/python-3.9%2B-blue.svg">
  <img alt="Self-hosted" src="https://img.shields.io/badge/hosting-self--hosted-informational">
</p>

<p align="center">
  <a href="#quickstart">Quickstart</a> ·
  <a href="#how-it-works">How it works</a> ·
  <a href="#cli-reference">CLI</a> ·
  <a href="docs/GUIDE.md">Feature guide</a> ·
  <a href="docs/DESIGN.md">Design notes</a>
</p>

<p align="center">
  <img alt="Project overview: every AI feature, the model it uses, and the proof behind it"
       src="docs/screenshots/overview-dark.png" width="900">
</p>

---

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

New models ship every few weeks. Nobody re-benchmarks forty candidates by hand,
so teams either overpay for a model a cheaper one now matches, or quietly ship a
regression nobody measured. Model CICD runs that comparison for you, on a
schedule, and puts one decision in front of a human: **approve, or don't.**

| | |
|---|---|
| **Per feature, not per model** | Each LLM call site gets its own test cases, its own rubric, its own approved model. No global leaderboard pretending one model wins everything. |
| **Ranked within price tiers** | A cheap model only competes against other cheap models, so "good enough for 10× less" is a visible, defensible answer. |
| **Proof, not just scores** | Every score comes with the candidate's real answer and the judge's stated reasoning, saved with a timestamp. |
| **Nothing changes silently** | `resolve()` keeps returning the old model until a human approves. Patching real source code is a second, separate confirmation. |
| **Self-hosted, flat files** | No database, no signup, no data leaving your machine. State is JSON and YAML you can read and commit. |

<p align="center">
  <img alt="Leaderboard with each candidate's real answer, score, and the judge's reasoning"
       src="docs/screenshots/leaderboard-proof.png" width="900">
</p>

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
copy .env.example .env            # Windows
cp .env.example .env              # macOS/Linux
# edit .env and set OPENROUTER_API_KEY — or set it later from the dashboard's
# "API keys" page, which writes to the same file

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

See the [feature guide](docs/GUIDE.md) for connecting a real repo, scanning it,
comparing against a live endpoint, and patching an approved model back into source.

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

**This is the actual differentiator: it's an agentic pipeline, not a
one-off script you have to remember to re-run.** Once a use case exists,
nobody manually re-benchmarks anything ever again — the discover → test →
score → rank → notify loop runs by itself, on a schedule, indefinitely, and
the only work left for a human is reading a report and clicking one of two
buttons. That's a fundamentally different amount of effort than "someone
remembers to re-check the model market every few months," which is what
every team does today in practice. And it's not one checkpoint doing double
duty — there are deliberately **two** separate human gates, not one:
approving a candidate (changes what `resolve()` returns) and, only if your
app isn't yet calling `resolve()`, a second and entirely separate
confirmation to patch the literal model string in your real source file.
Neither ever happens without a person explicitly clicking it — the
automation covers everything *except* the decision that actually matters.

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

## Dashboard

`python -m modelcicd.cli ui` starts a local, read-only-by-default web view
over the same files the CLI already writes — nothing it displays is computed
anywhere else, and nothing about it is required to use the CLI. Think
`mlflow ui`, pointed at `state/` and `out/` instead of `mlruns/`.

- **Home page** — every use case at a glance: approved model, pending
  candidate, run count, and a trend sparkline of the best score each past run
  found (so a slow drift or a newly competitive cheap model is visible
  without opening a single JSON file).
- **Project page** — a step-by-step guide (Connect → Scan → Define → Run →
  Approve) showing exactly which of those you've done and which one's next,
  instead of a flat row of equally-weighted buttons. Any feature that's been
  defined but never benched is called out by name with a direct "Run" link,
  so a saved feature never quietly disappears.
- **Run page** — pick a price tier and an optional candidate limit, see the
  exact candidate list and call-count cost *before* anything is spent, then
  run the bench right there — the same `runner.execute` the CLI's `run`
  command calls, so this is a second door into one implementation, not a
  second copy of it. Long-running actions (scanning, drafting test cases,
  benching) show a working state instead of looking frozen — these can
  genuinely take several minutes on free-tier models.
- **Use case page** — the full approval history and a link to every past
  run's leaderboard. If a candidate is pending, an **Approve** button sits
  right next to it — behind a confirmation dialog, and wired to the exact
  same `state.approve()` function the CLI's `approve` command calls, so
  there's still only one thing in this project that can change what
  `resolve()` returns.
- **Run detail page** — the leaderboard as a per-tier bar chart (score,
  price, would-ship rate on hover) with the full data table underneath it,
  so nothing is chart-only.

`run` and `status` print the relevant dashboard link after they finish, so
you don't have to know the URL scheme by heart.

---

## CLI reference

`--use-case`/`--use-case-name` also accept `--feature`/`--feature-name` as
plain aliases — same flag, same behavior, just matching the dashboard's own
word for the same thing ("AI feature"), since bouncing between the two
under two different names was its own small source of confusion.

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
| `cache [--clear-stale \| --clear-all]` | free | Inspects or reclaims the cached model responses |
| `scheduler run-due` | **$** | Runs every AI feature that's due right now, once, and exits |
| `scheduler serve [--interval-minutes <n>]` | **$** | Loops forever, re-checking every AI feature's schedule |
| `ui [--host <addr>] [--port <n>]` | free | Launches the local dashboard at `http://127.0.0.1:5000` |

Useful `run` flags: `--tier {free,paid-low,paid-mid,paid-high}` to restrict
which price band is benched, `--models a,b,c` to bench an explicit list
instead of the guardrail-filtered shortlist, `--limit N` to cap candidate
count, `--cache` to reuse answers already bought, `--no-health-check` to skip
the pre-run probe, `--yes` to skip the interactive cost confirmation (for scheduled
jobs — a `scheduler` run always behaves as if `--yes` was passed, since
nobody is there to answer). If installed with `pip install -e .`, every
command above also works as `modelcicd <command>` instead of
`python -m modelcicd.cli <command>`.

---

## Health check — catch a dead model before spending a run on it

A deprecated model id keeps looking fine everywhere it's read from. It sits
in the catalogue, passes every guardrail, gets picked as a candidate — and
only fails on a real call, minutes into a run. A dead **judge** is worse:
every answer gets paid for and then scored as a failure, so the whole run is
wasted.

So a run probes each candidate and the judge with one trivial call first. If
the judge is gone it refuses outright, before spending anything. Off with
`--no-health-check`; the cost is disclosed in the preview like everything
else that spends.

**Only a provable death drops a candidate — and the HTTP status does not tell
you that.** Checked live against OpenRouter:

| status | platform's message | truth |
|---|---|---|
| `404` | "unavailable for free… use this slug instead: `minimax/minimax-m3`" | **gone** — and the message names the fix |
| `404` | "Provider returned error" · `provider_name: Nvidia` | **not gone** — the upstream host had a moment |
| `400` | "`vendor/x` is not a valid model ID" | **gone** — never existed |

Two 404s meaning opposite things isn't an edge case, it's the normal
situation. The first version of this excluded on 404 and would have dropped
this project's own configured judge — whose 404 was a transient Nvidia error
— while missing a fake id entirely, because OpenRouter refuses those with a
400.

So the decision reads the platform's message, and it's deliberately
**one-directional**: a message must match a known-permanent phrase to be
fatal, and anything unrecognized falls through to *unknown*, which never
excludes. A new phrasing costs a wasted run rather than a working model
silently dropped. Rate limits and timeouts never exclude either — a 429 is a
fact about the minute, not the model.

**A passing probe proves one thing only: the id exists and answered.** It is
*not* a prediction that the model will handle your test cases. This project
already learned that the hard way — a model that answered a trivial probe
went on to fail the real scan prompt outright, narrating past the token limit
before reaching any JSON. A health check implying more than it knows would be
worse than none, because someone would trust it.

Nothing is cached here, deliberately: the whole point is to know what's dead
*now*. And whatever gets excluded is listed on the run page with the
platform's own words — a candidate that vanishes with no explanation is the
failure mode this project keeps having to design against.

---

## Measurement fingerprints — a trend line that can't lie about itself

A trend line asserts something: *this got better*. That's only true if both
scores came from the same measuring stick — and this tool actively encourages
you to improve your rubric, which changes the stick.

It happened in this repo's own data. `agent_router` scored **2.25**, then
**4.45**. No model changed between those runs; deterministic checks were added
to the feature. Both points sat on one line, reading as a large improvement
that nobody measured.

So every run now records a fingerprint of what was measured, and the trend
line has **three states** instead of two:

| | meaning |
|---|---|
| **solid** | same fingerprint — comparable |
| **dashed** | at least one run has no fingerprint. Drawn, but *unverified* |
| **gap** | fingerprints differ — provably not the same scale |

The dashed state matters more than it looks. A gap asserts "the measurement
changed"; a solid line asserts "it didn't". For a run recorded before
fingerprints existed neither is known, so it gets a dashed edge, which
asserts nothing. (I first rendered those as gaps. On real history that turned
four pre-fingerprint runs into four isolated dots and four "breaks" — which
reads as noise and teaches people to ignore breaks entirely. An unknown never
being resolved *favorably* doesn't mean it gets resolved unfavorably.)

**Five components, so "what changed?" is answerable** — not one opaque hash.
The run page shows each, and the CLI and dashboard name them the moment a run
drifts:

```
NOT COMPARABLE TO THE PREVIOUS RUN: the rubric changed since then, so this
score was produced by a different measuring stick. The trend line breaks
here rather than joining the two.
```

The components are the system prompt (what was asked), the test cases (what
it was asked about), the rubric (how it was judged), the deterministic checks
(what was verified), and the judge model — a different judge is a different
scale even against an identical rubric. Each edit moves exactly one.

**Guardrails are deliberately excluded.** Price ceilings and tiers decide
*which* candidates get measured, not how. Raising a ceiling adds rows to a
leaderboard; it doesn't make the existing rows mean something different.
Folding it in would cry wolf on a routine edit and train people to click past
the warning that matters.

The reason this is worth the trouble: in a live test, a rubric reword moved a
score from **4.45 to 4.40**. That looks perfectly continuous. Without the
fingerprint nobody would ever suspect those two numbers came from different
rubrics — which is the dangerous version of this bug, not the obvious one.

---

## Deterministic checks — the half of "is this good?" that needs no model

A rubric judged by an LLM is the right tool for *"is this reply helpful and
on-brand"*. It is the wrong tool for *"does the response have an `intent`
key"*, because that question has an answer, and asking a model to guess at it
puts noise into something that could have been certain.

So anything decidable in code is decided in code, free, before a judge call
is made — and the result blends into the same 1–5 score:

```yaml
# a router whose integration reads response.team
assertions:
  - type: has-keys
    value: [team]
    required: true                        # nothing to route on — don't even judge it
  - type: regex
    value: ^(billing|technical|general)$
    path: team
    weight: 2.0

testCases:
  - id: billing_double_charge
    input: I was charged twice last month and I need a refund.
    assertions:
      - type: equals                      # ADDS to the shared checks above
        value: billing
        path: team
```

That second check is the point. A model answering `{"team": "Billing
Department"}` scores full marks on *"is the reply clear and on-topic?"* and
breaks the router anyway. No rubric wording catches it; a regex does.

**Four types, deliberately** — `contains`, `equals`, `regex`, `has-keys`,
each also as `not-<type>`. This is not a metric library and isn't trying to
become one: no embedding similarity, no BLEU, no model-graded assertions
(that's what the rubric is for). Four checks cover the failures that actually
sink an integration — a required field went missing, a forbidden phrase came
back, a format changed shape.

| Option | Meaning |
|---|---|
| `path` | A field in the JSON response (`intent`, `data.reply`). Omit to check the whole answer. |
| `weight` | How much this counts in the blended 1–5 score, alongside the rubric's own weights. |
| `required` | Failing it disqualifies the answer **and skips its judge call** — no money spent scoring the prose of something already out. |
| `caseSensitive` | Off by default. Against model prose, `Refund` vs `refund` is almost never the difference you meant to catch. |

**A missing `path` fails; it does not skip.** Same rule the price guardrails
follow — an unknown is never resolved in the candidate's favor, because the
permissive version is how something broken gets quietly promoted.

**Test-case checks ADD to the use case's, unlike the rubric, which replaces.**
That difference is deliberate: two rubrics are competing scales for one
judgement, so a test case's own has to replace the default. Two assertions are
independent facts and combine fine. Under replace-semantics a shared "must
have a `team` key" would silently vanish from exactly the test cases whose
answers were pinned down most precisely.

**A bad check is refused at load, not mid-run.** An unparseable regex or a
typo'd type found thirty candidates into a bench has already cost the money
the check was meant to save, so `config.load` and the wizard both run the
real parser and name the file and test case at fault.

Set them per feature in the wizard, or once for a whole project under Project
Defaults. Every run's saved file records each check, its verdict, and *why* —
the part of a score a reader can verify without trusting the judge at all.

---

## Response caching — so iterating on a rubric doesn't re-buy the answers

Changing a rubric changes how answers are **scored**, not what the answers
**are**. Without a cache, fixing one word in one criterion re-generates every
candidate's answer to every test case — the expensive half of a run — to
arrive at the same answers it already had.

```bash
python -m modelcicd.cli run --use-case ... --cache   # reuse what's already bought
python -m modelcicd.cli cache                        # how much is stored, how much is stale
python -m modelcicd.cli cache --clear-stale          # reclaim only the expired entries
```

In the dashboard it's a checkbox on the run form, and the cost preview counts
the ready answers from local files first, so it quotes what the run will
actually buy rather than the full price.

**Off by default, and recorded when it's on.** A reused answer is a *replay*,
not a fresh measurement of what that model does today, so the run's own file
carries `cacheStats` — how many of its calls were replays — and the CLI and
dashboard both say so afterward. A trend chart that compares a replayed run
to a measured one at least can't do it silently.

Keyed on every argument that reaches the model: model id, the test-case
input, the system prompt, temperature, max tokens, and the endpoint URL (the
same id on two marketplaces is two deployments). Edit the prompt or the
rubric and you miss the cache, which is the correct answer. Entries expire
after 14 days, because a provider can change the weights behind a stable id.

**Failures are never cached, and the two noise checks never read it.**
A rate limit or a parse failure is a fact about one moment, not about the
model. And `judge.score_repeated` and `bench.resample_candidates_for_spread`
measure how far apart two *identical* calls land — served from a cache, every
repeat would return the first answer and the spread would come out as exactly
`0.00`, reading as a rock-steady model. Neither function accepts a cache
parameter at all, so it can't be passed by accident, and a test asserts that
stays true.

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
  cache.py           content-addressed model responses, so a rubric edit doesn't re-buy the answers
  endpoint_client.py the one place this project calls a use case's OWN live endpoint
  sandbox.py         run one candidate (through ITS OWN platform) or the live endpoint, against the test cases
  assertions.py      deterministic checks on an answer — decided in code, no model, free
  health.py          one tiny probe per model before a run — drops only what's provably gone
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


---

## Testing

```bash
python -m modelcicd.tests.test_modelcicd
```

No `pytest` dependency — a small, self-contained runner that checks the
specific edge cases the design relies on (negative-price sentinels, the
judge never being a candidate, tier ranking, stale-state bugs, and so on).

---

## Documentation

| Document | What's in it |
|---|---|
| [docs/GUIDE.md](docs/GUIDE.md) | Connecting a real application, repo scanning, project defaults, bulk-create, live endpoints, code patching, the scheduler. |
| [docs/DESIGN.md](docs/DESIGN.md) | Why it works this way — the constraints, the judge-bias mitigations, rate-limit honesty, and the invariants to preserve when extending it. |
| [CONTRIBUTING.md](CONTRIBUTING.md) | Dev setup, running the tests, and the conventions this codebase actually follows. |

---

## Contributing

Bug reports, ideas, and pull requests are welcome. See
[CONTRIBUTING.md](CONTRIBUTING.md) for setup, how to run the test suite, and the
conventions this codebase follows.

## License

Apache 2.0 — see [LICENSE](LICENSE).
