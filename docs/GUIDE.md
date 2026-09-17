# Feature guide

Reference for the parts of Modus you reach for after the
[Quickstart](../README.md#quickstart) — connecting a real application,
scanning its code, comparing against a live endpoint, patching an approved
model back into source, and running the whole loop on a schedule.

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
(the default scan model is a free one), shown and confirmed first, exactly
like `run`. A scan only ever *proposes*: picking a candidate pre-fills the
wizard's system prompt and code target, both still shown and editable before
anything is saved. The same flow exists in the dashboard: a "Scan repo for
AI features" button on the project page, a results list, "Use this" into the
same wizard form.

**It reads more than the call site itself.** For each candidate, it also
follows that file's own local imports (one hop, best-effort) so a prompt
built in a separate file still gets attributed correctly, and it reports
`inputStructure`/`outputStructure` — what the code actually sends and
expects back, read from the real variables and parsing logic, not guessed.

**Errors are never silently hidden.** A file that fails to scan (a broken
key, a rate limit, anything) is shown distinctly from a file that was
genuinely clean — the two used to look identical, which was a real bug this
project shipped and fixed: a failed scan reporting "no likely call sites
found" is the one thing worse than an error, because it looks like success.

**Read the confidence, don't just read the result.** Every candidate carries
a `confidence: high|low` — the scanning model's own certainty that this is
really an LLM call, not a second, independent check. A low-confidence one is
flagged distinctly ("verify before using") rather than presented the same as
a clean hit, and a scan that finds nothing or only low-confidence candidates
points straight at `wizard --project <slug>` (no `--from-scan`) as the
reliable fallback. This matters most exactly where it's weakest: code that
builds a prompt dynamically, or through an agent/tool-calling framework, or
across more than one hop of imports, is real code the scanner is likely to
miss or misread — it reads for a literal call site, not that kind of
indirection. Don't read a quiet scan as "there's nothing here."

### Project defaults, and turning every scan result into a feature at once

Every feature needs a rubric, a price ceiling, and a judge — re-typing the
same ones for a project with several AI features is exactly the kind of
manual work this tool exists to remove. `/projects/<slug>/defaults` (or
`project set-defaults` from the CLI) sets that once; every **new** feature
starts from it, and it never rewrites a `use_case.yaml` already saved —
editing the template doesn't rewrite documents already made from it.

With that set, the scan results page offers "Create AI features from all N
candidate(s)": for every candidate, it applies the project's shared
defaults, and drafts realistic test cases (and a candidate reference answer)
from that feature's own prompt and input/output structure using a model.
**Nothing saves until you've reviewed it** — a screen shows every drafted
feature's name, prompt, and test cases, all still editable, with a per-item
"skip this one" if a draft isn't good enough to keep. This is the one place
in the whole tool where a model's own draft becomes ground truth (the
generated reference answer) rather than a human writing it from scratch —
which is exactly why it's never saved without that review screen first.

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

