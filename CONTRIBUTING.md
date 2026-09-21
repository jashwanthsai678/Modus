# Contributing to MoCICD

Thanks for looking at this. This doc covers how to get set up, how to run
things, and the conventions this codebase actually follows in practice — so
you can match them instead of guessing from reading everything first.

## Getting set up

```
git clone https://github.com/jashwanthsai678/mocicd.git
cd mocicd
python -m venv .venv
.venv\Scripts\activate        (Windows)   /   source .venv/bin/activate   (macOS/Linux)
pip install -e .
copy .env.example .env        (Windows)   /   cp .env.example .env       (macOS/Linux)
```

You only need a real API key (`OPENROUTER_API_KEY` in `.env`) for the parts
that actually call a model — scanning a repo or running a bench. Everything
else (installing, running the dashboard, connecting a project, defining a
feature, running the test suite) works with no key at all.

See `README.md` for the full walkthrough of using the app itself. This doc
is about changing its code.

## Running the tests

```
python -m modelcicd.tests.test_modelcicd
```

No pytest, no test-runner config, nothing extra to install. It's one module
with plain `test_*` functions, run top to bottom, using a tiny `check(name,
condition, detail)` helper instead of `assert` — a failed check prints and
is collected, but doesn't stop the rest of the suite from running, so one
early failure doesn't hide a later one. It's also entirely free — no real
model calls, everything that would cost money is faked with a stub, since
this suite exists to catch the errors that would otherwise cost money or
produce a wrong promotion, not to test the models themselves.

Run this before opening a PR. CI runs it too, across Python 3.9 (the
project's declared minimum) and a couple of newer versions — 3.9 is kept in
the matrix on purpose, since a transitive dependency silently requiring a
newer Python than it claims to support is a real failure mode this project
already hit once.

## Conventions this codebase actually follows

These aren't arbitrary style preferences — each one exists because of a
specific failure mode this project is designed to avoid. Worth understanding
the *why*, not just matching the pattern:

- **One implementation, two doors.** The CLI and the dashboard always call
  the exact same underlying functions (`state.approve()`, `project.create()`,
  `wizard.to_yaml()`, etc.) — never two versions of the same logic. If you're
  adding a feature reachable from both, write it once in the relevant module
  and have both interfaces call it.
- **State-changing functions are single-writer.** `state.approve()` is the
  only function allowed to change what `resolve()` returns; `state.reject()`
  is the only other function allowed to clear a pending candidate. Don't
  mutate state files directly from a route or CLI command — route through
  the function that owns that invariant.
- **Unknown is never assumed favorable.** A guardrail that can't determine a
  model's price, JSON-mode support, or rate limit treats that as *unsafe*,
  not as "probably fine." Never fill in a guessed default that happens to
  let something through.
- **Comments explain the non-obvious why, not the what.** A well-named
  function doesn't need a comment restating what it does. Write one only
  for a hidden constraint, a subtle invariant, or a reason a reader would
  otherwise have to reconstruct by digging through git history.
- **No speculative abstraction.** Don't add a config flag, a plugin hook, or
  a generalized version of something for a use case that doesn't exist yet.
  Three similar lines beat a premature shared helper.
- **Anything that spends money shows the cost and asks first.** Scanning a
  repo and running a bench both print what they're about to do and require
  explicit confirmation before making a real API call. Any new feature that
  calls a model needs the same pattern, not a silent call.
- **A failure is a result, not an exception to hide.** A candidate model
  that errors out is recorded on the leaderboard with its error, not
  silently dropped — the same principle applies anywhere else a partial
  failure could otherwise look identical to success. (This project already
  shipped one real bug from violating this — a scan that failed outright
  looked exactly like a scan that found nothing, because the error was
  discarded before anyone decided what to show. Don't reintroduce that
  shape of bug elsewhere.)

## Proposing a change

- Open an issue first for anything non-trivial, so the approach can be
  agreed on before code is written.
- Keep PRs scoped to one change — easier to review, easier to revert if
  something's wrong.
- Add or update a test in `modelcicd/tests/test_modelcicd.py` for any new
  behavior, following the existing `check()` style rather than introducing
  a second testing approach.
- If your change touches anything documented in `README.md` (a new CLI
  command, a new guardrail, a new setup step), update it in the same PR.
