"""Who may reach this dashboard, once it's reachable by more than just you.

WHY THIS EXISTS AT ALL. Every route in `dashboard.py` was written assuming
only the person running `cli ui` on their own machine could ever open it —
`127.0.0.1` was the ENTIRE security model. Nothing enforced that; it was
true only because nothing outside localhost could route to the process. The
moment this dashboard needs to answer a request from somewhere else (a
website's own backend, a real deployed host), that assumption is gone, and
eleven POST routes that approve models, spend real money, write API keys
into `.env`, and patch a connected app's actual source code become
reachable by anyone who finds the URL.

ONE SHARED SECRET, OFF BY DEFAULT. Set `MODELCICD_API_KEY` and every route
— the dashboard's pages and the read-only `/api/*` routes alike — requires
it. Leave it unset and NOTHING CHANGES: the existing localhost-only
workflow, and every one of this project's existing tests that call
`dashboard.create_app()` with no key configured, behaves exactly as it
always has. Same opt-in shape as `--cache`, `--resample-shortlist`, and
`health_check` elsewhere in this project — the safe legacy behavior is what
happens when you do nothing.

TWO WAYS TO PRESENT THE SAME KEY, because a person in a browser and a
server calling the API need different mechanics for the same fact:

    a server     sends `Authorization: Bearer <key>` on every request — no
                cookie jar, no login step, exactly what a website's own
                backend already does for every other API it calls.
    a person     visits `/login`, pastes the key once, gets a signed
                session cookie — so clicking "Approve" doesn't require
                attaching a header by hand.

Both check the SAME key. `check()` compares in constant time — never `==`,
which leaks how many leading characters matched through how long the
comparison takes.

WHAT THIS DELIBERATELY DOES NOT COVER. There is still no per-caller
identity, no rate limiting, and no CSRF protection on the session-cookie
path (a malicious page could auto-submit a form to a logged-in browser's
dashboard). This is one shared secret for one operator's own deployment,
matching the project's existing single-tenant design — not a login system
for multiple distinct users. See the README for what that would actually
require.
"""
import hashlib
import hmac
import os
from typing import Optional

API_KEY_ENV = "MODELCICD_API_KEY"


def configured() -> bool:
    """Whether auth is turned on at all. Read live from the environment on
    every call — like every other secret in this project, so a key set
    while a dashboard process is already running (via `set_dashboard_key`,
    which updates `os.environ` immediately) takes effect on the very next
    request without a restart."""
    return bool((os.environ.get(API_KEY_ENV) or "").strip())


def check(candidate: Optional[str]) -> bool:
    """Constant-time comparison against the configured key.

    Returns False — never raises — for an unset key or an empty candidate,
    so a caller can use this directly as a gate without first checking
    `configured()` separately. `hmac.compare_digest` requires both sides be
    the same type; both are coerced to bytes explicitly rather than relying
    on str/str comparison happening to work, so this stays correct even if
    Werkzeug ever hands back header bytes instead of str.
    """
    expected = (os.environ.get(API_KEY_ENV) or "").strip()
    if not expected or not candidate:
        return False
    return hmac.compare_digest(expected.encode("utf-8"), candidate.strip().encode("utf-8"))


# A plain cookie this module signs and checks BY HAND, deliberately not
# Flask's built-in `session`. First attempt used `session[...]`, signed with
# a secret set fresh in a `before_request` hook — and it doesn't work:
# Flask OPENS the session object when the request context is first pushed,
# BEFORE any `before_request` handler runs, using whatever `app.secret_key`
# was at THAT moment. Set it later in the same request and the session
# object handed to the view is already a `NullSession`, which raises the
# instant anything tries to write to it — no matter what `app.secret_key`
# becomes afterward. Verified live: reproduced the exact `RuntimeError`
# outside the test suite before finding the cause. A hand-rolled cookie
# sidesteps that timing entirely — nothing here depends on when in the
# request lifecycle it's computed, because it's just a value, checked fresh
# against the live key every time, not a session object Flask manages.
COOKIE_NAME = "modelcicd_auth"


def session_token() -> str:
    """The value a signed-in browser's cookie should hold — an HMAC of a
    fixed string, keyed on the CURRENT `MODELCICD_API_KEY`. Recomputed here,
    never cached: rotating the key changes what this returns, so every
    cookie signed under the old key stops matching the moment
    `session_cookie_valid` next checks it — that's what makes rotation
    invalidate every existing session without a separate revocation list.
    Empty string when auth is off, which never validates against anything."""
    if not configured():
        return ""
    return hmac.new(os.environ[API_KEY_ENV].strip().encode("utf-8"), b"modelcicd-authed",
                    hashlib.sha256).hexdigest()


def session_cookie_valid(token: Optional[str]) -> bool:
    """Constant-time check of a cookie value against what `session_token()`
    computes right now. False for a missing cookie or auth being off —
    never raises, same contract as `check()`."""
    if not token or not configured():
        return False
    return hmac.compare_digest(session_token(), token)


def bearer_token(header_value: Optional[str]) -> Optional[str]:
    """Pulls the token out of an `Authorization: Bearer <token>` header, or
    None if the header is missing or a different scheme."""
    if not header_value or not header_value.lower().startswith("bearer "):
        return None
    return header_value[7:].strip() or None
