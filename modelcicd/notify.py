"""Tell a human a better model showed up. They approve it, nothing else does.

EMAIL, NOT A DASHBOARD, FOR V1. Self-hosted and single-tenant means there is no
always-on service to push a notification to — the email IS the interrupt, sent
from whatever kicked off the scheduled run (cron, a GitHub Action, a Task
Scheduler job), and it names the exact command that makes the promotion happen.

NEVER SENT FOR NOISE. `state.record_run` already applied the use case's
`minImprovement` threshold before flagging anything pending — this module only
sends when there IS something pending, so a human's inbox reflects real
candidates, not every run that happened to complete.

FAILS QUIETLY. A missing SMTP config or a send failure is printed and swallowed
— the bench run itself, and the state it wrote, are the parts of this pipeline
that must survive; the email is best-effort on top of them.
"""
import os
import smtplib
from email.message import EmailMessage
from typing import Optional


def send_pending(use_case: str, state: dict, board_report: str,
                 to: Optional[str] = None) -> bool:
    pending = state.get("pending")
    if not pending:
        return False

    to = to or os.environ.get("MODELCICD_NOTIFY_EMAIL")
    if not to:
        print(f"[modelcicd] a candidate is pending for {use_case!r} "
              f"({pending['model']}, score {pending['score']}) but no notify "
              f"email is configured — see it with "
              f"`python -m modelcicd.cli status --use-case {use_case}`.")
        return False

    host = os.environ.get("SMTP_HOST")
    if not host:
        print(f"[modelcicd] SMTP_HOST is not set — cannot email. The candidate "
              f"is still recorded as pending; approve it directly:\n"
              f"    python -m modelcicd.cli approve --use-case {use_case}")
        return False

    approved = state.get("approvedModel") or "(nothing approved yet)"
    approved_score = state.get("approvedScore")
    subject = f"MoCICD: a better model for '{use_case}'"
    body = f"""A scheduled bench run found a candidate that beats what is
currently approved for '{use_case}'.

  currently approved   {approved}"""
    if approved_score is not None:
        body += f"  (score {approved_score})"
    body += f"""
  candidate            {pending['model']}  (score {pending['score']})
  found at             {pending['foundAt']}

Nothing has changed in production. This is a proposal, not a promotion.

To approve it:
    python -m modelcicd.cli approve --use-case {use_case}

To approve a DIFFERENT model from this run instead:
    python -m modelcicd.cli approve --use-case {use_case} --model <model-id>

Full leaderboard from this run:

{board_report}
"""
    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = os.environ.get("SMTP_FROM", "modelcicd@localhost")
    msg["To"] = to
    msg.set_content(body)

    port = int(os.environ.get("SMTP_PORT", "587"))
    user = os.environ.get("SMTP_USER")
    password = os.environ.get("SMTP_PASSWORD")
    try:
        with smtplib.SMTP(host, port, timeout=15) as server:
            if os.environ.get("SMTP_STARTTLS", "1") == "1":
                server.starttls()
            if user and password:
                server.login(user, password)
            server.send_message(msg)
        print(f"[modelcicd] notified {to} about a pending candidate for "
              f"{use_case!r}.")
        return True
    except Exception as exc:                        # noqa: BLE001
        print(f"[modelcicd] could not send the notification email "
              f"({str(exc)[:160]}). The candidate is still recorded as pending.")
        return False
