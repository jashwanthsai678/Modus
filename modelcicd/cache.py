"""A content-addressed cache of model responses, so iterating on a rubric
doesn't re-buy the answers.

WHAT THIS IS ACTUALLY FOR. Changing a rubric changes how answers are SCORED,
not what the answers ARE. Without a cache, fixing one word in one criterion
re-generates every candidate's answer to every test case — the expensive
half of a run — to arrive at the same answers it already had. With one, that
iteration costs only the judge calls, which is what you actually changed.

ONLY SUCCESSES ARE STORED. A failure, a rate limit, a timeout is a fact
about one moment, not about the model — caching it would pin a candidate to
a bad afternoon for two weeks. Every failure path leaves the cache untouched
and will be retried on the next run, exactly as if no cache existed.

A CACHE MUST NOT BE ABLE TO BREAK A RUN. Every read here is defensive: an
unreadable file, a truncated write, a JSON error, a permission problem, a
missing directory — all of them return a miss and, where it's a stored
entry's own fault, delete it. Nothing in this module raises into a bench.

ENTRIES EXPIRE, BECAUSE A MODEL ID IS NOT A MODEL. Providers update the
weights behind a stable id, and marketplaces re-point an id at a different
host. A month-old answer is not necessarily that id's answer any more, so
entries carry a TTL (14 days by default) and a stale one is a miss.

THE TWO PLACES THIS MUST NEVER BE USED — and it is worth being blunt,
because the failure is silent and it invalidates a number the leaderboard
shows a human:

    judge.score_repeated               measures JUDGE-side noise by scoring
                                       the SAME answer several times
    bench.resample_candidates_for_spread
                                       measures CANDIDATE-side noise by
                                       re-generating the SAME answer

Both work by making a call that is byte-for-byte identical to one they just
made, and reading how much the result moved. A cache keyed on the call would
return the first result every time, the spread would come out as exactly
0.0, and the leaderboard would report "no measurable noise" about a
measurement that never happened. Neither of them is ever handed a cache —
they don't take the parameter at all, which is the point: it can't be passed
by accident.

NO GLOBAL SWITCH. A cache is an object a caller creates and passes down
(`runner.execute(use_cache=True)` builds one). It is deliberately not a
module-level flag that `enable()` flips: the dashboard serves requests in
worker threads, and two concurrent runs — one asking to reuse answers, one
asking to measure fresh ones — must not be able to reach into each other.
"""
import hashlib
import json
import os
import shutil
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

DEFAULT_TTL_DAYS = 14
SCHEMA = 1


def default_root() -> Path:
    return Path(__file__).resolve().parent.parent / "cache"


def fingerprint(*, model: str, prompt: str, system: Optional[str],
                temperature: float, max_tokens: int, base_url: str,
                required: tuple = ()) -> str:
    """The cache key: everything that can change the response.

    EVERY ARGUMENT THAT REACHES THE MODEL IS IN HERE. Miss one and the
    cache serves an answer to a different question — the one failure mode
    of a cache that looks exactly like a working cache. `system` is in it
    because that's the use case's prompt (edit the prompt, get fresh
    answers, which is correct). `temperature` and `max_tokens` are in it
    because they change the output. `base_url` is, because the same model
    id served by two marketplaces is two different deployments. `required`
    is, because it decides whether a response was acceptable at all.

    SCHEMA IS IN IT TOO, so changing what's stored in an entry
    invalidates every old entry instead of misreading them.
    """
    payload = json.dumps({
        "schema": SCHEMA, "model": model, "prompt": prompt,
        "system": system or "", "temperature": temperature,
        "max_tokens": max_tokens, "base_url": base_url,
        "required": sorted(required or ()),
    }, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class Cache:
    """One cache rooted at a directory. Not shared between threads by
    design — each run builds its own — but concurrent instances over the
    same directory are safe anyway: writes are atomic replaces, and a read
    that loses a race just misses."""

    def __init__(self, root: Optional[Path] = None, *,
                 ttl_days: int = DEFAULT_TTL_DAYS) -> None:
        self.root = Path(root) if root else default_root()
        self.ttl = timedelta(days=ttl_days)
        self.hits = 0
        self.misses = 0
        self.writes = 0

    # The duck-typed contract `client.call_json` expects, so that file can
    # use a cache without importing this one. Delegates to the module-level
    # function, which stays the single definition of what a key is.
    fingerprint = staticmethod(fingerprint)

    # ── paths ──────────────────────────────────────────────────────────
    #
    # SHARDED BY THE FIRST TWO HEX CHARACTERS. A long-lived cache over a
    # forty-candidate field reaches tens of thousands of entries, and a
    # single directory that size is slow to open on Windows and unpleasant
    # to look at. 256 shards keeps both fine.

    def _path(self, key: str) -> Path:
        return self.root / key[:2] / f"{key}.json"

    # ── read ───────────────────────────────────────────────────────────

    def get(self, key: str) -> Optional[dict]:
        """The stored response, or None for a miss. Never raises."""
        path = self._path(key)
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            self.misses += 1
            return None
        except Exception:                            # noqa: BLE001
            # Corrupt or half-written: the entry's own fault, so drop it
            # rather than miss on it forever.
            self._discard(path)
            self.misses += 1
            return None

        if raw.get("schema") != SCHEMA or "value" not in raw:
            self._discard(path)
            self.misses += 1
            return None

        saved_at = _parse_time(raw.get("savedAt"))
        if saved_at is None or datetime.now(timezone.utc) - saved_at > self.ttl:
            self._discard(path)
            self.misses += 1
            return None

        self.hits += 1
        return raw["value"]

    # ── write ──────────────────────────────────────────────────────────

    def put(self, key: str, value: dict, *, model: str = "") -> None:
        """Stores one successful response. Never raises — a cache that
        can't be written is a cache that isn't used, not a failed run."""
        path = self._path(key)
        entry = {"schema": SCHEMA, "savedAt": datetime.now(timezone.utc).isoformat(),
                 "model": model, "value": value}
        tmp = path.with_suffix(f".{os.getpid()}.tmp")
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp.write_text(json.dumps(entry, ensure_ascii=False), encoding="utf-8")
            os.replace(tmp, path)                    # atomic: no half-entry is ever readable
            self.writes += 1
        except Exception:                            # noqa: BLE001
            self._discard(tmp)

    @staticmethod
    def _discard(path: Path) -> None:
        try:
            path.unlink()
        except Exception:                            # noqa: BLE001
            pass

    # ── reporting ──────────────────────────────────────────────────────

    def stats(self) -> dict:
        """What a run file records about itself, so a leaderboard built
        partly from replayed answers can never be mistaken for one where
        every answer was measured fresh."""
        looked_up = self.hits + self.misses
        return {"hits": self.hits, "misses": self.misses, "writes": self.writes,
                "hitRate": round(self.hits / looked_up, 3) if looked_up else None}


def _parse_time(value) -> Optional[datetime]:
    if not isinstance(value, str):
        return None
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


# ── maintenance, for the CLI ────────────────────────────────────────────

def describe(root: Optional[Path] = None, *, ttl_days: int = DEFAULT_TTL_DAYS) -> dict:
    """How big the cache is and how much of it is already stale — a plain
    count, so `cache` can be answered without loading every entry."""
    root = Path(root) if root else default_root()
    if not root.exists():
        return {"root": str(root), "entries": 0, "stale": 0, "bytes": 0}
    entries = stale = size = 0
    cutoff = datetime.now(timezone.utc) - timedelta(days=ttl_days)
    for path in root.rglob("*.json"):
        entries += 1
        try:
            size += path.stat().st_size
            saved = _parse_time(json.loads(path.read_text(encoding="utf-8")).get("savedAt"))
        except Exception:                            # noqa: BLE001
            stale += 1                               # unreadable counts as reclaimable
            continue
        if saved is None or saved < cutoff:
            stale += 1
    return {"root": str(root), "entries": entries, "stale": stale, "bytes": size}


def clear(root: Optional[Path] = None, *, stale_only: bool = False,
          ttl_days: int = DEFAULT_TTL_DAYS) -> int:
    """Deletes cached responses. Returns how many entries went.

    `stale_only` is the safe default a person usually wants — it reclaims
    expired entries without throwing away answers a rubric edit is about
    to reuse."""
    root = Path(root) if root else default_root()
    if not root.exists():
        return 0
    if not stale_only:
        removed = sum(1 for _ in root.rglob("*.json"))
        shutil.rmtree(root, ignore_errors=True)
        return removed
    cutoff = datetime.now(timezone.utc) - timedelta(days=ttl_days)
    removed = 0
    for path in root.rglob("*.json"):
        try:
            saved = _parse_time(json.loads(path.read_text(encoding="utf-8")).get("savedAt"))
        except Exception:                            # noqa: BLE001
            saved = None
        if saved is None or saved < cutoff:
            Cache._discard(path)
            removed += 1
    return removed
