"""Exact-item operator quarantine; no delivery, retry, reset, or history rewriting.

Run ``python -m tools.async_delegation_quarantine --help`` in the intended
profile scope. Inspection is read-only; apply requires a fresh whole-row digest
and an external approval reference. This is an OS-owner administrative interface,
not a model tool or a substitute for operator authorization. No release API.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
from contextlib import closing
from pathlib import Path

from hermes_constants import get_hermes_home, hermes_home_key
from hermes_cli.sqlite_safe_read import connect_tracked


# Persisted guards outlive the Python runtime. Older versions understand neither
# the audit column nor retention exemption, but still execute SQLite triggers.
# Only explicitly quarantined rows are guarded; neighbors remain ordinary rows.
_GUARDS = {
    "async_quarantine_no_update": """CREATE TRIGGER async_quarantine_no_update
BEFORE UPDATE ON async_delegations WHEN OLD.quarantine_json IS NOT NULL
BEGIN SELECT RAISE(IGNORE); END""",
    "async_quarantine_no_delete": """CREATE TRIGGER async_quarantine_no_delete
BEFORE DELETE ON async_delegations WHEN OLD.quarantine_json IS NOT NULL
BEGIN SELECT RAISE(IGNORE); END""",
    "async_quarantine_no_replace": """CREATE TRIGGER async_quarantine_no_replace
BEFORE INSERT ON async_delegations
WHEN EXISTS (SELECT 1 FROM async_delegations
             WHERE delegation_id=NEW.delegation_id AND quarantine_json IS NOT NULL)
BEGIN SELECT RAISE(ABORT, 'delegation_quarantined'); END""",
}


def _guards(conn, *, install: bool = False) -> None:
    """Install transactionally or verify exact guards, never replace unknown DDL."""
    for name, sql in _GUARDS.items():
        stored = conn.execute("SELECT sql FROM sqlite_master WHERE type='trigger' AND name=?", (name,)).fetchone()
        if stored is None and install:
            conn.execute(sql)
        elif stored is None or stored[0] != sql:
            raise ValueError("quarantine_guard_mismatch")


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _fingerprint(row: dict) -> str:
    return _digest(json.dumps({k: v for k, v in row.items() if k != "quarantine_json"},
                              sort_keys=True, separators=(",", ":")))


def _home(expected_home: str) -> Path:
    home = get_hermes_home()
    if not expected_home or hermes_home_key(Path(expected_home)) != hermes_home_key(home):
        raise ValueError("quarantine_profile_mismatch")
    if not (home / "state.db").is_file():
        raise ValueError("quarantine_missing_ledger")
    return home


def _read(conn, delegation_id: str) -> dict:
    cursor = conn.execute("SELECT * FROM async_delegations WHERE delegation_id=?", (delegation_id,))
    row = cursor.fetchone()
    if row is None:
        raise ValueError("quarantine_missing_target")
    return dict(zip((col[0] for col in cursor.description), row))


def _validate(conn, row: dict) -> None:
    if (row["state"] not in {"error", "failed", "failure", "timeout", "stalled", "interrupted"}
            or row["delivery_state"] != "pending" or row["delivery_claim"] is not None
            or row["delivery_claimed_at"] is not None or row["delivered_at"] is not None
            or row["owner_execution_id"] is not None):
        raise ValueError("quarantine_requires_unclaimed_failed_pending_target")
    for field in ("event_json", "result_json", "task_json"):
        if not row[field] or not isinstance(json.loads(row[field]), dict):
            raise ValueError("quarantine_incomplete_history")
    event = json.loads(row["event_json"])
    if (event.get("delegation_id") != row["delegation_id"] or event.get("type") != "async_delegation"
            or not row["parent_session_id"] or event.get("parent_session_id") != row["parent_session_id"]
            or not row["origin_session"] or event.get("session_key") != row["origin_session"]):
        raise ValueError("quarantine_ambiguous_identity")
    # This bounded interface will not silently cancel or rewrite a resumable child.
    # A unit with any such record needs a separately reviewed child disposition.
    if conn.execute("SELECT 1 FROM resumable_subagents WHERE json_extract(record_json, '$.unit_id')=? LIMIT 1",
                    (row["delegation_id"],)).fetchone():
        raise ValueError("quarantine_has_resumable_children")


def _receipt(row: dict) -> dict:
    return {"delegation_id": row["delegation_id"], "state": row["state"],
            "delivery_state": row["delivery_state"], "delivery_attempts": row["delivery_attempts"],
            "row_sha256": _fingerprint(row),
            "history_sha256": {key: _digest(row[key] or "") for key in ("event_json", "result_json", "task_json")},
            "audit": json.loads(row.get("quarantine_json") or "null")}


def inspect_quarantine(delegation_id: str, *, expected_home: str) -> dict:
    """Read only, including on old-schema ledgers; never initializes a database."""
    home = _home(expected_home)
    path = (home / "state.db").resolve()
    with closing(connect_tracked(path.as_uri() + "?mode=ro", uri=True, tracking_path=path)) as conn:
        row = _read(conn, delegation_id)
        if not row.get("quarantine_json"):
            _validate(conn, row)
        else:
            _guards(conn)
        return _receipt(row)


def quarantine_completion(delegation_id: str, *, expected_home: str, expected_sha256: str,
                          authorization_ref: str) -> dict:
    """Atomically quarantine one unclaimed failure, retaining every original field except delivery_state.

    The audit stores the whole-row preimage digest, payload digests and approval
    reference in the same transaction. Repeated identical requests return the
    original audit; changed approval, payload, target, or profile fails closed.
    """
    from hermes_cli.sqlite_util import transaction
    from tools import async_delegation as ad
    from tools.async_delegation_worker import require_ledger_owner

    require_ledger_owner()
    home = _home(expected_home)
    if not re.fullmatch(r"[0-9a-f]{64}", expected_sha256):
        raise ValueError("quarantine_invalid_digest")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9:/_.#-]{0,255}", authorization_ref):
        raise ValueError("quarantine_invalid_authorization_reference")
    with ad._DB_LOCK, transaction(ad._connect(), immediate=True) as conn:
        row = _read(conn, delegation_id)
        if row.get("quarantine_json"):
            _guards(conn)
            audit = json.loads(row["quarantine_json"])
            original = {**row, "delivery_state": "pending"}
            if (row["delivery_state"] != "quarantined" or _fingerprint(original) != expected_sha256
                    or audit["preimage_sha256"] != expected_sha256 or audit["authorization_ref"] != authorization_ref
                    or audit["home_sha256"] != _digest(hermes_home_key(home))):
                raise ValueError("quarantine_idempotency_conflict")
            return _receipt(row)
        _validate(conn, row)
        if _fingerprint(row) != expected_sha256:
            raise ValueError("quarantine_stale_preimage")
        _guards(conn, install=True)
        audit = {"version": 1, "action": "quarantine_without_delivery", "at": time.time(),
                 "authorization_ref": authorization_ref, "preimage_sha256": expected_sha256,
                 "guard_sha256": _digest(json.dumps(_GUARDS, sort_keys=True)),
                 "home_sha256": _digest(hermes_home_key(home)), "history_sha256": _receipt(row)["history_sha256"]}
        conn.execute("UPDATE async_delegations SET delivery_state='quarantined', quarantine_json=? "
                     "WHERE delegation_id=?", (json.dumps(audit, sort_keys=True), delegation_id))
        return _receipt(_read(conn, delegation_id))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("delegation_id")
    parser.add_argument("--expected-home", required=True)
    parser.add_argument("--apply", action="store_true", help="Requires separately authorized exact target")
    parser.add_argument("--expected-sha256")
    parser.add_argument("--authorization-ref")
    args = parser.parse_args(argv)
    if args.apply:
        if not args.expected_sha256 or not args.authorization_ref:
            parser.error("--apply requires --expected-sha256 and --authorization-ref")
        result = quarantine_completion(args.delegation_id, expected_home=args.expected_home,
                                       expected_sha256=args.expected_sha256, authorization_ref=args.authorization_ref)
    else:
        result = inspect_quarantine(args.delegation_id, expected_home=args.expected_home)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
