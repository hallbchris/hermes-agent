"""Exact operator quarantine preserves failed history without replay or re-execution."""
import json
import queue
import sqlite3
import time

import pytest

from hermes_constants import get_hermes_home
from tools import async_delegation as ad
from tools.async_delegation_quarantine import inspect_quarantine, main, quarantine_completion


@pytest.fixture(autouse=True)
def clean_state():
    ad._reset_for_tests()
    yield
    ad._reset_for_tests()


def seed(delegation_id="held"):
    event = {"type": "async_delegation", "delegation_id": delegation_id, "session_key": "fixture-route",
             "parent_session_id": "fixture-parent", "dispatched_at": time.time(), "completed_at": time.time(),
             "status": "error", "goal": "harmless fixture"}
    ad._persist_dispatch(event)
    ad._persist_completion(event, {"status": "error", "error": "retained original failure"})
    return event


def row(delegation_id="held"):
    with ad._transaction() as conn:
        conn.row_factory = sqlite3.Row
        return dict(conn.execute("SELECT * FROM async_delegations WHERE delegation_id=?", (delegation_id,)).fetchone())


def arguments(delegation_id="held"):
    home = str(get_hermes_home())
    receipt = inspect_quarantine(delegation_id, expected_home=home)
    return {"expected_home": home, "expected_sha256": receipt["row_sha256"],
            "authorization_ref": "approval:fixture/exact-item"}


def test_quarantine_history_duplicate_restart_and_retention(monkeypatch):
    event = seed()
    seed("neighbor")
    before, neighbor = row(), row("neighbor")
    args = arguments()
    first = quarantine_completion("held", **args)
    assert first == quarantine_completion("held", **args)
    after = row()
    assert after["delivery_state"] == "quarantined"
    assert {k: v for k, v in after.items() if k not in ("delivery_state", "quarantine_json")} == {
        k: v for k, v in before.items() if k not in ("delivery_state", "quarantine_json")}
    assert row("neighbor") == neighbor
    assert first["audit"]["preimage_sha256"] == args["expected_sha256"]
    assert first["audit"]["history_sha256"] == first["history_sha256"]
    assert not ad.claim_event_delivery(event, "fixture-consumer")
    assert not ad.mark_completion_delivered("held")
    assert not ad.complete_completion_delivery("held", "invented")
    assert not ad.drop_completion_delivery("held", "invented")
    assert not ad.defer_completion_delivery("held", "invented")
    assert not ad.release_completion_delivery("held", "invented")
    offered = queue.Queue()
    assert not ad.requeue_admitted_completion("held", offered)
    with pytest.raises(ValueError, match="delegation_quarantined"):
        ad._persist_dispatch(event)
    with pytest.raises(ValueError, match="delegation_quarantined"):
        ad._persist_completion(event, {"status": "completed"})
    calls = []
    with pytest.raises(ValueError, match="delegation_quarantined"):
        ad.dispatch_async_delegation_batch(delegation_id="held", goals=["harmless fixture"], context=None,
                                          toolsets=None, role="", model=None, session_key="fixture-route",
                                          parent_session_id="fixture-parent", runner=lambda: calls.append("ran"))
    assert calls == [] and ad.active_count() == 0
    ad._reset_for_tests()
    assert ad.restore_undelivered_completions(offered) == 1
    assert offered.get_nowait()["delegation_id"] == "neighbor"
    assert offered.empty()
    assert ad.sweep_orphaned_completions(offered) == 0
    monkeypatch.setattr(ad, "_MAX_RETAINED_COMPLETED", 0)
    monkeypatch.setattr(ad, "_MAX_DURABLE_PENDING", 0)
    ad._prune_durable_records()
    assert row() == after
    assert ad.failed_delegations_for_session(parent_session_id="fixture-parent")[0]["error"] == "retained original failure"


@pytest.mark.parametrize("changes", [
    {"state": "running"}, {"state": "finalizing"}, {"state": "completed"},
    {"delivery_state": "delivered"}, {"delivery_state": "dropped"},
    {"delivery_claim": "consumer"}, {"delivery_claimed_at": 1.0}, {"delivered_at": 1.0},
    {"event_json": None}, {"result_json": "[]"}, {"parent_session_id": "wrong"},
    {"origin_session": "wrong"}, {"owner_execution_id": "registered-worker"},
])
def test_refuse_unsafe_or_ambiguous_target(changes):
    seed()
    args = arguments()
    with ad._transaction() as conn:
        conn.execute("UPDATE async_delegations SET " + ",".join(f"{key}=?" for key in changes)
                     + " WHERE delegation_id='held'", tuple(changes.values()))
    before = row()
    with pytest.raises(ValueError):
        quarantine_completion("held", **args)
    assert row() == before


def test_refuse_stale_profile_missing_and_changed_approval(tmp_path):
    seed()
    args = arguments()
    before = row()
    for target, updates in [("missing", {}), ("held", {"expected_sha256": "0" * 64}),
                            ("held", {"expected_home": str(tmp_path / "wrong-profile")}),
                            ("held", {"authorization_ref": ""})]:
        with pytest.raises(ValueError):
            quarantine_completion(target, **(args | updates))
        assert row() == before
    quarantine_completion("held", **args)
    after = row()
    with pytest.raises(ValueError, match="idempotency_conflict"):
        quarantine_completion("held", **(args | {"authorization_ref": "approval:other"}))
    assert row() == after


def test_refuse_any_resumable_child():
    seed()
    args = arguments()
    with ad._transaction() as conn:
        conn.execute("INSERT INTO resumable_subagents (subagent_id,state,record_json,created_at,updated_at) "
                     "VALUES ('child','resumable',?,1,1)", (json.dumps({"unit_id": "held"}),))
    before = row()
    with pytest.raises(ValueError, match="resumable_children"):
        quarantine_completion("held", **args)
    assert row() == before


def test_readonly_old_schema_migration_and_cli(capsys):
    seed()
    # A previous-shape persisted ledger, not only a fresh-schema fixture.
    with ad._transaction() as conn:
        conn.execute("ALTER TABLE async_delegations DROP COLUMN quarantine_json")
    args = arguments()
    assert main(["held", "--expected-home", args["expected_home"]]) == 0
    assert json.loads(capsys.readouterr().out)["delivery_state"] == "pending"
    with sqlite3.connect(get_hermes_home() / "state.db") as conn:
        assert "quarantine_json" not in {r[1] for r in conn.execute("PRAGMA table_info(async_delegations)")}
    assert main(["held", "--expected-home", args["expected_home"], "--apply", "--expected-sha256",
                 args["expected_sha256"], "--authorization-ref", args["authorization_ref"]]) == 0
    applied = json.loads(capsys.readouterr().out)
    assert applied == inspect_quarantine("held", expected_home=args["expected_home"])
    assert applied["audit"]["preimage_sha256"] == args["expected_sha256"]


def test_persisted_guards_block_old_writer_shapes_and_preserve_neighbors():
    seed()
    seed("neighbor")
    quarantine_completion("held", **arguments())
    before = row()
    # A separate connection has neither Python locks nor new-runtime predicates.
    with sqlite3.connect(get_hermes_home() / "state.db") as conn:
        assert conn.execute("UPDATE async_delegations SET delivery_state='delivered'").rowcount == 1
        assert conn.execute("UPDATE async_delegations SET quarantine_json=NULL WHERE delegation_id='held'").rowcount == 0
        with pytest.raises(sqlite3.IntegrityError, match="delegation_quarantined"):
            conn.execute("INSERT OR REPLACE INTO async_delegations "
                         "(delegation_id,origin_session,state,dispatched_at,updated_at) VALUES ('held','','running',1,1)")
        assert conn.execute("DELETE FROM async_delegations").rowcount == 1
    assert row() == before
    assert inspect_quarantine("held", expected_home=str(get_hermes_home()))["audit"]["guard_sha256"]


def test_guard_name_collision_rolls_back_whole_quarantine():
    seed()
    args = arguments()
    before = row()
    with ad._transaction() as conn:
        conn.execute("CREATE TRIGGER async_quarantine_no_delete BEFORE DELETE ON async_delegations BEGIN SELECT 1; END")
    with pytest.raises(ValueError, match="guard_mismatch"):
        quarantine_completion("held", **args)
    assert row() == before
    with ad._transaction() as conn:
        assert conn.execute("SELECT count(*) FROM sqlite_master WHERE name='async_quarantine_no_update'").fetchone()[0] == 0


@pytest.mark.parametrize("claim_first", [True, False])
def test_cross_connection_claim_quarantine_exclusion(monkeypatch, claim_first):
    import threading
    from concurrent.futures import ThreadPoolExecutor
    import tools.async_delegation_quarantine as quarantine

    seed()
    args = arguments()
    entered, release = threading.Event(), threading.Event()
    original = quarantine._validate

    def validate(conn, record):
        original(conn, record)
        entered.set()
        assert release.wait(10)

    def claim():
        with sqlite3.connect(get_hermes_home() / "state.db", timeout=15) as conn:
            conn.execute("BEGIN IMMEDIATE")
            changed = conn.execute("UPDATE async_delegations SET delivery_claim='competitor', delivery_claimed_at=1 "
                                   "WHERE delegation_id='held' AND delivery_state='pending'").rowcount
            if claim_first:
                entered.set()
                assert release.wait(10)
            return changed

    with ThreadPoolExecutor(max_workers=2) as pool:
        if claim_first:
            claimed = pool.submit(claim)
            assert entered.wait(10)
            held = pool.submit(quarantine_completion, "held", **args)
            release.set()
            assert claimed.result(timeout=20) == 1
            with pytest.raises(ValueError, match="unclaimed"):
                held.result(timeout=20)
            assert row()["quarantine_json"] is None
        else:
            monkeypatch.setattr(quarantine, "_validate", validate)
            held = pool.submit(quarantine_completion, "held", **args)
            assert entered.wait(10)
            claimed = pool.submit(claim)
            release.set()
            assert held.result(timeout=20)["delivery_state"] == "quarantined"
            assert claimed.result(timeout=20) == 0
            assert row()["delivery_claim"] is None
