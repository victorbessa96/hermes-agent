"""Tests for the store-level CAS fire claim (Phase 4C).

`claim_job_for_fire` gives multi-machine at-most-once semantics when an external
scheduler (Chronos) fires a job: across N gateway replicas, exactly ONE wins the
claim for a given fire. Single-machine deployments always win (unaffected).

These exercise the real store against a temp HERMES_HOME (no mocks) per the
E2E-over-mocks discipline for file-touching code.
"""
import threading
import time

import pytest


@pytest.fixture
def temp_home(tmp_path, monkeypatch):
    """Isolated HERMES_HOME so jobs.json doesn't touch the real store."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    # cron.jobs caches no home at import; get_hermes_home() reads the env live.
    yield tmp_path


def test_claim_succeeds_once_then_blocks(temp_home):
    """First claim for a fire wins; a second claim for the same fire loses, and
    next_run_at is advanced (a re-delivery for the old time can't re-fire)."""
    from cron.jobs import create_job, claim_job_for_fire, get_job

    job = create_job(prompt="x", schedule="every 5m", name="t")
    jid = job["id"]
    before = get_job(jid)["next_run_at"]

    assert claim_job_for_fire(jid) is True
    assert claim_job_for_fire(jid) is False
    assert get_job(jid)["next_run_at"] != before


def test_claim_oneshot_cannot_be_double_claimed(temp_home):
    """A one-shot can't be double-claimed (the fresh claim blocks the retry)."""
    from cron.jobs import create_job, claim_job_for_fire

    job = create_job(prompt="x", schedule="in 30m", name="o")
    assert claim_job_for_fire(job["id"]) is True
    assert claim_job_for_fire(job["id"]) is False


def test_claim_unknown_job_returns_false(temp_home):
    from cron.jobs import claim_job_for_fire

    assert claim_job_for_fire("nope-does-not-exist") is False


def test_claim_paused_job_returns_false(temp_home):
    """A paused job can't be claimed."""
    from cron.jobs import create_job, claim_job_for_fire, pause_job

    job = create_job(prompt="x", schedule="every 5m", name="p")
    pause_job(job["id"])
    assert claim_job_for_fire(job["id"]) is False


def test_forced_claim_atomically_resumes_paused_job(temp_home):
    """Explicit manual fire may resume a paused job without exposing a due
    intermediate state to the ticker."""
    from cron.jobs import create_job, claim_job_for_fire, get_job, pause_job

    job = create_job(prompt="x", schedule="every 5m", name="manual")
    pause_job(job["id"])

    assert claim_job_for_fire(job["id"], force=True) is True
    claimed = get_job(job["id"])
    assert claimed["enabled"] is True
    assert claimed["state"] == "scheduled"
    assert claimed["paused_at"] is None
    assert claimed["paused_reason"] is None
    assert claimed["fire_claim"] is not None


def test_stale_claim_is_reclaimable(temp_home, monkeypatch):
    """A claim older than the TTL is overwritten — the fire isn't stuck forever
    if the winning machine crashed before mark_job_run cleared the claim."""
    from cron.jobs import create_job, claim_job_for_fire

    job = create_job(prompt="x", schedule="every 5m", name="s")
    jid = job["id"]
    assert claim_job_for_fire(jid) is True
    # With a 0s TTL, the existing claim is always considered stale.
    assert claim_job_for_fire(jid, claim_ttl_seconds=0) is True


def test_mark_job_run_clears_claim(temp_home):
    """After a recurring job completes, its claim is cleared so the next fire
    can be claimed again."""
    from cron.jobs import create_job, claim_job_for_fire, mark_job_run, get_job

    job = create_job(prompt="x", schedule="every 5m", name="c")
    jid = job["id"]
    assert claim_job_for_fire(jid) is True
    assert get_job(jid).get("fire_claim") is not None

    mark_job_run(jid, success=True)
    assert get_job(jid).get("fire_claim") is None
    # …and the re-armed recurring job is claimable again.
    assert claim_job_for_fire(jid) is True


def test_fire_claim_heartbeat_refreshes_only_expected_owner(temp_home, monkeypatch):
    from datetime import datetime, timedelta

    import cron.jobs as jobs

    job = jobs.create_job(prompt="x", schedule="every 5m", name="heartbeat")
    assert jobs.claim_job_for_fire(job["id"]) is True
    claimed = jobs.get_job(job["id"])["fire_claim"]
    claimed_at = datetime.fromisoformat(claimed["at"])
    monkeypatch.setattr(
        jobs,
        "_hermes_now",
        lambda: claimed_at + timedelta(seconds=30),
    )

    assert jobs.heartbeat_fire_claim(
        job["id"],
        expected_owner=claimed["by"],
    ) is True
    refreshed = jobs.get_job(job["id"])["fire_claim"]
    assert refreshed["at"] != claimed["at"]
    assert refreshed["by"] == claimed["by"]
    assert jobs.heartbeat_fire_claim(
        job["id"],
        expected_owner="replacement-owner",
    ) is False


def test_reclaimed_fire_uses_new_owner_token(temp_home, monkeypatch):
    from datetime import datetime, timedelta

    import cron.jobs as jobs

    job = jobs.create_job(prompt="x", schedule="every 5m", name="reclaim")
    assert jobs.claim_job_for_fire(job["id"]) is True
    original = dict(jobs.get_job(job["id"])["fire_claim"])
    original_at = datetime.fromisoformat(original["at"])
    monkeypatch.setattr(
        jobs,
        "_hermes_now",
        # Period-aware TTL for every-5m is 600s (max(300, 2*300)); step past it so
        # the claim is stale on TTL alone, then report the owner pid dead so the
        # probe's grace window doesn't defer reclaim (the crashed-worker case).
        lambda: original_at + timedelta(seconds=601),
    )
    monkeypatch.setattr(jobs, "_pid_alive", lambda *a, **k: False)

    assert jobs.claim_job_for_fire(job["id"]) is True
    replacement = dict(jobs.get_job(job["id"])["fire_claim"])
    assert replacement["by"] != original["by"]
    assert jobs.heartbeat_fire_claim(
        job["id"],
        expected_owner=original["by"],
    ) is False
    assert jobs.get_job(job["id"])["fire_claim"] == replacement


def test_stale_fire_owner_cannot_mark_replacement_run(temp_home):
    import cron.jobs as jobs

    job = jobs.create_job(prompt="x", schedule="every 5m", name="fenced")
    assert jobs.claim_job_for_fire(job["id"]) is True
    original = dict(jobs.get_job(job["id"])["fire_claim"])
    records = jobs.load_jobs()
    records[0]["fire_claim"] = {"at": original["at"], "by": "replacement"}
    jobs.save_jobs(records)

    assert jobs.mark_job_run(
        job["id"],
        success=True,
        expected_fire_owner=original["by"],
    ) is False
    persisted = jobs.get_job(job["id"])
    assert persisted["fire_claim"]["by"] == "replacement"
    assert persisted.get("last_run_at") is None


def test_fire_claim_fence_serializes_terminal_revocation(temp_home):
    """A side effect authorized by owner linearizes before terminal revocation."""
    from cron.jobs import (
        claim_job_for_fire,
        create_job,
        fire_claim_fence,
        mark_job_run,
    )

    job = create_job(prompt="x", schedule="every 5m", name="fenced-side-effect")
    claimed = claim_job_for_fire(job["id"], return_job=True)
    assert isinstance(claimed, dict)
    owner = claimed["fire_claim"]["by"]
    terminal_done = threading.Event()

    def finish_run():
        mark_job_run(job["id"], True, expected_fire_owner=owner)
        terminal_done.set()

    with fire_claim_fence(job["id"], expected_owner=owner) as owns_claim:
        assert owns_claim is True
        thread = threading.Thread(target=finish_run)
        thread.start()
        time.sleep(0.05)
        assert terminal_done.is_set() is False

    thread.join(timeout=1)
    assert terminal_done.is_set() is True


def test_fire_claim_fence_rejects_stale_owner(temp_home):
    from cron.jobs import claim_job_for_fire, create_job, fire_claim_fence

    job = create_job(prompt="x", schedule="every 5m", name="stale-fence")
    claim_job_for_fire(job["id"])

    with fire_claim_fence(job["id"], expected_owner="stale") as owns_claim:
        assert owns_claim is False


def test_same_process_fire_fence_refuses_second_claim_after_timeout(temp_home, monkeypatch):
    """A wedged local holder must not indefinitely block another claimant."""
    import cron.jobs as jobs

    job = jobs.create_job(prompt="x", schedule="every 5m", name="local-fence-timeout")
    monkeypatch.setattr(jobs, "_JOBS_LOCK_TIMEOUT_SECONDS", 0.1)
    completed = threading.Event()
    result = {}

    def second_claimant():
        result["claimed"] = jobs.claim_job_for_fire(job["id"])
        completed.set()

    with jobs._fire_job_lock(job["id"]) as acquired:
        assert acquired is True
        thread = threading.Thread(target=second_claimant)
        thread.start()
        assert completed.wait(timeout=2), "same-process claimant waited past the fire-fence timeout"
        assert result["claimed"] is False

    thread.join(timeout=2)
    assert thread.is_alive() is False
    assert jobs.claim_job_for_fire(job["id"]) is True


def test_same_thread_fire_fence_reentrancy_preserves_ownership(temp_home):
    """Nested same-thread callers retain the existing fire fence."""
    import cron.jobs as jobs

    job = jobs.create_job(prompt="x", schedule="every 5m", name="local-fence-reentrant")
    completed = threading.Event()
    result = {}

    def reentrant_claimant():
        with jobs._fire_job_lock(job["id"]) as outer_acquired:
            result["outer"] = outer_acquired
            with jobs._fire_job_lock(job["id"]) as inner_acquired:
                result["inner"] = inner_acquired
        completed.set()

    thread = threading.Thread(target=reentrant_claimant, daemon=True)
    thread.start()
    assert completed.wait(timeout=2), "same-thread nested fire fence did not return"
    assert result == {"outer": True, "inner": True}
    thread.join(timeout=2)
    assert thread.is_alive() is False


def test_expired_claim_with_live_owner_pid_defers_reclaim(temp_home, monkeypatch):
    """A claim aged past the base 300s TTL whose owner pid is still alive on this
    host is NOT reclaimed yet (bounded grace up to 2x TTL). This protects a
    slow-but-alive run from being double-fired when claim_ttl equals the job
    period (the */5 stall case).

    With the period-aware TTL, a 301s-old claim on an every-5m job is still inside
    the plain TTL (max(300, 2*300) = 600) — no pid probe needed. The pid probe
    then guards the 600s→1200s window. Test the same shape a longer-period job
    exercises: 301s is past the base TTL, before the period-aware TTL binds."""
    from datetime import datetime, timedelta

    import cron.jobs as jobs

    job = jobs.create_job(prompt="x", schedule="every 5m", name="live-owner")
    assert jobs.claim_job_for_fire(job["id"]) is True
    original = dict(jobs.get_job(job["id"])["fire_claim"])
    original_at = datetime.fromisoformat(original["at"])
    monkeypatch.setattr(
        jobs,
        "_hermes_now",
        lambda: original_at + timedelta(seconds=301),  # past 300s base TTL
    )
    # Owner pid (= this test process) is genuinely alive; no monkeypatch needed.

    # Reclaim must be refused while the owner is alive within the grace window.
    assert jobs.claim_job_for_fire(job["id"]) is False
    assert jobs.get_job(job["id"])["fire_claim"]["by"] == original["by"]


def test_expired_claim_with_dead_owner_pid_is_reclaimed(temp_home, monkeypatch):
    """The crash-reclaim behavior must be preserved: past the period-aware TTL plus
    a dead owner pid still reclaims exactly as before (no wedge from a crashed
    worker). For every-5m the TTL is 600s (max(300, 2*300)), so 601s is stale."""
    from datetime import datetime, timedelta

    import cron.jobs as jobs

    job = jobs.create_job(prompt="x", schedule="every 5m", name="dead-owner")
    assert jobs.claim_job_for_fire(job["id"]) is True
    original = dict(jobs.get_job(job["id"])["fire_claim"])
    original_at = datetime.fromisoformat(original["at"])
    monkeypatch.setattr(
        jobs,
        "_hermes_now",
        lambda: original_at + timedelta(seconds=601),
    )
    monkeypatch.setattr(jobs, "_pid_alive", lambda *a, **k: False)

    assert jobs.claim_job_for_fire(job["id"]) is True
    assert jobs.get_job(job["id"])["fire_claim"]["by"] != original["by"]


def test_live_owner_pid_past_grace_window_reclaims(temp_home, monkeypatch):
    """Even a live owner cannot hold past 2x the period-aware TTL: the grace window
    is bounded so a wedged (zombie-with-live-pid, heartbeat-starved) worker
    eventually loses. For every-5m, TTL = 600s and grace = 2 * 600 = 1200s."""
    from datetime import datetime, timedelta

    import cron.jobs as jobs

    job = jobs.create_job(prompt="x", schedule="every 5m", name="grace-expired")
    assert jobs.claim_job_for_fire(job["id"]) is True
    original_at = datetime.fromisoformat(jobs.get_job(job["id"])["fire_claim"]["at"])
    monkeypatch.setattr(
        jobs,
        "_hermes_now",
        lambda: original_at + timedelta(seconds=1201),  # past 2x 600s TTL
    )
    # Owner alive, but past the grace bound — reclaim proceeds.
    assert jobs.claim_job_for_fire(job["id"]) is True


def test_pid_probe_helper_unit_behaviour():
    """The pid probe itself: live self-pid, impossible pid, garbage owner string."""
    import os
    from datetime import datetime, timedelta, timezone

    import cron.jobs as jobs

    now = datetime.now(timezone.utc)
    at = (now - timedelta(seconds=400)).isoformat()

    # Live same-host pid within grace -> live
    live_by = f"{jobs._machine_id()}:deadbeef"
    assert jobs._claim_is_live_with_pid_probe({"at": at, "by": live_by}, now, 300) is True

    # Garbage by-strings fall through to TTL-only -> stale
    for bad in ("no-colons", "host:notapid:uuid", "", 12345, None):
        assert jobs._claim_is_live_with_pid_probe({"at": at, "by": bad}, now, 300) is False

    # A pid that cannot exist fails liveness; our own pid lives.
    assert jobs._pid_alive(2**22) is False
    assert jobs._pid_alive(os.getpid()) is True


def test_period_aware_ttl_short_period_uses_two_x_period():
    """``_fire_claim_ttl_for_job`` shapes per schedule: */5 → 600; every 15m →
    floor 300 (2*900 > 300, so 1800); 3am daily → 2 * 86400; unknown → floor."""
    import cron.jobs as jobs

    assert jobs._fire_claim_ttl_for_job({"kind": "interval", "minutes": 5}) == 600
    assert jobs._fire_claim_ttl_for_job({"kind": "interval", "minutes": 15}) == 1800
    assert jobs._fire_claim_ttl_for_job({"kind": "interval", "minutes": 1}) == 300
    # Unknown cadence falls back to the flat floor.
    assert jobs._fire_claim_ttl_for_job({"kind": "once", "run_at": "2026-09-12T03:00:00"}) == 300
    assert jobs._fire_claim_ttl_for_job(None) == 300


def test_short_period_job_stale_at_past_base_ttl_still_blocks_reclaim(temp_home, monkeypatch):
    """The */5 stall race the patch closes: claim at T=0, clock at T=301s. Without
    the period-aware TTL (plain 300s) a crashed pid would reclaim; with it, the
    TTL is 600s so the claim is merely "within the pid-probe grace window" and
    only a dead-pid probe should allow reclaim.

    Live pid + 301s => reclaim refused (TTL not yet expired)."""
    from datetime import datetime, timedelta

    import cron.jobs as jobs

    job = jobs.create_job(prompt="x", schedule="every 5m", name="stall-shield")
    assert jobs.claim_job_for_fire(job["id"]) is True
    original_at = datetime.fromisoformat(jobs.get_job(job["id"])["fire_claim"]["at"])
    monkeypatch.setattr(
        jobs,
        "_hermes_now",
        lambda: original_at + timedelta(seconds=301),
    )
    # Live pid (this test process). The reclaim must be refused.
    assert jobs.claim_job_for_fire(job["id"]) is False
