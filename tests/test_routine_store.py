"""Routine persistence: survives reloads, bounded history, crash recovery."""
from __future__ import annotations

import json

import pytest

import routine_store as store


@pytest.fixture
def d(tmp_path):
    store.load(tmp_path)
    yield tmp_path
    store.load(tmp_path / "gone")


def make(**kw):
    base = {"name": "AI news", "prompt": "Summarize", "query": None, "timezone": "UTC",
            "schedule": {"kind": "interval", "hours": 1},
            "sources": {"web": True, "library": False}, "enabled": True,
            "next_run_at": "2026-11-25T13:00:00+00:00"}
    return store.create({**base, **kw})


def test_round_trip(d):
    r = make()
    run = store.start_run(r["id"], "manual")
    store.finish_run(r["id"], run["id"], {"status": "ok", "content": "hi", "sources": []})
    store.load(d)
    got = store.get(r["id"])
    assert got["name"] == "AI news"
    assert got["runs"][0]["status"] == "ok" and got["runs"][0]["read"] is False


def test_returned_objects_are_copies(d):
    r = make()
    r["name"] = "changed"
    store.get(r["id"])["runs"].append({"x": 1})
    assert store.get(r["id"])["name"] == "AI news" and store.get(r["id"])["runs"] == []


def test_history_is_bounded_newest_first(d):
    r = make()
    for i in range(store.MAX_RUNS + 5):
        run = store.start_run(r["id"], "manual")
        store.finish_run(r["id"], run["id"], {"status": "ok", "content": str(i)})
    runs = store.get(r["id"])["runs"]
    assert len(runs) == store.MAX_RUNS and runs[0]["content"] == str(store.MAX_RUNS + 4)


def test_recover_marks_running_as_interrupted(d):
    r = make()
    store.start_run(r["id"], "schedule")
    store.load(d)
    assert store.recover() == 1
    run = store.get(r["id"])["runs"][0]
    assert run["status"] == "interrupted" and run["finished_at"]


def test_finish_after_delete_is_a_noop(d):
    r = make()
    run = store.start_run(r["id"], "manual")
    assert store.delete(r["id"]) is True
    store.finish_run(r["id"], run["id"], {"status": "ok"})
    assert store.get(r["id"]) is None and not list(d.glob("*.json"))
    assert store.delete(r["id"]) is False


def test_corrupt_file_is_skipped_and_kept(d):
    (d / "bad.json").write_text("{not json")
    (d / "list.json").write_text("[]")
    r = make()
    store.load(d)
    assert [x["id"] for x in store.list_all()] == [r["id"]]
    assert (d / "bad.json").read_text() == "{not json"


def test_unread_and_mark_read(d):
    a, b = make(name="a"), make(name="b")
    for r in (a, b):
        run = store.start_run(r["id"], "manual")
        assert store.unread_count() == (0 if r is a else 1)   # running is not unread
        store.finish_run(r["id"], run["id"], {"status": "ok", "content": r["name"]})
    assert store.unread_count() == 2
    inbox = store.inbox(10)
    assert {i["routine_name"] for i in inbox} == {"a", "b"}
    assert store.mark_read(a["id"], inbox[[i["routine_id"] for i in inbox].index(a["id"])]["id"]) is True
    assert store.mark_read(a["id"], "nope") is False
    assert store.unread_count() == 1
    assert store.mark_all_read() == 1 and store.unread_count() == 0


def test_writes_are_atomic_and_private(d):
    r = make()
    f = d / f"{r['id']}.json"
    assert json.loads(f.read_text())["id"] == r["id"]
    assert not list(d.glob("*.tmp"))
    assert f.stat().st_mode & 0o077 == 0


def test_failed_write_leaves_memory_unchanged(d, monkeypatch):
    r = make()

    def boom(*_a, **_k):
        raise OSError("disk full")
    monkeypatch.setattr(store.os, "replace", boom)
    with pytest.raises(OSError):
        store.update(r["id"], {"name": "new"})
    assert store.get(r["id"])["name"] == "AI news"
