"""Tests the HTTP layer with a fake Neo4j.   python -m tests.test_api"""
from fastapi.testclient import TestClient
from app import db, main, repository
from app.demo_data import STATIONS, TRACKS, TRAINS

db.connect = lambda: None
db.close = lambda: None
repository.fetch_all = lambda: (STATIONS, TRACKS, TRAINS)

with TestClient(main.app) as c:
    print("health  ", c.get("/api/health").json())
    print("network ", len(c.get("/api/network").json()["tracks"]), "tracks")
    b = c.get("/api/baseline").json()["metrics"]
    print("baseline", b["conflicts"], "conflicts")
    o = c.post("/api/solve/os", json={"policy": "priority"}).json()["after"]["metrics"]
    print("os      ", o["conflicts"], "conflicts, fitness", o["fitness"])
    r = c.post("/api/solve/optimize", json={
        "start_from": "os",
        "iterations": 3000,
        "disturb_count": 3,
        "disturb_seed": 1,
    }).json()
    m = r["after"]["metrics"]
    print("optimize", m["conflicts"], "conflicts, fitness", m["fitness"], "history pts", len(r["history"]))
    assert m["fitness"] <= r["best_os_fitness"] + 0.001
    assert m["fitness"] < r["best_os_fitness"]
    assert set(r["algorithm_status"]) == {
        "simulated_annealing",
        "pso",
        "ga",
        "constraint_programming",
    }
    assert all(
        status["status"] == "completed"
        for status in r["algorithm_status"].values()
    )
    assert r["history"]
