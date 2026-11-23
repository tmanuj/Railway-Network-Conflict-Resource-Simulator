from contextlib import asynccontextmanager
from dataclasses import asdict

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from . import config, db, repository
from .evaluation import Evaluation, Evaluator
from .models import SolveParams
from .optimizer import anneal
from .os_scheduler import os_schedule
from .scenario import make_scenario
from .schedule import Network

state: dict = {"net": None, "error": None}


def load_network():
    stations, tracks, trains = repository.fetch_all()
    state["net"] = Network(stations, tracks, trains)
    state["error"] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        db.connect()
        load_network()
    except Exception as e:  # keep the API up so /api/health can show what is wrong
        state["error"] = f"{type(e).__name__}: {e}"
    yield
    db.close()


app = FastAPI(title="Railway Conflict Solver API", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=config.CORS_ORIGINS, allow_methods=["*"], allow_headers=["*"])


def net() -> Network:
    if state["net"] is None:
        raise HTTPException(503, state["error"] or "Network not loaded")
    return state["net"]


def serialize(ev: Evaluation, holds: dict, n: Network) -> dict:
    return {
        "holds": holds,
        "schedule": [{"train_id": s.train_id, "steps": [asdict(x) for x in s.steps]} for s in ev.schedules],
        "conflicts": [asdict(c) for c in ev.conflicts],
        "metrics": {
            "conflicts": len(ev.conflicts),
            "solver_delay_min": round(sum(ev.delays.values()), 2),
            "weighted_delay": round(ev.weighted_delay, 2),
            "fitness": round(ev.fitness, 2),
            "delay_by_train": {k: round(v, 2) for k, v in ev.delays.items() if v > 0},
        },
        "disturbance": n.delays,
    }


@app.get("/api/health")
def health():
    n = state["net"]
    return {"ok": n is not None, "error": state["error"],
            "scope": config.SCOPE_STATIONS or f"{config.SCOPE_FROM} -> {config.SCOPE_TO}",
            "counts": n and {"stations": len(n.stations), "tracks": len(n.tracks), "trains": len(n.trains),
                             "single_track": sum(t.single_track for t in n.tracks.values())}}


@app.post("/api/reload")
def reload():
    try:
        load_network()
    except Exception as e:
        state["net"], state["error"] = None, f"{type(e).__name__}: {e}"
        raise HTTPException(500, state["error"])
    return health()


@app.get("/api/network")
def get_network():
    n = net()
    return {"stations": [s.model_dump() for s in n.stations.values()],
            "tracks": [t.model_dump() for t in n.tracks.values()]}


@app.get("/api/trains")
def get_trains():
    return [{"id": t.id, "name": t.name, "priority": t.priority, "departure_min": t.departure_min,
             "route": t.route} for t in net().trains.values()]


@app.get("/api/baseline")
def baseline(headway_min: float = 2.0, disturb_count: int = 6, disturb_seed: int = 1):
    """Timetable with the disturbance applied and no repairs: this is where the conflicts appear."""
    p = SolveParams(headway_min=headway_min, disturb_count=disturb_count, disturb_seed=disturb_seed)
    n = make_scenario(net(), p)
    return serialize(Evaluator(n, p).evaluate(n.zero_holds()), n.zero_holds(), n)


@app.post("/api/solve/os")
def solve_os(p: SolveParams):
    n = make_scenario(net(), p)
    ev = Evaluator(n, p)
    holds = os_schedule(n, p)
    return {"policy": p.policy, "before": serialize(ev.evaluate(n.zero_holds()), n.zero_holds(), n),
            "after": serialize(ev.evaluate(holds), holds, n)}


@app.post("/api/solve/optimize")
def solve_optimize(p: SolveParams):
    n = make_scenario(net(), p)
    ev = Evaluator(n, p)
    start = os_schedule(n, p) if p.start_from == "os" else n.zero_holds()
    best, best_e, history = anneal(n, ev, start, p)
    return {"start_from": p.start_from, "before": serialize(ev.evaluate(n.zero_holds()), n.zero_holds(), n),
            "after": serialize(best_e, best, n), "history": history}
