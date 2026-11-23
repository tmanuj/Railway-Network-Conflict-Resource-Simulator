
from contextlib import asynccontextmanager
from dataclasses import asdict

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from . import config, db, repository
from .evaluation import Evaluation, Evaluator
from .models import SolveParams
from .metaheuristics import run_algorithm, run_hybrid_optimizer
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
    except Exception as e:
        state["error"] = f"{type(e).__name__}: {e}"
    yield
    db.close()


app = FastAPI(title="Railway Conflict Solver API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=config.CORS_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"]
)


def net() -> Network:
    if state["net"] is None:
        raise HTTPException(503, state["error"] or "Network not loaded")
    return state["net"]


def serialize(ev: Evaluation, holds: dict, n: Network) -> dict:
    conflict_trains: dict[str, dict] = {}
    conflict_groups: dict[tuple, dict] = {}
    for conflict in ev.conflicts:
        for train_id in (conflict.train_a, conflict.train_b):
            summary = conflict_trains.setdefault(train_id, {
                "train_id": train_id,
                "conflict_count": 0,
                "peers": set(),
                "tracks": set(),
                "first_time": conflict.time,
                "kinds": set(),
            })
            peer = conflict.train_b if train_id == conflict.train_a else conflict.train_a
            summary["conflict_count"] += 1
            summary["peers"].add(peer)
            summary["tracks"].add(conflict.track_id)
            summary["first_time"] = min(summary["first_time"], conflict.time)
            summary["kinds"].add(conflict.kind)

        bucket = round(conflict.time, 1)
        group_key = (conflict.track_id, bucket)
        group = conflict_groups.setdefault(group_key, {
            "track_id": conflict.track_id,
            "time": bucket,
            "trains": set(),
            "kind": conflict.kind,
        })
        group["trains"].update((conflict.train_a, conflict.train_b))

    conflict_summaries = []
    for train_id in sorted(conflict_trains):
        summary = conflict_trains[train_id]
        conflict_summaries.append({
            "train_id": train_id,
            "conflict_count": summary["conflict_count"],
            "peers": sorted(summary["peers"]),
            "tracks": sorted(summary["tracks"]),
            "first_time": round(summary["first_time"], 2),
            "kind": "Head-on" if "head_on" in summary["kinds"] and len(summary["kinds"]) == 1 else "Too close",
        })

    conflict_group_summaries = [
        {
            "track_id": group["track_id"],
            "time": group["time"],
            "trains": sorted(group["trains"]),
            "kind": "Head-on" if group["kind"] == "head_on" else "Too close",
        }
        for group in sorted(conflict_groups.values(), key=lambda item: (item["time"], item["track_id"]))
    ]

    conflicting_ids = set(conflict_trains)
    clear_trains = sorted(set(n.trains) - conflicting_ids)

    return {
        "holds": holds,
        "schedule": [
            {
                "train_id": s.train_id,
                "steps": [asdict(x) for x in s.steps]
            }
            for s in ev.schedules
        ],
        "conflicts": [asdict(c) for c in ev.conflicts],
        "metrics": {
            "conflicts": len(ev.conflicts),
            "solver_delay_min": round(sum(ev.delays.values()), 2),
            "weighted_delay": round(ev.weighted_delay, 2),
            "fitness": round(ev.fitness, 2),
            "delay_by_train": {
                k: round(v, 2)
                for k, v in ev.delays.items()
                if v > 0
            },
            "conflict_trains": conflict_summaries,
            "clear_trains": clear_trains,
            "conflict_groups": conflict_group_summaries,
        },
        "disturbance": n.delays,
    }


@app.get("/api/health")
def health():
    n = state["net"]
    return {
        "ok": n is not None,
        "error": state["error"],
        "scope": config.SCOPE_STATIONS or f"{config.SCOPE_FROM} -> {config.SCOPE_TO}",
        "counts": n and {
            "stations": len(n.stations),
            "tracks": len(n.tracks),
            "trains": len(n.trains),
            "single_track": sum(t.single_track for t in n.tracks.values())
        }
    }


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
    return {
        "stations": [s.model_dump() for s in n.stations.values()],
        "tracks": [t.model_dump() for t in n.tracks.values()]
    }


@app.get("/api/trains")
def get_trains():
    return [
        {
            "id": t.id,
            "name": t.name,
            "priority": t.priority,
            "departure_min": t.departure_min,
            "route": t.route
        }
        for t in net().trains.values()
    ]


@app.get("/api/baseline")
def baseline(
    headway_min: float = 2.0,
    disturb_count: int = 6,
    disturb_seed: int = 1
):
    p = SolveParams(
        headway_min=headway_min,
        disturb_count=disturb_count,
        disturb_seed=disturb_seed
    )
    n = make_scenario(net(), p)
    return serialize(
        Evaluator(n, p).evaluate(n.zero_holds()),
        n.zero_holds(),
        n
    )


@app.post("/api/solve/os")
def solve_os(p: SolveParams):
    n = make_scenario(net(), p)
    ev = Evaluator(n, p)
    holds = os_schedule(n, p)

    return {
        "policy": p.policy,
        "before": serialize(ev.evaluate(n.zero_holds()), n.zero_holds(), n),
        "after": serialize(ev.evaluate(holds), holds, n)
    }


@app.post("/api/solve/optimize")
def solve_optimize(p: SolveParams):
    n = make_scenario(net(), p)
    ev = Evaluator(n, p)

    os_holds = os_schedule(n, p)
    os_e = ev.evaluate(os_holds)

    best, best_e, history, method, search_info = run_hybrid_optimizer(
        n,
        ev,
        os_holds,
        p
    )

    return {
        "start_from": p.start_from,
        "before": serialize(ev.evaluate(n.zero_holds()), n.zero_holds(), n),
        "after": serialize(best_e, best, n),
        "history": history,
        "method": method,
        "algorithm_history": search_info["algorithm_history"],
        "algorithm_status": search_info["algorithm_status"],
        "search_info": search_info,
        "os_fitness": round(os_e.fitness, 3),
        "os_conflicts": len(os_e.conflicts),
        "os_weighted_delay": round(os_e.weighted_delay, 3),
        "best_os_policy": search_info["best_os_policy"],
        "best_os_fitness": round(search_info["best_os_fitness"], 3),
        "best_os_conflicts": search_info["best_os_conflicts"],
        "best_os_weighted_delay": round(search_info["best_os_weighted_delay"], 3),
        "optimized_conflicts": len(best_e.conflicts),
        "optimized_weighted_delay": round(best_e.weighted_delay, 3),
        "fitness_reduction": round(max(
            0.0,
            search_info["best_os_fitness"] - best_e.fitness
        ), 3),
        "conflict_reduction": search_info["best_os_conflicts"] - len(best_e.conflicts),
        "weighted_delay_reduction": round(
            search_info["best_os_weighted_delay"] - best_e.weighted_delay,
            3
        )
    }


# Compare all five optimization algorithms
@app.post("/api/solve/compare")
def solve_compare(p: SolveParams):
    n = make_scenario(net(), p)
    ev = Evaluator(n, p)

    start = os_schedule(n, p) if p.start_from == "os" else n.zero_holds()

    algorithms = ["pso", "ga", "aco", "abc", "gurobi"]
    results = {}

    for name in algorithms:
        try:
            best, best_e, history = run_algorithm(
                name, n, ev, start, p
            )

            results[name] = {
                "after": serialize(best_e, best, n),
                "history": history,
                "status": "success"
            }

        except Exception as e:
            results[name] = {
                "status": "error",
                "message": f"{type(e).__name__}: {e}"
            }

    return {
        "before": serialize(
            ev.evaluate(n.zero_holds()),
            n.zero_holds(),
            n
        ),
        "results": results
    }
