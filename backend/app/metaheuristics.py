
"""Optimization algorithms for Railway Network Conflict Resource Simulator.

Algorithms:
1. PSO - Particle Swarm Optimization
2. GA - Genetic Algorithm
3. ACO - Ant Colony Optimization
4. ABC - Artificial Bee Colony
5. CP-SAT - Constraint Programming
6. Gurobi - Exact MILP Solver (optional)
"""

import copy
import math
import random
from typing import Callable

from .evaluation import Evaluator
from .models import SolveParams
from .optimizer import anneal
from .schedule import Network, simulate


# --------------------------------------------------
# COMMON HELPER FUNCTIONS
# --------------------------------------------------

def _layout(net: Network):
    return [
        (train.id, i)
        for train in net.trains.values()
        for i in range(len(train.route) - 1)
    ]


def _decode(keys, vector):
    holds = {}

    for (train_id, segment), value in zip(keys, vector):
        holds.setdefault(train_id, [])

        while len(holds[train_id]) <= segment:
            holds[train_id].append(0.0)

        holds[train_id][segment] = round(
            max(0.0, float(value)) * 2
        ) / 2

    return holds


def _start_vector(keys, start):
    return [
        float(start.get(train_id, [])[segment])
        if segment < len(start.get(train_id, []))
        else 0.0
        for train_id, segment in keys
    ]


def _score(net, ev, keys, vector, p):
    vector = [
        min(p.max_hold_min, max(0.0, float(x)))
        for x in vector
    ]

    result = ev.evaluate(_decode(keys, vector))
    return result.fitness, result


def _initial(keys, p, rnd):
    return [
        rnd.uniform(0, min(2.0, p.max_hold_min))
        for _ in keys
    ]


def _add_history(history, iteration, best, current, result):
    history.append({
        "iter": iteration,
        "best": round(best, 3),
        "current": round(current, 3),
        "conflicts": len(result.conflicts)
    })


def _history_interval(p):
    return max(1, p.iterations // 150)


# --------------------------------------------------
# 1. PARTICLE SWARM OPTIMIZATION (PSO)
# --------------------------------------------------

def pso(net: Network, ev: Evaluator, start, p: SolveParams):

    rnd = random.Random(p.seed)
    keys = _layout(net)

    if not keys:
        raise ValueError("No train segments available")

    count = min(24, max(8, len(keys) // 2))

    particles = [
        _initial(keys, p, rnd)
        for _ in range(count)
    ]

    particles[0] = _start_vector(keys, start)

    velocities = [
        [0.0] * len(keys)
        for _ in range(count)
    ]

    personal = [x[:] for x in particles]

    personal_scores = [
        _score(net, ev, keys, x, p)[0]
        for x in particles
    ]

    best_index = min(
        range(count),
        key=lambda i: personal_scores[i]
    )

    global_best = personal[best_index][:]
    best_score, best_eval = _score(
        net, ev, keys, global_best, p
    )

    history = []

    for iteration in range(p.iterations):

        for j in range(count):

            for k in range(len(keys)):

                r1 = rnd.random()
                r2 = rnd.random()

                velocities[j][k] = (
                    0.65 * velocities[j][k]
                    + 1.4 * r1 * (
                        personal[j][k] - particles[j][k]
                    )
                    + 1.6 * r2 * (
                        global_best[k] - particles[j][k]
                    )
                )

                particles[j][k] = min(
                    p.max_hold_min,
                    max(
                        0.0,
                        particles[j][k] + velocities[j][k]
                    )
                )

            score, result = _score(
                net, ev, keys, particles[j], p
            )

            if score < personal_scores[j]:
                personal[j] = particles[j][:]
                personal_scores[j] = score

            if score < best_score:
                global_best = particles[j][:]
                best_score = score
                best_eval = result

        if (
            iteration % _history_interval(p) == 0
            or iteration == p.iterations - 1
        ):
            _add_history(
                history, iteration,
                best_score, best_score, best_eval
            )

    return _decode(keys, global_best), best_eval, history


# --------------------------------------------------
# 2. GENETIC ALGORITHM (GA)
# --------------------------------------------------

def ga(net: Network, ev: Evaluator, start, p: SolveParams):

    rnd = random.Random(p.seed)
    keys = _layout(net)

    if not keys:
        raise ValueError("No train segments available")

    size = 30

    population = [
        _initial(keys, p, rnd)
        for _ in range(size)
    ]

    population[0] = _start_vector(keys, start)

    best_vector = population[0][:]
    best_score, best_eval = _score(
        net, ev, keys, best_vector, p
    )

    history = []

    for iteration in range(p.iterations):

        ranked = sorted(
            [
                (_score(net, ev, keys, x, p)[0], x)
                for x in population
            ],
            key=lambda item: item[0]
        )

        if ranked[0][0] < best_score:
            best_score = ranked[0][0]
            best_vector = ranked[0][1][:]
            best_eval = _score(
                net, ev, keys, best_vector, p
            )[1]

        new_population = [
            ranked[0][1][:],
            ranked[1][1][:]
        ]

        while len(new_population) < size:

            def tournament():
                group = rnd.sample(
                    ranked[:max(4, size // 2)], 3
                )
                return min(group, key=lambda x: x[0])[1]

            parent1 = tournament()
            parent2 = tournament()

            cut = rnd.randrange(len(keys))

            child = (
                parent1[:cut] + parent2[cut:]
            )

            for k in range(len(child)):

                if rnd.random() < 0.12:
                    child[k] += rnd.gauss(0, 3)

                child[k] = min(
                    p.max_hold_min,
                    max(0.0, child[k])
                )

            new_population.append(child)

        population = new_population

        if (
            iteration % _history_interval(p) == 0
            or iteration == p.iterations - 1
        ):
            _add_history(
                history,
                iteration,
                best_score,
                ranked[0][0],
                best_eval
            )

    return _decode(keys, best_vector), best_eval, history


# --------------------------------------------------
# 3. ANT COLONY OPTIMIZATION (ACO)
# --------------------------------------------------

def aco(net: Network, ev: Evaluator, start, p: SolveParams):

    rnd = random.Random(p.seed)
    keys = _layout(net)

    if not keys:
        raise ValueError("No train segments available")

    choices = [
        i * 0.5
        for i in range(int(p.max_hold_min * 2) + 1)
    ]

    pheromone = [
        [1.0 for _ in choices]
        for _ in keys
    ]

    best_vector = _start_vector(keys, start)

    best_score, best_eval = _score(
        net, ev, keys, best_vector, p
    )

    history = []
    ants = min(16, max(6, len(keys)))

    for iteration in range(p.iterations):

        iteration_best = None

        for _ in range(ants):

            vector = []

            for row in pheromone:
                weights = [
                    max(0.001, value)
                    for value in row
                ]

                selected = rnd.choices(
                    choices,
                    weights=weights,
                    k=1
                )[0]

                vector.append(selected)

            score, result = _score(
                net, ev, keys, vector, p
            )

            if (
                iteration_best is None
                or score < iteration_best[0]
            ):
                iteration_best = (
                    score, vector, result
                )

            if score < best_score:
                best_score = score
                best_vector = vector[:]
                best_eval = result

        # Evaporation
        pheromone = [
            [
                max(0.05, value * 0.94)
                for value in row
            ]
            for row in pheromone
        ]

        score, vector, result = iteration_best

        deposit = 1.0 / max(1.0, score)

        for k, value in enumerate(vector):

            index = min(
                range(len(choices)),
                key=lambda j: abs(choices[j] - value)
            )

            pheromone[k][index] += deposit * 100.0

        if (
            iteration % _history_interval(p) == 0
            or iteration == p.iterations - 1
        ):
            _add_history(
                history,
                iteration,
                best_score,
                score,
                best_eval
            )

    return _decode(keys, best_vector), best_eval, history


# --------------------------------------------------
# 4. ARTIFICIAL BEE COLONY (ABC)
# --------------------------------------------------

def abc(net: Network, ev: Evaluator, start, p: SolveParams):

    rnd = random.Random(p.seed)
    keys = _layout(net)

    if not keys:
        raise ValueError("No train segments available")

    count = 20
    limit = max(10, len(keys) * 3)

    food = [
        _initial(keys, p, rnd)
        for _ in range(count)
    ]

    food[0] = _start_vector(keys, start)

    scores = [
        _score(net, ev, keys, x, p)[0]
        for x in food
    ]

    trials = [0] * count

    best_index = min(
        range(count),
        key=lambda i: scores[i]
    )

    best_vector = food[best_index][:]
    best_score = scores[best_index]

    best_eval = _score(
        net, ev, keys, best_vector, p
    )[1]

    history = []

    for iteration in range(p.iterations):

        # Employed bees phase
        for i in range(count):

            k = rnd.randrange(count - 1)

            if k >= i:
                k += 1

            j = rnd.randrange(len(keys))

            candidate = food[i][:]

            candidate[j] += (
                rnd.uniform(-1, 1)
                * (candidate[j] - food[k][j])
            )

            candidate[j] = min(
                p.max_hold_min,
                max(0.0, candidate[j])
            )

            score, _ = _score(
                net, ev, keys, candidate, p
            )

            if score < scores[i]:
                food[i] = candidate
                scores[i] = score
                trials[i] = 0
            else:
                trials[i] += 1

        # Onlooker bees phase
        fitness = [
            1.0 / (1.0 + max(0.0, score))
            for score in scores
        ]

        for _ in range(count):

            i = rnd.choices(
                range(count),
                weights=fitness,
                k=1
            )[0]

            k = rnd.randrange(count - 1)

            if k >= i:
                k += 1

            j = rnd.randrange(len(keys))

            candidate = food[i][:]

            candidate[j] += (
                rnd.uniform(-1, 1)
                * (candidate[j] - food[k][j])
            )

            candidate[j] = min(
                p.max_hold_min,
                max(0.0, candidate[j])
            )

            score, _ = _score(
                net, ev, keys, candidate, p
            )

            if score < scores[i]:
                food[i] = candidate
                scores[i] = score
                trials[i] = 0
            else:
                trials[i] += 1

        # Scout bees phase
        for i in range(count):

            if trials[i] >= limit:
                food[i] = _initial(keys, p, rnd)

                scores[i] = _score(
                    net, ev, keys, food[i], p
                )[0]

                trials[i] = 0

        best_index = min(
            range(count),
            key=lambda i: scores[i]
        )

        if scores[best_index] < best_score:

            best_vector = food[best_index][:]
            best_score = scores[best_index]

            best_eval = _score(
                net, ev, keys, best_vector, p
            )[1]

        if (
            iteration % _history_interval(p) == 0
            or iteration == p.iterations - 1
        ):
            _add_history(
                history,
                iteration,
                best_score,
                scores[best_index],
                best_eval
            )

    return _decode(keys, best_vector), best_eval, history


# --------------------------------------------------
# 5. CONSTRAINT PROGRAMMING
# --------------------------------------------------

# --------------------------------------------------
# CP-SAT resource allocation
# --------------------------------------------------

def constraint_programming(
    net: Network,
    ev: Evaluator,
    start,
    p: SolveParams
):
    """Solve the bounded hold-allocation model with OR-Tools CP-SAT."""
    try:
        from ortools.sat.python import cp_model
    except ImportError as exc:
        raise RuntimeError(
            "Constraint programming requires OR-Tools. Install backend requirements."
        ) from exc

    keys = _layout(net)
    if not keys:
        raise ValueError("No train segments available")

    scale = 100
    model = cp_model.CpModel()
    start_values = dict(zip(keys, _start_vector(keys, start)))
    max_hold_units = {
        key: max(
            0,
            int(math.floor(p.max_hold_min * 2 + 1e-9)),
            int(math.ceil(max(0.0, start_values[key]) * 2 - 1e-9)),
        )
        for key in keys
    }
    hold_vars = {
        key: model.new_int_var(0, max_hold_units[key], f"hold_{key[0]}_{key[1]}")
        for key in keys
    }

    for key, value in start_values.items():
        model.add_hint(
            hold_vars[key],
            min(max_hold_units[key], max(0, int(round(value * 2))))
        )

    occupancies = {}
    for schedule in simulate(net):
        train = net.trains[schedule.train_id]
        for segment, step in enumerate(schedule.steps):
            track = net.tracks[step.track_id]
            resource = (
                step.track_id
                if track.single_track
                else f"{step.track_id}:{step.direction}"
            )
            previous_holds = [
                hold_vars[(train.id, index)]
                for index in range(segment + 1)
            ]
            start_expr = (
                math.floor(step.enter * scale + 1e-8)
                + scale // 2 * sum(previous_holds)
            )
            start_var = model.new_int_var(
                math.floor(step.enter * scale + 1e-8),
                math.floor(step.enter * scale + 1e-8)
                + (scale // 2) * sum(
                    max_hold_units[(train.id, index)]
                    for index in range(segment + 1)
                ),
                f"entry_{train.id}_{segment}"
            )
            model.add(start_var == start_expr)
            duration = math.ceil(
                (step.exit - step.enter + p.headway_min) * scale - 1e-9
            )
            interval = model.new_interval_var(
                start_var,
                duration,
                start_var + duration,
                f"occupancy_{train.id}_{segment}"
            )
            occupancies.setdefault(resource, []).append(interval)

    for intervals in occupancies.values():
        if len(intervals) > 1:
            model.add_no_overlap(intervals)

    model.minimize(sum(
        net.trains[train_id].priority * hold_vars[(train_id, segment)]
        for train_id, segment in keys
    ))

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 1.0
    solver.parameters.num_search_workers = 1
    solver.parameters.random_seed = p.seed
    status = solver.solve(model)
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        raise RuntimeError(
            f"CP-SAT did not find a feasible hold allocation: {solver.status_name(status)}"
        )

    holds = _decode(
        keys,
        [solver.value(hold_vars[key]) / 2 for key in keys]
    )
    result = ev.evaluate(holds)
    if result.conflicts:
        raise RuntimeError(
            "CP-SAT candidate failed exact conflict validation"
        )

    history = [{
        "iter": int(solver.num_branches),
        "best": round(result.fitness, 3),
        "current": round(result.fitness, 3),
        "conflicts": 0,
    }]
    return holds, result, history

def gurobi(net: Network, ev: Evaluator, start, p: SolveParams):

    try:
        import gurobipy as gp
        from gurobipy import GRB

    except ImportError as exc:
        raise RuntimeError(
            "Gurobi is not installed. Install gurobipy "
            "and configure a valid Gurobi license."
        ) from exc

    from .schedule import simulate

    keys = _layout(net)

    baseline = simulate(net)

    base_steps = {
        schedule.train_id: schedule.steps
        for schedule in baseline
    }

    model = gp.Model(
        "railway_conflict_resolution"
    )

    model.Params.OutputFlag = 0
    model.Params.TimeLimit = 60

    h = {
        key: model.addVar(
            lb=0,
            ub=p.max_hold_min,
            name=f"hold_{key[0]}_{key[1]}"
        )
        for key in keys
    }

    occupancy = []

    for train in net.trains.values():

        for i, step in enumerate(base_steps[train.id]):

            track = net.tracks[step.track_id]

            resource = (
                step.track_id
                if track.single_track
                else f"{step.track_id}:{step.direction}"
            )

            occupancy.append((
                train.id,
                i,
                resource,
                step.enter,
                step.exit
            ))

    big_m = 100000.0
    pair_number = 0

    for a in range(len(occupancy)):

        ta, ia, ra, ea, xa = occupancy[a]

        for b in range(a + 1, len(occupancy)):

            tb, ib, rb, eb, xb = occupancy[b]

            if ta == tb or ra != rb:
                continue

            if (ta, ia) not in h or (tb, ib) not in h:
                continue

            delay_a = gp.quicksum(
                h[(ta, j)]
                for j in range(ia + 1)
                if (ta, j) in h
            )

            delay_b = gp.quicksum(
                h[(tb, j)]
                for j in range(ib + 1)
                if (tb, j) in h
            )

            order = model.addVar(
                vtype=GRB.BINARY,
                name=f"order_{pair_number}"
            )

            pair_number += 1

            model.addConstr(
                eb + delay_b >=
                xa + delay_a + p.headway_min
                - big_m * (1 - order)
            )

            model.addConstr(
                ea + delay_a >=
                xb + delay_b + p.headway_min
                - big_m * order
            )

    weighted = gp.quicksum(
        net.trains[tid].priority * h[(tid, i)]
        for tid, i in keys
    )

    model.setObjective(
        weighted + 0.001 * gp.quicksum(h.values()),
        GRB.MINIMIZE
    )

    model.optimize()

    if (
        model.Status not in (
            GRB.OPTIMAL,
            GRB.TIME_LIMIT,
            GRB.SUBOPTIMAL
        )
        or model.SolCount == 0
    ):
        raise RuntimeError(
            f"Gurobi could not find a feasible solution. "
            f"Status: {model.Status}"
        )

    vector = [
        h[key].X
        for key in keys
    ]

    holds = _decode(keys, vector)
    result = ev.evaluate(holds)

    history = [{
        "iter": int(model.IterCount),
        "best": round(result.fitness, 3),
        "current": round(result.fitness, 3),
        "conflicts": len(result.conflicts)
    }]

    return holds, result, history


# --------------------------------------------------
# ALGORITHM REGISTRY
# --------------------------------------------------

ALGORITHMS: dict[str, Callable] = {
    "pso": pso,
    "ga": ga,
    "aco": aco,
    "abc": abc,
    "cp": constraint_programming,
    "gurobi": gurobi
}


# --------------------------------------------------
# MAIN FUNCTION USED BY main.py
# --------------------------------------------------

def run_algorithm(
    name: str,
    net: Network,
    ev: Evaluator,
    start,
    params: SolveParams
):
    key = name.strip().lower()

    if key not in ALGORITHMS:
        raise ValueError(
            f"Unknown algorithm '{name}'. "
            f"Choose one of: {', '.join(ALGORITHMS)}"
        )

    return ALGORITHMS[key](
        net,
        ev,
        start,
        params
    )


def run_hybrid_optimizer(
    net: Network,
    ev: Evaluator,
    start,
    params: SolveParams
):
    """Run SA, PSO, GA, and CP-SAT from OS schedules; retain the best valid result."""
    from .os_scheduler import os_schedule

    def rank(candidate):
        result = candidate
        return (
            result.fitness,
            len(result.conflicts),
            round(sum(conflict.overlap_min for conflict in result.conflicts), 6),
            round(result.weighted_delay, 6),
        )

    initial_result = ev.evaluate(start)
    if initial_result.conflicts:
        start = os_schedule(net, params)
        initial_result = ev.evaluate(start)

    schedules = [(f"os-{params.policy}", copy.deepcopy(start), initial_result)]
    for policy in ("fcfs", "priority", "sjf"):
        policy_params = params.model_copy(update={"policy": policy})
        if policy == params.policy:
            continue
        holds = os_schedule(net, policy_params)
        schedules.append((f"os-{policy}", holds, ev.evaluate(holds)))

    best_os_method, holds, best_os_result = min(
        schedules,
        key=lambda item: rank(item[2])
    )
    method = best_os_method
    result = best_os_result
    best_holds = copy.deepcopy(holds)
    best_result = result
    algorithm_histories = {}
    algorithm_status = {}
    history = [{
        "iter": 0,
        "best": round(best_result.fitness, 3),
        "current": round(best_result.fitness, 3),
        "conflicts": len(best_result.conflicts),
    }]

    layout_size = max(1, len(_layout(net)))
    size_factor = max(1, math.ceil(layout_size / 2000))
    algorithm_runs = (
        (
            "simulated_annealing",
            anneal,
            min(100, max(1, params.iterations // size_factor)),
            params.seed + 11,
        ),
        (
            "pso",
            pso,
            min(4, max(1, params.iterations // (500 * size_factor))),
            params.seed + 23,
        ),
        (
            "ga",
            ga,
            min(3, max(1, params.iterations // (1000 * size_factor))),
            params.seed + 37,
        ),
    )
    for name, algorithm, iterations, seed in algorithm_runs:
        algorithm_params = params.model_copy(
            update={"iterations": iterations, "seed": seed}
        )
        candidate_holds, candidate_result, candidate_history = algorithm(
            net,
            ev,
            holds,
            algorithm_params
        )
        algorithm_histories[name] = candidate_history
        algorithm_status[name] = {
            "status": "completed",
            "fitness": round(candidate_result.fitness, 3),
        }
        if rank(candidate_result) < rank(best_result):
            best_holds = copy.deepcopy(candidate_holds)
            best_result = candidate_result
            method = name
            history = candidate_history

    try:
        cp_holds, cp_result, cp_history = constraint_programming(
            net,
            ev,
            holds,
            params
        )
        algorithm_histories["constraint_programming"] = cp_history
        algorithm_status["constraint_programming"] = {
            "status": "completed",
            "fitness": round(cp_result.fitness, 3),
        }
        if rank(cp_result) < rank(best_result):
            best_holds = copy.deepcopy(cp_holds)
            best_result = cp_result
            method = "constraint_programming"
            history = cp_history
    except RuntimeError as exc:
        algorithm_histories["constraint_programming"] = []
        algorithm_status["constraint_programming"] = {
            "status": "unavailable",
            "message": str(exc),
        }

    # Keep evaluation work bounded for large real-world timetables. Each
    # segment is checked by binary search rather than running every metaheuristic
    # for thousands of iterations over the whole network.
    budget = min(240, max(0, params.iterations))
    prune_budget = budget * 2 // 3
    keys = [
        (train_id, segment)
        for train_id, values in best_holds.items()
        for segment, value in enumerate(values)
        if value > 0
    ]
    keys.sort(
        key=lambda key: (
            -best_holds[key[0]][key[1]]
            * net.trains[key[0]].priority
        )
    )
    evaluations = 0

    for train_id, segment in keys:
        if evaluations >= prune_budget:
            break
        original = best_holds[train_id][segment]
        best_local_value = original
        best_local_result = best_result

        trial_holds = copy.deepcopy(best_holds)
        trial_holds[train_id][segment] = 0.0
        zero_result = ev.evaluate(trial_holds)
        evaluations += 1
        if rank(zero_result) <= rank(best_local_result):
            best_local_value = 0.0
            best_local_result = zero_result
        else:
            # Search for a smaller feasible hold; maintain the upper bound as
            # a known-valid schedule and never accept extra conflicts.
            low, high = 0.0, original
            while high - low > 0.5 and evaluations < prune_budget:
                middle = round(((low + high) / 2) * 2) / 2
                if middle <= low or middle >= high:
                    break
                trial_holds = copy.deepcopy(best_holds)
                trial_holds[train_id][segment] = middle
                trial_result = ev.evaluate(trial_holds)
                evaluations += 1
                if rank(trial_result) <= rank(best_local_result):
                    high = middle
                    best_local_value = middle
                    best_local_result = trial_result
                else:
                    low = middle

        best_holds[train_id][segment] = best_local_value
        best_result = best_local_result
        if evaluations % 50 < 2:
            history.append({
                "iter": evaluations,
                "best": round(best_result.fitness, 3),
                "current": round(best_result.fitness, 3),
                "conflicts": len(best_result.conflicts),
            })

    # Try reallocating some waiting from higher-priority trains to lower-
    # priority trains sharing the same single-track block.
    track_cells = {}
    for train_id, train in net.trains.items():
        for segment, (a, b) in enumerate(zip(train.route, train.route[1:])):
            track = net.track_between(a, b)
            if track and track.single_track:
                track_cells.setdefault(track.id, []).append((train_id, segment))

    transfer_rng = random.Random(params.seed)
    while evaluations < budget:
        eligible = [
            (track_id, cell)
            for track_id, cells in track_cells.items()
            for cell in cells
            if best_holds[cell[0]][cell[1]] >= 0.5
            and any(
                other_id != cell[0]
                and net.trains[other_id].priority < net.trains[cell[0]].priority
                and params.max_hold_min - best_holds[other_id][other_segment] >= 0.5
                for other_id, other_segment in cells
            )
        ]
        if not eligible:
            break

        track_id, (source_id, source_segment) = transfer_rng.choice(eligible)
        destinations = [
            cell for cell in track_cells[track_id]
            if cell[0] != source_id
            and net.trains[cell[0]].priority < net.trains[source_id].priority
            and params.max_hold_min - best_holds[cell[0]][cell[1]] >= 0.5
        ]
        if not destinations:
            continue

        destination_id, destination_segment = transfer_rng.choice(destinations)
        amount = min(
            1.0,
            best_holds[source_id][source_segment],
            params.max_hold_min - best_holds[destination_id][destination_segment],
        )
        amount = round(amount * 2) / 2
        if amount <= 0:
            continue

        candidate_holds = copy.deepcopy(best_holds)
        candidate_holds[source_id][source_segment] -= amount
        candidate_holds[destination_id][destination_segment] += amount
        candidate_result = ev.evaluate(candidate_holds)
        evaluations += 1
        if rank(candidate_result) < rank(best_result):
            best_holds = candidate_holds
            best_result = candidate_result
        if evaluations % 20 == 0:
            history.append({
                "iter": evaluations,
                "best": round(best_result.fitness, 3),
                "current": round(candidate_result.fitness, 3),
                "conflicts": len(best_result.conflicts),
            })

    if rank(best_result) > rank(best_os_result):
        best_holds = copy.deepcopy(holds)
        best_result = best_os_result
        method = best_os_method

    search_info = {
        "best_os_policy": best_os_method,
        "best_os_fitness": best_os_result.fitness,
        "best_os_conflicts": len(best_os_result.conflicts),
        "best_os_weighted_delay": best_os_result.weighted_delay,
        "algorithm_status": algorithm_status,
        "algorithm_history": algorithm_histories,
    }
    return best_holds, best_result, history, f"{method}-optimized", search_info
