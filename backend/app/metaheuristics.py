
"""Optimization algorithms for Railway Network Conflict Resource Simulator.

Algorithms:
1. PSO - Particle Swarm Optimization
2. GA - Genetic Algorithm
3. ACO - Ant Colony Optimization
4. ABC - Artificial Bee Colony
5. Gurobi - Exact MILP Solver (optional)
"""

import random
from typing import Callable

from .evaluation import Evaluator
from .models import SolveParams
from .schedule import Network


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
# 5. GUROBI EXACT OPTIMIZATION
# --------------------------------------------------

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
