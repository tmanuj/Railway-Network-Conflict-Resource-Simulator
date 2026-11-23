
"""PSO, GA, ACO and ABC for railway dispatch optimization.

The algorithms search train/track-leg dispatch priorities.
The dispatch scheduler creates the holds and the real
Evaluator calculates conflicts, delays and fitness.

The initial solution is retained unless a better result
is found.
"""

import math
import random

from .os_scheduler import dispatch_schedule


MAX_EVALS = 8000


def clamp01(x):
    return 0.0 if x < 0.0 else 1.0 if x > 1.0 else x


def budget(p):
    return max(1, min(3000, int(p.iterations)))


class Search:

    def __init__(self, name, net, evaluator, start, params):
        self.name = name
        self.net = net
        self.ev = evaluator
        self.p = params

        seed = (
            f"{name}|"
            f"{getattr(params, 'seed', 0)}|"
            f"{getattr(params, 'disturb_seed', 0)}"
        )

        self.rng = random.Random(seed)

        self.keys = [
            (t.id, i)
            for t in net.trains.values()
            for i in range(len(t.route) - 1)
        ]

        self.cache = {}

        self.best_holds = {
            k: list(v)
            for k, v in start.items()
        }

        self.best_eval = evaluator.evaluate(
            self.best_holds
        )

        self.best_score = self.score(
            self.best_eval
        )

        self.best_vector = (
            self.seed_vectors()[0]
            if self.keys else []
        )

        self.history = []
        self.evaluations = 0

    @staticmethod
    def score(e):
        return (
            len(e.conflicts),
            round(sum(e.delays.values()), 8),
            round(e.weighted_delay, 8)
        )

    @property
    def exhausted(self):
        return self.evaluations >= MAX_EVALS

    def evaluate(self, vector):

        ranks = tuple(
            sorted(
                range(len(vector)),
                key=lambda i: (vector[i], i)
            )
        )

        if ranks in self.cache:
            return self.cache[ranks]

        preferences = {
            self.keys[i]: rank
            for rank, i in enumerate(ranks)
        }

        holds = dispatch_schedule(
            self.net,
            self.p,
            preferences=preferences
        )

        ev = self.ev.evaluate(holds)

        score = self.score(ev)

        self.evaluations += 1

        if score < self.best_score:
            self.best_score = score
            self.best_eval = ev
            self.best_holds = holds
            self.best_vector = list(vector)

        if len(self.cache) > 10000:
            self.cache.clear()

        self.cache[ranks] = (score, ev)

        return score, ev

    def seed_vectors(self):

        trains = self.net.trains
        n = max(1, len(self.keys))

        def vec(keyfn):
            order = sorted(
                range(len(self.keys)),
                key=lambda i: keyfn(self.keys[i])
            )

            v = [0.0] * len(self.keys)

            for rank, i in enumerate(order):
                v[i] = rank / n

            return v

        try:
            dep = lambda k: (
                trains[k[0]].departure_min,
                k[1]
            )

            hi = lambda k: (
                -trains[k[0]].priority,
                trains[k[0]].departure_min,
                k[1]
            )

            lo = lambda k: (
                trains[k[0]].priority,
                trains[k[0]].departure_min,
                k[1]
            )

            return [
                vec(dep),
                vec(hi),
                vec(lo)
            ]

        except Exception:
            return [
                [
                    self.rng.random()
                    for _ in self.keys
                ]
                for _ in range(3)
            ]

    def random_vector(self):
        return [
            self.rng.random()
            for _ in self.keys
        ]

    def population(self, n):
        pop = [
            list(v)
            for v in self.seed_vectors()
        ]

        while len(pop) < n:
            pop.append(self.random_vector())

        return pop[:n]

    def mutate(self, vector, scale=0.3):

        result = list(vector)

        if not result:
            return result

        count = max(
            1,
            min(
                len(result),
                round(math.sqrt(len(result)))
            )
        )

        for i in self.rng.sample(
            range(len(result)),
            count
        ):
            result[i] = clamp01(
                result[i]
                + self.rng.gauss(0, scale)
            )

        if (
            len(result) > 1
            and self.rng.random() < 0.5
        ):
            a, b = self.rng.sample(
                range(len(result)), 2
            )

            result[a], result[b] = (
                result[b], result[a]
            )

        return result

    def record(self, iteration, current=None):

        cur = (
            current
            if current is not None
            else self.best_eval
        )

        self.history.append({
            "iter": iteration,
            "best": round(
                self.best_eval.fitness, 3
            ),
            "current": round(
                cur.fitness, 3
            ),
            "conflicts": len(
                self.best_eval.conflicts
            ),
            "delay_min": round(
                sum(self.best_eval.delays.values()),
                2
            )
        })

    def finish(self):
        return (
            self.best_holds,
            self.best_eval,
            self.history
        )


# ============================================
# PARTICLE SWARM OPTIMIZATION (PSO)
# ============================================

def pso(net, ev, start, p):

    s = Search("pso", net, ev, start, p)

    if not s.keys:
        return s.finish()

    n = 14
    dim = len(s.keys)
    iters = budget(p)

    pos = s.population(n)

    vel = [
        [
            s.rng.uniform(-0.1, 0.1)
            for _ in range(dim)
        ]
        for _ in range(n)
    ]

    personal = [
        v[:] for v in pos
    ]

    pscore = [
        s.evaluate(v)[0]
        for v in pos
    ]

    for it in range(iters):

        if s.exhausted:
            break

        w = (
            0.9
            - 0.5 * it / max(1, iters - 1)
        )

        iter_best = None

        for k in range(n):

            for j in range(dim):

                v = (
                    w * vel[k][j]
                    + 1.6 * s.rng.random()
                    * (
                        personal[k][j]
                        - pos[k][j]
                    )
                    + 1.6 * s.rng.random()
                    * (
                        s.best_vector[j]
                        - pos[k][j]
                    )
                )

                vel[k][j] = max(
                    -0.4,
                    min(0.4, v)
                )

                pos[k][j] = clamp01(
                    pos[k][j] + vel[k][j]
                )

            score, e = s.evaluate(pos[k])

            if score < pscore[k]:
                pscore[k] = score
                personal[k] = pos[k][:]

            if (
                iter_best is None
                or score < iter_best[0]
            ):
                iter_best = (score, e)

        s.record(it, iter_best[1])

    return s.finish()


# ============================================
# GENETIC ALGORITHM (GA)
# ============================================

def ga(net, ev, start, p):

    s = Search("ga", net, ev, start, p)

    if not s.keys:
        return s.finish()

    size = 16
    dim = len(s.keys)
    iters = budget(p)

    population = s.population(size)

    def tournament(scored, k=3):
        return min(
            s.rng.sample(
                scored,
                min(k, len(scored))
            ),
            key=lambda x: x[0]
        )[1]

    for it in range(iters):

        if s.exhausted:
            break

        scored = [
            (s.evaluate(v)[0], v)
            for v in population
        ]

        scored.sort(
            key=lambda x: x[0]
        )

        elite = [
            v[:]
            for _, v in scored[:2]
        ]

        children = []

        while (
            len(children) + len(elite)
            < size - 1
        ):

            a = tournament(scored)
            b = tournament(scored)

            child = [
                a[j]
                if s.rng.random() < 0.5
                else b[j]
                for j in range(dim)
            ]

            if s.rng.random() < 0.85:
                child = s.mutate(
                    child, 0.25
                )

            children.append(child)

        children.append(
            s.random_vector()
        )

        population = elite + children

        s.record(it)

    return s.finish()


# ============================================
# ANT COLONY OPTIMIZATION (ACO)
# ============================================

def aco(net, ev, start, p):

    s = Search("aco", net, ev, start, p)

    if not s.keys:
        return s.finish()

    bins = 10
    ants = 10
    rho = 0.15
    iters = budget(p)

    pher = [
        [1.0] * bins
        for _ in s.keys
    ]

    gbest_score = None
    gbest_choice = None

    for it in range(iters):

        if s.exhausted:
            break

        ibest = None

        for _ in range(ants):

            chosen = [
                s.rng.choices(
                    range(bins),
                    weights=row,
                    k=1
                )[0]
                for row in pher
            ]

            vec = [
                (c + s.rng.random()) / bins
                for c in chosen
            ]

            score, e = s.evaluate(vec)

            if (
                ibest is None
                or score < ibest[0]
            ):
                ibest = (
                    score,
                    chosen,
                    e
                )

        if (
            gbest_score is None
            or ibest[0] <= gbest_score
        ):
            gbest_score = ibest[0]
            gbest_choice = ibest[1]

        # Pheromone evaporation
        for row in pher:
            for b in range(bins):
                row[b] = max(
                    0.05,
                    row[b] * (1 - rho)
                )

        # Iteration-best pheromone deposit
        for row, idx in zip(
            pher, ibest[1]
        ):
            row[idx] += 1.0

        # Global-best pheromone deposit
        for row, idx in zip(
            pher, gbest_choice
        ):
            row[idx] += 0.5

        s.record(it, ibest[2])

    return s.finish()


# ============================================
# ARTIFICIAL BEE COLONY (ABC)
# ============================================

def abc(net, ev, start, p):

    s = Search("abc", net, ev, start, p)

    if not s.keys:
        return s.finish()

    n = 12
    dim = len(s.keys)
    limit = 10
    iters = budget(p)

    food = s.population(n)

    scores = [
        s.evaluate(v)[0]
        for v in food
    ]

    trials = [0] * n

    def neighbour(i):

        k = s.rng.choice([
            j for j in range(n)
            if j != i
        ])

        cand = food[i][:]

        for j in s.rng.sample(
            range(dim),
            max(1, dim // 8)
        ):

            cand[j] = clamp01(
                cand[j]
                + s.rng.uniform(-1, 1)
                * (
                    cand[j]
                    - food[k][j]
                )
            )

        if cand == food[i]:
            cand = s.mutate(
                cand, 0.15
            )

        return cand

    def try_source(i):

        cand = neighbour(i)

        score, e = s.evaluate(cand)

        if score < scores[i]:
            food[i] = cand
            scores[i] = score
            trials[i] = 0

        else:
            trials[i] += 1

        return e

    for it in range(iters):

        if s.exhausted:
            break

        last = None

        # Employed bees
        for i in range(n):
            last = try_source(i)

        # Onlooker bees
        order = sorted(
            range(n),
            key=lambda i: scores[i]
        )

        weights = [
            n - r
            for r in range(n)
        ]

        for _ in range(n):

            i = s.rng.choices(
                order,
                weights=weights,
                k=1
            )[0]

            last = try_source(i)

        # Scout bees
        for i in range(n):

            if trials[i] >= limit:

                food[i] = (
                    s.random_vector()
                )

                scores[i] = s.evaluate(
                    food[i]
                )[0]

                trials[i] = 0

        s.record(it, last)

    return s.finish()


# ============================================
# ALGORITHM REGISTRY
# ============================================

ALGORITHMS = {
    "pso": pso,
    "ga": ga,
    "aco": aco,
    "abc": abc
}


def run_algorithm(
    name,
    net,
    ev,
    start,
    params
):

    key = name.strip().lower()

    if key not in ALGORITHMS:
        raise ValueError(
            f"Unknown algorithm {name}; "
            f"choose {list(ALGORITHMS)}"
        )

    return ALGORITHMS[key](
        net,
        ev,
        start,
        params
    )
