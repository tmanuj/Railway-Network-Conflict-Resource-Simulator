
"""Fitness function shared by the OS scheduler and optimizer."""

from dataclasses import dataclass

from .conflicts import Conflict, detect
from .models import SolveParams
from .schedule import Network, TrainSchedule, simulate


@dataclass
class Evaluation:
    schedules: list[TrainSchedule]
    conflicts: list[Conflict]
    delays: dict[str, float]
    weighted_delay: float
    fitness: float


class Evaluator:
    def __init__(self, net: Network, params: SolveParams):
        self.net = net
        self.p = params

        # Free-flow arrival times without holds
        self.baseline = {
            s.train_id: s.arrival
            for s in simulate(net)
        }

    def evaluate(self, holds: dict[str, list[float]]) -> Evaluation:
        schedules = simulate(self.net, holds)

        conflicts = detect(
            self.net,
            schedules,
            self.p.headway_min
        )

        delays = {
            s.train_id: max(
                0.0,
                s.arrival - self.baseline[s.train_id]
            )
            for s in schedules
        }

        weighted = sum(
            self.net.trains[t].priority * d
            for t, d in delays.items()
        )

        overlap = sum(
            c.overlap_min for c in conflicts
        )

        fitness = (
            self.p.w_conflict * len(conflicts)
            + self.p.w_overlap * overlap
            + self.p.w_delay * weighted
        )

        return Evaluation(
            schedules,
            conflicts,
            delays,
            weighted,
            fitness
        )
