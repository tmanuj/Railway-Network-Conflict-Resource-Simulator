
"""Layer: conflict engine.

Rules
  * single track: any two trains (either direction) closer than `headway` -> conflict
  * double track: one track per direction, so only same-direction trains can conflict
"""

import math
from dataclasses import dataclass

from .schedule import Network, TrainSchedule

EPS = 1e-6  # float tolerance so back-to-back trains at exactly `headway` are not flagged


@dataclass
class Conflict:
    track_id: str
    train_a: str
    train_b: str
    time: float
    overlap_min: float
    kind: str  # "head_on" | "following"


def detect(
    net: Network,
    schedules: list[TrainSchedule],
    headway: float,
) -> list[Conflict]:

    # Validate headway before detecting conflicts
    if not math.isfinite(headway) or headway < 0:
        raise ValueError(
            "Headway must be a finite, non-negative number"
        )

    groups: dict[str, list[tuple]] = {}

    for sch in schedules:
        for s in sch.steps:
            single = net.tracks[s.track_id].single_track

            key = (
                s.track_id
                if single
                else f"{s.track_id}:{s.direction}"
            )

            groups.setdefault(key, []).append(
                (s.enter, s.exit, sch.train_id, s.direction, s.track_id)
            )

    out: list[Conflict] = []

    for occ in groups.values():
        occ.sort()

        for i, p in enumerate(occ):
            for q in occ[i + 1:]:

                if q[0] >= p[1] + headway - EPS:
                    break

                if p[2] == q[2]:
                    continue

                out.append(
                    Conflict(
                        track_id=p[4],
                        train_a=p[2],
                        train_b=q[2],
                        time=q[0],
                        overlap_min=p[1] + headway - q[0],
                        kind=(
                            "head_on"
                            if p[3] != q[3]
                            else "following"
                        ),
                    )
                )

    out.sort(key=lambda c: c.time)

    return out
