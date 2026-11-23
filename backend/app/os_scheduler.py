
"""Non-preemptive railway track dispatch with four selection policies.

FCFS, Priority, SJF and Round Robin.
Each decision reserves one complete track section.
Round Robin rotates at section boundaries, not mid-track.
"""

from .models import SolveParams
from .schedule import Network


def dispatch_schedule(
    net: Network,
    params: SolveParams,
    policy: str | None = None,
    preferences: dict | None = None
) -> dict[str, list[float]]:

    policy = (policy or params.policy).lower()

    if policy not in {
        "fcfs", "priority", "sjf", "rr", "round_robin"
    }:
        raise ValueError(f"Unknown policy: {policy}")

    headway = max(0.0, float(params.headway_min))

    quantum = max(
        0.01,
        float(getattr(params, "rr_quantum_min", 5.0))
    )

    # Look-ahead window for competing trains.
    window = max(10.0, quantum * 2.0)

    if preferences is not None:
        window = max(window, 30.0)

    trains = list(net.trains.values())

    holds = net.zero_holds()

    leg = {t.id: 0 for t in trains}

    ready = {
        t.id: float(net.start(t))
        for t in trains
    }

    reservations = {}

    last_turn = {t.id: -1 for t in trains}
    turn = 0

    def resource(track, frm):
        if track.single_track:
            return track.id

        direction = 1 if track.a == frm else -1
        return (track.id, direction)

    def earliest_free(key, time, duration):
        time = float(time)

        for left, right in sorted(
            reservations.get(key, [])
        ):
            if time + duration + headway <= left:
                break

            if (
                time < right + headway
                and time + duration + headway > left
            ):
                time = right + headway

        return time

    while True:
        candidates = []

        for t in trains:
            i = leg[t.id]

            if i >= len(t.route) - 1:
                continue

            track = net.track_between(
                t.route[i],
                t.route[i + 1]
            )

            duration = max(
                0.000001,
                float(net.run_minutes(t, i, track))
            )

            key = resource(track, t.route[i])

            candidates.append(
                (
                    t,
                    i,
                    key,
                    duration,
                    ready[t.id]
                )
            )

        if not candidates:
            break

        earliest_ready = min(
            c[4] for c in candidates
        )

        contenders = [
            c for c in candidates
            if c[4] <= earliest_ready + window + 1e-9
        ]

        if preferences is not None:

            chosen = min(
                contenders,
                key=lambda c: (
                    float(
                        preferences.get(
                            (c[0].id, c[1]),
                            0.0
                        )
                    ),
                    c[4],
                    c[0].id
                )
            )

        elif policy == "priority":

            chosen = min(
                contenders,
                key=lambda c: (
                    -c[0].priority,
                    c[4],
                    c[0].id
                )
            )

        elif policy == "sjf":

            chosen = min(
                contenders,
                key=lambda c: (
                    c[3],
                    c[4],
                    c[0].id
                )
            )

        elif policy in {"rr", "round_robin"}:

            chosen = min(
                contenders,
                key=lambda c: (
                    last_turn[c[0].id],
                    c[4],
                    c[0].id
                )
            )

        else:
            # First Come First Served
            chosen = min(
                contenders,
                key=lambda c: (
                    c[4],
                    c[0].id
                )
            )

        t, i, key, duration, available = chosen

        enter = earliest_free(
            key,
            available,
            duration
        )

        holds[t.id][i] = max(
            0.0,
            enter - available
        )

        reservations.setdefault(
            key, []
        ).append(
            (enter, enter + duration)
        )

        ready[t.id] = (
            enter
            + duration
            + float(net.dwell(t, i))
        )

        leg[t.id] += 1

        last_turn[t.id] = turn
        turn += 1

    return holds


def os_schedule(
    net: Network,
    params: SolveParams
) -> dict[str, list[float]]:

    return dispatch_schedule(net, params)
