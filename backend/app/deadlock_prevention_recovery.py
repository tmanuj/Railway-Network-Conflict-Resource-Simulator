"""Deadlock prevention and recovery helpers for railway resource allocation.

Prevention uses the Banker's safety test: a train's resource request is granted
only when the resulting allocation leaves at least one safe completion order.
Recovery chooses a victim train, aborts it, and releases its held resources.

This is a reusable resource-manager module; it does not silently alter the
existing timetable solver. Call it from the API/scheduler when resource
allocation and waiting state are available.
"""
from __future__ import annotations

from copy import deepcopy
from typing import Mapping

Vector = Mapping[str, int]


def _check_vector(label: str, vector: Vector) -> None:
    for resource, units in vector.items():
        if not isinstance(units, int) or units < 0:
            raise ValueError(f"{label}[{resource!r}] must be a non-negative integer")


def _safe_sequence(
    available: dict[str, int],
    allocation: dict[str, dict[str, int]],
    need: dict[str, dict[str, int]],
) -> list[str] | None:
    """Return a safe train order, or None if the state is unsafe."""
    resources = set(available)
    for row in allocation.values():
        resources.update(row)
    for row in need.values():
        resources.update(row)
    work = {r: available.get(r, 0) for r in resources}
    finish = {t: False for t in allocation}
    sequence: list[str] = []

    while len(sequence) < len(allocation):
        candidate = next((t for t in sorted(allocation) if not finish[t]
                          and all(need[t].get(r, 0) <= work[r] for r in resources)), None)
        if candidate is None:
            return None
        for resource in resources:
            work[resource] += allocation[candidate].get(resource, 0)
        finish[candidate] = True
        sequence.append(candidate)
    return sequence


class RailwayResourceManager:
    """Banker's-algorithm manager for tracks/platforms with finite capacities.

    ``total`` maps each resource name to its total available units. ``maximum``
    maps each train to its declared maximum possible demand. A train must be
    registered before requesting resources.
    """

    def __init__(self, total: Vector, maximum: Mapping[str, Vector]):
        _check_vector("total", total)
        self.total = dict(total)
        self.available = dict(total)
        self.maximum = {t: dict(v) for t, v in maximum.items()}
        self.allocation = {t: {r: 0 for r in self.total} for t in self.maximum}
        self.need = {t: {r: self.maximum[t].get(r, 0) for r in self.total}
                     for t in self.maximum}
        for train, vector in self.maximum.items():
            _check_vector(f"maximum[{train}]", vector)
            for resource, units in vector.items():
                if resource not in self.total:
                    raise ValueError(f"Unknown resource {resource!r} in maximum demand")
                if units > self.total[resource]:
                    raise ValueError(f"Train {train} maximum exceeds total {resource} capacity")

    def request_resources(self, train_id: str, request: Vector) -> dict:
        """Grant a request only if it is available and keeps the state safe."""
        if train_id not in self.allocation:
            raise KeyError(f"Train {train_id!r} is not registered")
        _check_vector("request", request)
        for resource, units in request.items():
            if resource not in self.total:
                raise ValueError(f"Unknown resource {resource!r}")
            if units > self.need[train_id][resource]:
                return {"granted": False, "reason": "exceeds_declared_maximum"}
            if units > self.available[resource]:
                return {"granted": False, "reason": "resource_unavailable"}

        # Tentatively allocate, then roll back if the new state is unsafe.
        for resource, units in request.items():
            self.available[resource] -= units
            self.allocation[train_id][resource] += units
            self.need[train_id][resource] -= units

        sequence = _safe_sequence(self.available, self.allocation, self.need)
        if sequence is None:
            for resource, units in request.items():
                self.available[resource] += units
                self.allocation[train_id][resource] -= units
                self.need[train_id][resource] += units
            return {"granted": False, "reason": "unsafe_state_prevented", "safe_sequence": None}

        return {"granted": True, "reason": "safe_state", "safe_sequence": sequence}

    def release_resources(self, train_id: str, release: Vector | None = None) -> dict:
        """Release selected units, or all units when ``release`` is omitted."""
        if train_id not in self.allocation:
            raise KeyError(f"Train {train_id!r} is not registered")
        amounts = dict(self.allocation[train_id]) if release is None else dict(release)
        _check_vector("release", amounts)
        for resource, units in amounts.items():
            if resource not in self.total or units > self.allocation[train_id][resource]:
                raise ValueError(f"Invalid release of {units} units of {resource} by {train_id}")
        for resource, units in amounts.items():
            self.allocation[train_id][resource] -= units
            self.available[resource] += units
            self.need[train_id][resource] += units
        return {"released": amounts, "available": dict(self.available)}

    def snapshot(self) -> dict:
        """Return a JSON-serializable view for API responses/dashboard display."""
        sequence = _safe_sequence(self.available, self.allocation, self.need)
        return {
            "available": dict(self.available),
            "allocation": deepcopy(self.allocation),
            "need": deepcopy(self.need),
            "safe": sequence is not None,
            "safe_sequence": sequence,
        }


def recover_deadlock(
    allocation: Mapping[str, Vector],
    priorities: Mapping[str, int] | None = None,
    waiting: Mapping[str, float] | None = None,
) -> dict:
    """Choose and abort one victim train, returning its resources for reuse.

    Victim policy: lowest priority first (smaller numeric priority is lower);
    tie-break by largest number of held resource units, then longest wait,
    then train ID. This is a simulation decision, not a real train-control
    command. In a production railway system, dispatchers must approve actions.
    """
    if not allocation:
        return {"recovered": False, "reason": "no_allocated_trains", "victim": None,
                "released_resources": {}}
    priorities = priorities or {}
    waiting = waiting or {}
    for train, vector in allocation.items():
        _check_vector(f"allocation[{train}]", vector)

    victim = min(
        allocation,
        key=lambda t: (priorities.get(t, 0),
                       -sum(allocation[t].values()),
                       -float(waiting.get(t, 0.0)), t),
    )
    released = {r: units for r, units in allocation[victim].items() if units > 0}
    return {
        "recovered": True,
        "victim": victim,
        "action": "abort_train_and_release_resources",
        "released_resources": released,
        "remaining_trains": sorted(t for t in allocation if t != victim),
        "note": "Simulation recovery decision only; require dispatcher approval before real-world use.",
    }
