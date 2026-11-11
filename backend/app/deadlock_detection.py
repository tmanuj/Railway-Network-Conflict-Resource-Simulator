"""Deadlock detection for railway shared resources.

A train is treated as a process. Tracks/platforms are resource types. The
module uses the standard multiple-instance deadlock-detection reduction:
resources held by a process are released virtually when its outstanding
request can be satisfied. Processes that cannot finish are deadlocked.

All resource quantities are non-negative integers. Example resource names:
"TRACK_DDN_HWR", "PLATFORM_DDN_1".
"""
from __future__ import annotations

from typing import Mapping

Vector = Mapping[str, int]
AllocationTable = Mapping[str, Vector]


def _validate_vector(name: str, vector: Vector) -> None:
    for resource, quantity in vector.items():
        if not isinstance(quantity, int) or quantity < 0:
            raise ValueError(f"{name}[{resource!r}] must be a non-negative integer")


def _resources(*tables: Mapping[str, Vector], available: Vector) -> set[str]:
    result = set(available)
    for table in tables:
        for vector in table.values():
            result.update(vector)
    return result


def detect_deadlocks(
    available: Vector,
    allocation: AllocationTable,
    request: AllocationTable,
) -> dict:
    """Detect deadlocked trains from available, allocated and requested units.

    Args:
        available: currently free units per resource type.
        allocation: ``{train_id: {resource: units_held}}``.
        request: outstanding requests that each train is currently waiting for.

    Returns a JSON-serializable dictionary with ``deadlocked_trains``,
    ``safe_trains``, ``wait_for_graph`` and ``cycles``.
    """
    _validate_vector("available", available)
    for train, vector in allocation.items():
        _validate_vector(f"allocation[{train}]", vector)
    for train, vector in request.items():
        _validate_vector(f"request[{train}]", vector)

    trains = sorted(set(allocation) | set(request))
    resources = _resources(allocation, request, available=available)
    initial_available = {r: int(available.get(r, 0)) for r in resources}
    work = dict(initial_available)
    alloc = {t: {r: int(allocation.get(t, {}).get(r, 0)) for r in resources} for t in trains}
    req = {t: {r: int(request.get(t, {}).get(r, 0)) for r in resources} for t in trains}

    # A process with no allocated resources cannot be part of a resource cycle;
    # mark it finished initially, as in the standard detection algorithm.
    finish = {t: not any(alloc[t].values()) for t in trains}
    progress = True
    while progress:
        progress = False
        for train in trains:
            if finish[train]:
                continue
            if all(req[train][r] <= work[r] for r in resources):
                for resource in resources:
                    work[resource] += alloc[train][resource]
                finish[train] = True
                progress = True

    deadlocked = [t for t in trains if not finish[t]]
    safe = [t for t in trains if finish[t]]

    # Build a diagnostic wait-for graph: an edge A -> B means A is waiting
    # for a resource currently held by B. For multi-instance resources, this
    # graph is explanatory; the reduction result above is authoritative.
    graph: dict[str, list[str]] = {t: [] for t in trains}
    for waiter in trains:
        for resource in resources:
            if req[waiter][resource] <= initial_available[resource]:
                continue
            holders = [holder for holder in trains
                       if holder != waiter and alloc[holder][resource] > 0]
            graph[waiter].extend(holders)
        graph[waiter] = sorted(set(graph[waiter]))

    cycles = _find_cycles(graph, set(deadlocked))
    return {
        "deadlock_detected": bool(deadlocked),
        "deadlocked_trains": deadlocked,
        "safe_trains": safe,
        "wait_for_graph": graph,
        "cycles": cycles,
        "available": dict(available),
    }


def _find_cycles(graph: dict[str, list[str]], allowed: set[str]) -> list[list[str]]:
    """Return simple directed cycles for readable diagnostics."""
    cycles: set[tuple[str, ...]] = set()
    for start in sorted(allowed):
        path: list[str] = []
        in_path: set[str] = set()

        def visit(node: str) -> None:
            path.append(node)
            in_path.add(node)
            for nxt in graph.get(node, []):
                if nxt not in allowed:
                    continue
                if nxt == start:
                    # Canonical rotation prevents duplicate representations.
                    cycle = path[:]
                    rotations = [tuple(cycle[i:] + cycle[:i]) for i in range(len(cycle))]
                    cycles.add(min(rotations))
                elif nxt not in in_path and nxt > start:
                    visit(nxt)
            in_path.remove(node)
            path.pop()

        visit(start)
    return [list(cycle) for cycle in sorted(cycles)]
