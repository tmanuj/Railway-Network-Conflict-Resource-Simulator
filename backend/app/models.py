
from typing import Literal
from pydantic import BaseModel, Field


# ==========================================
# STATION MODEL
# ==========================================

class Station(BaseModel):
    id: str
    name: str
    lat: float | None = None
    lon: float | None = None


# ==========================================
# TRACK MODEL
# ==========================================

class Track(BaseModel):
    id: str
    a: str
    b: str
    length_km: float = 1.0
    single_track: bool = True


# ==========================================
# TRAIN MODEL
# ==========================================

class Train(BaseModel):
    id: str
    name: str = ""
    priority: int = 1
    speed_kmh: float = 80.0
    departure_min: float = 0.0
    dwell_min: float = 0.0
    route: list[str]
    run_min: list[float] | None = None
    dwells: list[float] | None = None


# ==========================================
# SOLVER PARAMETERS
# ==========================================

class SolveParams(BaseModel):

    # Track safety
    headway_min: float = 2.0

    # Disturbance settings
    disturb_count: int = 6
    disturb_min: float = 15.0
    disturb_max: float = 45.0
    disturb_seed: int = 1

    delays: dict[str, float] = Field(default_factory=dict)

    # ======================================
    # OS SCHEDULING ALGORITHMS
    # ======================================

    # FCFS, Priority, SJF, Round Robin
    policy: Literal[
        "fcfs",
        "priority",
        "sjf",
        "rr"
    ] = "priority"

    # Round Robin quantum in minutes
    rr_quantum_min: float = Field(
        default=5.0,
        gt=0
    )

    # ======================================
    # METAHEURISTIC OPTIMIZATION
    # ======================================

    start_from: Literal["planned", "os"] = "os"

    iterations: int = 3000

    start_temp: float = 60.0
    cooling: float = 0.999

    max_hold_min: float = 20.0

    seed: int = 7

    # ======================================
    # FITNESS WEIGHTS
    # ======================================

    w_conflict: float = 100.0
    w_overlap: float = 5.0
    w_delay: float = 1.0
