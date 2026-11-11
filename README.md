# Railway Conflict Solver - backend

FastAPI service that reads the railway graph from Neo4j, detects track conflicts,
resolves them with an OS-style scheduler and improves the result with simulated annealing.

## Setup (macOS, from the project root)
    source venv/bin/activate
    cd backend
    pip install -r requirements.txt
    cp .env.example .env          # then set NEO4J_PASSWORD

## Check, in this order
    python -m tests.test_engine           # algorithms only, no Neo4j. Must end with OK
    python -m scripts.inspect_graph       # shows what your Neo4j contains
    uvicorn app.main:app --reload --port 8000
    open http://localhost:8000/api/health   # ok: true means the graph loaded
    open http://localhost:8000/docs         # try every endpoint

## Neo4j schema assumed (edit app/repository.py if yours differs)
    (:Station {id, name, lat?, lon?})
    (:Station)-[:TRACK {id?, length_km, single_track}]->(:Station)
    (:Train {id, name?, priority, speed_kmh, departure_min, dwell_min?})
    (:Train)-[:VISITS {seq}]->(:Station)

## Endpoints
    GET  /api/health          status and counts, or the startup error
    POST /api/reload          re-read Neo4j
    GET  /api/network         stations and tracks
    GET  /api/trains          trains with routes
    GET  /api/baseline        planned timetable and its conflicts
    POST /api/solve/os        {"policy":"fcfs|priority|sjf","headway_min":2}
    POST /api/solve/optimize  {"start_from":"os","iterations":4000, ...}

## Layers
    repository.py -> schedule.py -> conflicts.py -> evaluation.py -> os_scheduler.py -> optimizer.py -> main.py
