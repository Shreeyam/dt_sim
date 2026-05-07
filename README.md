# dt_sim

Dynamic tasking simulator for agile Earth-observation satellites.

`dt_sim` models imaging requests, orbit access windows, cloud uncertainty,
slew-constrained scheduling, and dynamic pitch-selection policies. The package
is self-contained: reusable simulator code lives in `src/dt_sim/`, command-line
experiment drivers live in `experiments/`, and bundled city data lives in
`data/`.

## Installation

```bash
git clone https://github.com/Shreeyam/dt_sim.git
cd dt_sim

python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -e ".[dev]"
```

The core simulator uses PySCIPOpt/SCIP for MILP scheduling. If `pip` cannot
install PySCIPOpt on your platform, install SCIP first and then rerun the
editable install.

Optional cloud-imagery helpers need additional dependencies:

```bash
python3 -m pip install -e ".[imagery]"
```

Meteosat downloads require EUMETSAT API credentials. Set these environment
variables before requesting an auth token:

```bash
export EUMETSAT_KEY="your-consumer-key"
export EUMETSAT_SECRET="your-consumer-secret"
```

If you keep them in a local `.env` file, load them into your shell first:

```bash
set -a
source .env
set +a
```

Then pass them to the Meteosat token helper:

```python
import os

from dt_sim.imagery import get_auth_token_meteosat

token = get_auth_token_meteosat(
    os.environ["EUMETSAT_KEY"],
    os.environ["EUMETSAT_SECRET"],
)
```

Meteosat GRIB loading also requires `pygrib`, which is most reliable on Linux.

## Components

- **Configuration:** `SimConfig` carries the simulation parameters.
- **Scenario generation:** `build_scenario` constructs repeatable request,
  orbit, access, and baseline schedule state for a trial.
- **Scheduling:** `milp_schedule` maximizes total utility subject to slew-time
  and one-observation-per-request constraints.
- **Policies:** `never`, `always`, `renewal`, and `omniscient` variants can be
  run independently or paired on identical scenarios.
- **Experiment drivers:** `experiments/run_one.py` runs one cell, while
  `experiments/run_sweep.py` runs paired baseline and agility sweeps.

## Layout

```
dt_sim/
├── README.md
├── pytest.ini
├── data/
│   └── worldcities.csv
├── src/dt_sim/
│   ├── constants.py    Physical/time constants
│   ├── geometry.py     Distances, rotations, frame builders
│   ├── orbits.py       Keplerian, frame conversions, propagation
│   ├── access.py       Request, Access, get_accesses
│   ├── cameras.py      Intrinsics, project/unproject, in-FoV filter
│   ├── scheduling.py   milp_schedule, load_worldcities
│   ├── config.py       SimConfig
│   ├── heuristics.py   Chain estimator + score_lookahead
│   ├── scenario.py     build_scenario, Scenario
│   └── simulator.py    simulate_dt, run_pipeline, save_result
├── experiments/
│   ├── run_one.py      CLI: one cell -> runs/<tag>.json
│   └── run_sweep.py    CLI: paired sweeps -> runs/*.json
├── tests/
│   └── test_smoke.py
└── runs/               JSON outputs
```

## Usage

From the project root:

```bash
# Single cell
python3 experiments/run_one.py --variant renewal --t-slew 25 --tag renewal_25s

# Tests
python3 -m pytest tests/                 # all (~1s)
python3 -m pytest tests/ -m "not slow"   # component tests only
```

## Sweeps

| Sweep | Description |
|---|---|
| `e2` | Baseline comparison for `variant in {never, always, renewal, omniscient}` |
| `e5` | Agility sweep over `t_slew in {5, 10, 15, 25, 40, 60}` |
