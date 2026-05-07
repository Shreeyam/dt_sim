# dt_sim

Clean-room rewrite of the dynamic-tasking experiments simulator for the Paper 1
(DT heuristics) experiments. Self-contained: the backend (`src/dt_sim/`) replaces
the legacy `phd_code/dynamic_tasker` package; `experiments/` holds the CLIs that
drive it.

## Design

- **One config, one entry point.** `SimConfig` carries every knob; `run_pipeline(cfg)`
  is the only thing experiments call. No per-cell forks of a `__main__`.
- **Decision rule is a callable.** Variants (`always`, `renewal`, plus `never`
  and `omniscient` baselines) plug into the simulator via
  `heuristics.score_lookahead`. `renewal` is the two-boundary chain estimator;
  everything else would be an approximation of it.
- **Scenario setup is pure.** `build_scenario` takes a config and a RAAN, returns
  a `Scenario`. Same scenario can be replayed under different policies for paired
  comparisons.
- **Sweeps are loops, not files.** `experiments/run_sweep.py` iterates DoE cells
  and calls `run_pipeline` per cell. Adding a sweep dimension is editing a list,
  not a script.
- **Stable Access ids.** `Access.aid` is a process-stable monotonic counter, so
  cloud-state dicts survive copies and pickles (unlike `id(a)` keying).
- **MILP solver: SCIP.** Profiling shows MILP scheduling is ~95% of wall time
  per trial, dominated by the 2 baseline schedules (conv + omni). HiGHS was
  bake-offed (2026-04-24) and ran neck-and-neck with SCIP on identical optima
  — this packing structure is presolve-friendly territory and SCIP handles it
  well. Real speedup would require caching baseline schedules per `(RAAN, t_slew)`
  to disk across sweep re-runs, not a different solver.

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
│   └── run_one.py      CLI: one cell -> runs/<tag>.json
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

## Mapping to paper experiments

| Paper section | Sweep |
|---|---|
| §5.3 baselines | `variant ∈ {never, always, renewal, omniscient}` at default cell |
| §5.5 agility   | `t_slew ∈ {5, 10, 15, 25, 40, 60}` × `variant ∈ {always, renewal}` |
