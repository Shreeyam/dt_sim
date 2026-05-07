"""Experiment sweep driver.

Two sweeps:
    e2 (baselines): one (t_slew, fov) cell, all requested variants paired.
    e5 (agility):   sweep t_slew at default fov, paired across {always, renewal,
                    never, omniscient}.

Both share the underlying scenarios across variants — building each scenario
once and replaying all variants on it cuts MILP baseline cost by ~4x.

Usage:
    python3 experiments/run_sweep.py e2 [--n-trials 20 --horizon 12]
    python3 experiments/run_sweep.py e5 [--n-trials 20 --horizon 12]
"""
import argparse
import sys
import time as clock
from pathlib import Path

DT_SIM = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(DT_SIM / "src"))

from dt_sim import SimConfig, run_paired, save_result
from dt_sim.scheduling import load_worldcities

DEFAULT_VARIANTS = ["never", "always", "renewal", "omniscient"]
E5_AGILITIES_S = [5, 10, 15, 25, 40, 60]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("sweep", choices=["e2", "e5"])
    p.add_argument("--n-trials", type=int, default=20, dest="n_trials")
    p.add_argument("--horizon", type=float, default=12.0, help="hours")
    p.add_argument("--n-cities", type=int, default=10000, dest="n_cities")
    p.add_argument("--cloud-prob", type=float, default=0.66, dest="cloud_prob")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--variants", nargs="+", default=DEFAULT_VARIANTS,
                   help="variants to evaluate per scenario")
    p.add_argument("--out-dir", default="runs", dest="out_dir")
    return p.parse_args()


def _run_cell(*, tag: str, t_slew: float, args, requests, variants):
    cfg = SimConfig(
        slew_at_for_s=t_slew,
        horizon_h=args.horizon,
        n_cities=args.n_cities,
        cloud_prob=args.cloud_prob,
        n_trials=args.n_trials,
        seed=args.seed,
        out_dir=args.out_dir,
        tag=tag,
    )
    results = run_paired(cfg, variants=variants, requests=requests)
    out_paths = []
    for variant, result in results.items():
        # Each variant is saved with its own tag (set by run_paired via cfg replace).
        sub_cfg = SimConfig(**result.cfg)
        out_paths.append(save_result(result, sub_cfg))
    return out_paths


def sweep_e2(args, requests):
    print("=" * 70)
    print(f"Sweep E2: baselines @ t_slew=25s, fov=45°, {args.n_trials} trials")
    print("=" * 70)
    paths = _run_cell(
        tag="e2", t_slew=25.0, args=args, requests=requests,
        variants=args.variants,
    )
    print(f"\nE2 done. Wrote {len(paths)} files:")
    for p in paths:
        print(f"  {p}")


def sweep_e5(args, requests):
    print("=" * 70)
    print(f"Sweep E5: agility @ t_slew ∈ {E5_AGILITIES_S}, fov=45°, "
          f"{args.n_trials} trials each")
    print("=" * 70)
    t_total = clock.time()
    all_paths = []
    for t_slew in E5_AGILITIES_S:
        print(f"\n--- t_slew = {t_slew}s ---")
        paths = _run_cell(
            tag=f"e5_t{t_slew:02d}", t_slew=float(t_slew),
            args=args, requests=requests, variants=args.variants,
        )
        all_paths.extend(paths)
    print(f"\nE5 done in {clock.time() - t_total:.0f}s. Wrote {len(all_paths)} files.")


def main():
    args = parse_args()
    print(f"Loading {args.n_cities} world cities...")
    requests = load_worldcities(n=args.n_cities)
    if args.sweep == "e2":
        sweep_e2(args, requests)
    else:
        sweep_e5(args, requests)


if __name__ == "__main__":
    main()
