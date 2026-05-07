"""Run one experiment cell.

Usage:
    python3 dt_sim/experiments/run_one.py --variant renewal --t-slew 25 --tag renewal_25s
"""
import argparse
import sys
from pathlib import Path

DT_SIM = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(DT_SIM / "src"))

from dt_sim import SimConfig, run_pipeline, save_result
from dt_sim.config import VARIANTS


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--variant", default="renewal", choices=VARIANTS)
    p.add_argument("--t-slew", type=float, default=25.0,
                   dest="t_slew", help="t_slew(theta_FoR) in seconds")
    p.add_argument("--fov", type=float, default=45.0)
    p.add_argument("--for", dest="for_deg", type=float, default=45.0,
                   help="field of regard in degrees")
    p.add_argument("--n-trials", type=int, default=20, dest="n_trials")
    p.add_argument("--horizon", type=float, default=12.0, help="hours")
    p.add_argument("--n-cities", type=int, default=10000, dest="n_cities")
    p.add_argument("--cloud-prob", type=float, default=0.66, dest="cloud_prob")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--tag", required=True)
    p.add_argument("--out-dir", default="dt_sim/runs", dest="out_dir")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = SimConfig(
        horizon_h=args.horizon,
        n_cities=args.n_cities,
        cloud_prob=args.cloud_prob,
        slew_at_for_s=args.t_slew,
        field_of_regard_deg=args.for_deg,
        fov_deg=args.fov,
        variant=args.variant,
        n_trials=args.n_trials,
        seed=args.seed,
        tag=args.tag,
        out_dir=args.out_dir,
    )
    result = run_pipeline(cfg)
    out_path = save_result(result, cfg)
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
