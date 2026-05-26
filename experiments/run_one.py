"""Run one experiment cell.

Usage:
    python3 experiments/run_one.py --variant renewal --t-slew 25 --tag renewal_25s
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
    p.add_argument("--horizon", type=float, default=24.0, help="hours")
    p.add_argument("--n-cities", type=int, default=10000, dest="n_cities")
    p.add_argument("--cloud-prob", type=float, default=0.66, dest="cloud_prob")
    p.add_argument("--cloud-source", choices=["iid", "bcm"], default="iid",
                   dest="cloud_source")
    p.add_argument("--bcm-data-dir", default=None, dest="bcm_data_dir")
    p.add_argument("--bcm-download-missing", action="store_true",
                   dest="bcm_download_missing")
    p.add_argument("--trial-sampling", choices=["dates", "raan"], default="dates",
                   dest="trial_sampling")
    p.add_argument("--trial-start", default="2025-01-01T00:00:00",
                   dest="trial_start_iso")
    p.add_argument("--trial-end", default="2025-12-31T23:30:00",
                   dest="trial_end_iso")
    p.add_argument("--raan-deg", type=float, default=0.0, dest="raan_deg")
    p.add_argument("--mean-anom-deg", type=float, default=0.0,
                   dest="mean_anom_deg")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--tag", required=True)
    p.add_argument("--out-dir", default="runs", dest="out_dir")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = SimConfig(
        horizon_h=args.horizon,
        n_cities=args.n_cities,
        cloud_prob=args.cloud_prob,
        cloud_source=args.cloud_source,
        bcm_data_dir=args.bcm_data_dir,
        bcm_download_missing=args.bcm_download_missing,
        trial_sampling=args.trial_sampling,
        trial_start_iso=args.trial_start_iso,
        trial_end_iso=args.trial_end_iso,
        raan_deg=args.raan_deg,
        mean_anom_deg=args.mean_anom_deg,
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
