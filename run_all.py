"""
INTEGRITY CODE SERIES -- Week 7
Run the full chain: diffusion, life prediction, Monte Carlo, LHS sweep,
GBR surrogate and every figure in assets/.

    python run_all.py           # 10,000 MC + 2,000 LHS on all cores, writes to assets/
    python run_all.py --quick   # 20 MC + 60 LHS, writes to a temp folder (about 2 minutes)
"""

import argparse
import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

from config import default_config
from hydrogen_diffusion import solve_diffusion
from ha_fcg import run_life_prediction, cycles_to_years
from monte_carlo import run_monte_carlo, run_lhs_sweep
from ml_surrogate import train_surrogate, validate_feature_ranking
from visualization import (
    plot_hero_diffusion_crack, plot_monte_carlo_cdf, plot_sensitivity_tornado,
    plot_surrogate_parity, plot_iso_risk_contour, plot_fad, generate_gif,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--quick", action="store_true",
                        help="small sample counts, output to a temp folder")
    args = parser.parse_args()

    cfg = default_config()
    geom, mat, h2t, he = cfg["geom"], cfg["mat"], cfg["h2_transport"], cfg["h2_embrittlement"]
    pit, ops, sim = cfg["pit"], cfg["ops"], cfg["sim"]
    workers = os.cpu_count() or 1
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")
    if args.quick:
        sim.n_mc_samples, sim.n_lhs_samples = 20, 60
        out = tempfile.mkdtemp(prefix="week7_quick_")

    print("Running hydrogen diffusion...")
    diff = solve_diffusion(geom, h2t, ops, sim)
    print(f"Diffusion solved: {len(diff.t_grid)} snapshots")

    print("Running life prediction...")
    life = run_life_prediction(geom, mat, h2t, he, pit, ops, sim, diffusion_result=diff)
    print(f"Failure at cycle {life.failure_cycle} "
          f"({cycles_to_years(life.failure_cycle, ops):.1f} years)")
    print(f"Failure mode: {life.failure_mode}")

    print(f"Running Monte Carlo ({sim.n_mc_samples:,} samples)...")
    mc = run_monte_carlo(geom, mat, h2t, he, pit, ops, sim, workers=workers)
    print(f"P5={mc.percentile_5:.1f} yr, P50={mc.percentile_50:.1f} yr, "
          f"P95={mc.percentile_95:.1f} yr")
    print(f"Fraction below 10yr: {mc.fraction_below_10yr:.1%}")

    print(f"Running LHS sweep ({sim.n_lhs_samples:,} samples)...")
    lhs = run_lhs_sweep(geom, mat, h2t, he, pit, ops, sim, workers=workers)
    print("Spearman correlations:")
    for k, v in sorted(lhs.spearman_correlations.items(), key=lambda x: -abs(x[1])):
        print(f"  {k}: {v:.3f}")

    print("Training GBR surrogate...")
    surr = train_surrogate(lhs.params, lhs.remaining_life_years, lhs.param_names)
    print(f"R2 test: {surr.r2_test:.3f}")
    for k, v in validate_feature_ranking(surr.feature_importances).items():
        print(f"  {k}: {v}")

    print("Generating visuals...")
    plot_hero_diffusion_crack(diff, life, geom, save_path=os.path.join(out, "hero_h2_crack.png"))
    plot_monte_carlo_cdf(mc, save_path=os.path.join(out, "mc_cdf.png"))
    plot_sensitivity_tornado(lhs.spearman_correlations,
                             save_path=os.path.join(out, "sensitivity_tornado.png"))
    plot_surrogate_parity(surr, save_path=os.path.join(out, "surrogate_parity.png"))
    plot_iso_risk_contour(lhs.params, lhs.remaining_life_years, lhs.param_names,
                          save_path=os.path.join(out, "iso_risk_contour.png"),
                          surrogate_model=surr.model)
    plot_fad(life, geom, mat, he, ops.p_max_MPa, save_path=os.path.join(out, "fad.png"))
    generate_gif(diff, life, geom, save_path=os.path.join(out, "h2_crack_evolution.gif"))
    print(f"All outputs generated in {out}")


if __name__ == "__main__":
    main()
