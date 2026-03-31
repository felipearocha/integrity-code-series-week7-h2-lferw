"""
INTEGRITY CODE SERIES -- Week 7
Monte Carlo Probabilistic Remaining Life and LHS Parametric Sweep

Monte Carlo samples over:
    1. Initial pit depth (lognormal, from NG service scatter)
    2. Pit aspect ratio (normal, clipped)
    3. ERW seam toughness K_IC (normal, scatter in bond line quality)
    4. Hydrogen diffusivity D_L (lognormal, microstructure scatter)
    5. Seam enhancement factor f_seam (uniform, quality uncertainty)

LHS parametric sweep over:
    1. H2 pressure (3 to 10 MPa)
    2. Initial pit depth (0.5 to 3.0 mm)
    3. D_L (5e-11 to 5e-10 m2/s)
    4. K_IC_seam (30 to 90 MPa*sqrt(m))
    5. f_seam (1.5 to 5.0)
    6. Aspect ratio (1.0 to 4.0)

Output: CDF of remaining life, sensitivity analysis, risk maps.
"""

import numpy as np
from typing import Dict, List, Tuple
from dataclasses import dataclass
import warnings

from config import (
    PipeGeometry, MaterialProps, HydrogenTransport, HydrogenEmbrittlement,
    PitGrowth, OperatingConditions, SimControl
)
from hydrogen_diffusion import solve_diffusion
from ha_fcg import run_life_prediction, cycles_to_years


@dataclass
class MonteCarloResult:
    """Monte Carlo simulation results."""
    n_samples: int
    remaining_life_years: np.ndarray
    failure_modes: List[str]
    initial_pit_depths_m: np.ndarray
    aspect_ratios: np.ndarray
    K_IC_seam_values: np.ndarray
    D_L_values: np.ndarray
    f_seam_values: np.ndarray
    pit_transition_flags: np.ndarray
    percentile_5: float
    percentile_50: float
    percentile_95: float
    fraction_below_10yr: float
    fraction_below_20yr: float


def latin_hypercube_sample(n_samples: int, n_dims: int, seed: int = 42) -> np.ndarray:
    """
    Latin Hypercube Sampling in [0,1]^n_dims.

    Parameters
    ----------
    n_samples : int
    n_dims : int
    seed : int

    Returns
    -------
    samples : ndarray, shape (n_samples, n_dims)
        Uniform samples in [0,1] via LHS.
    """
    rng = np.random.RandomState(seed)
    samples = np.zeros((n_samples, n_dims))
    for j in range(n_dims):
        perm = rng.permutation(n_samples)
        for i in range(n_samples):
            samples[perm[i], j] = (i + rng.random()) / n_samples
    return samples


def run_monte_carlo(
    geom: PipeGeometry,
    mat: MaterialProps,
    h2t: HydrogenTransport,
    he: HydrogenEmbrittlement,
    pit: PitGrowth,
    ops: OperatingConditions,
    sim: SimControl,
) -> MonteCarloResult:
    """
    Run Monte Carlo simulation for probabilistic remaining life.

    Samples uncertain parameters and runs life prediction for each.
    Uses a shared diffusion solution (recomputed only when D_L changes).
    For speed, uses reduced max_cycles for MC.

    Returns
    -------
    MonteCarloResult
    """
    rng = np.random.RandomState(sim.mc_seed)
    n = sim.n_mc_samples

    # Sample distributions
    # Pit depth: lognormal with mean = pit_depth_after_ng_service
    mean_depth = pit.k_pit_m * pit.f_seam * pit.ng_service_years ** pit.n_pit
    depth_sigma_log = 0.4  # [ASSUMED] log-std
    pit_depths = rng.lognormal(
        mean=np.log(mean_depth) - 0.5 * depth_sigma_log ** 2,
        sigma=depth_sigma_log,
        size=n,
    )
    pit_depths = np.clip(pit_depths, 1.0e-4, 0.8 * geom.wall_thickness_m)

    # Aspect ratio: normal
    aspect_ratios = rng.normal(pit.aspect_ratio_mean, pit.aspect_ratio_std, size=n)
    aspect_ratios = np.clip(aspect_ratios, 0.5, 6.0)

    # K_IC at seam: normal with scatter
    K_IC_seam_mean = mat.K_IC_air_seam_MPa_sqrtm
    K_IC_seam_std = 10.0  # [ASSUMED] MPa*sqrt(m)
    K_IC_values = rng.normal(K_IC_seam_mean, K_IC_seam_std, size=n)
    K_IC_values = np.clip(K_IC_values, 20.0, 120.0)

    # D_L: lognormal
    D_L_log_mean = np.log(h2t.D_L_m2s)
    D_L_log_std = 0.3  # [ASSUMED]
    D_L_values = rng.lognormal(
        mean=D_L_log_mean - 0.5 * D_L_log_std ** 2,
        sigma=D_L_log_std,
        size=n,
    )

    # f_seam: uniform
    f_seam_values = rng.uniform(1.5, 5.0, size=n)

    # Run simulations (with reduced max_cycles for speed)
    sim_fast = SimControl(
        n_wall_nodes=sim.n_wall_nodes,
        dt_diffusion_s=sim.dt_diffusion_s,
        max_cycles=200_000,  # reduced for MC speed
        da_increment_m=sim.da_increment_m,
        n_mc_samples=n,
        mc_seed=sim.mc_seed,
        n_lhs_samples=sim.n_lhs_samples,
        lhs_seed=sim.lhs_seed,
    )

    remaining_lives = np.zeros(n)
    failure_modes = []
    transition_flags = np.zeros(n, dtype=bool)

    for i in range(n):
        # Create modified transport with sampled D_L
        h2t_i = HydrogenTransport(
            D_L_m2s=D_L_values[i],
            D_L_activation_kJmol=h2t.D_L_activation_kJmol,
            sieverts_constant=h2t.sieverts_constant,
            V_H_m3mol=h2t.V_H_m3mol,
            E_trap_kJmol=h2t.E_trap_kJmol,
            T_ref_K=h2t.T_ref_K,
        )

        # Solve diffusion with this sample's D_L
        diff_i = solve_diffusion(geom, h2t_i, ops, sim_fast)

        # Create modified material with sampled K_IC
        mat_i = MaterialProps(
            SMYS_MPa=mat.SMYS_MPa,
            SMTS_MPa=mat.SMTS_MPa,
            youngs_modulus_GPa=mat.youngs_modulus_GPa,
            poissons_ratio=mat.poissons_ratio,
            density_kg_m3=mat.density_kg_m3,
            K_IC_air_base_MPa_sqrtm=mat.K_IC_air_base_MPa_sqrtm,
            K_th_air_base_MPa_sqrtm=mat.K_th_air_base_MPa_sqrtm,
            K_IC_air_seam_MPa_sqrtm=K_IC_values[i],
            K_th_air_seam_MPa_sqrtm=mat.K_th_air_seam_MPa_sqrtm,
            C_paris_air=mat.C_paris_air,
            m_paris_air=mat.m_paris_air,
        )

        result = run_life_prediction(
            geom, mat_i, h2t_i, he, pit, ops, sim_fast,
            pit_depth_override=pit_depths[i],
            aspect_ratio_override=aspect_ratios[i],
            diffusion_result=diff_i,
        )

        remaining_lives[i] = cycles_to_years(result.failure_cycle, ops)
        failure_modes.append(result.failure_mode)
        transition_flags[i] = result.pit_transitioned

    # Clip at max simulation time
    max_years = cycles_to_years(sim_fast.max_cycles, ops)
    remaining_lives = np.clip(remaining_lives, 0, max_years)

    return MonteCarloResult(
        n_samples=n,
        remaining_life_years=remaining_lives,
        failure_modes=failure_modes,
        initial_pit_depths_m=pit_depths,
        aspect_ratios=aspect_ratios,
        K_IC_seam_values=K_IC_values,
        D_L_values=D_L_values,
        f_seam_values=f_seam_values,
        pit_transition_flags=transition_flags,
        percentile_5=float(np.percentile(remaining_lives, 5)),
        percentile_50=float(np.percentile(remaining_lives, 50)),
        percentile_95=float(np.percentile(remaining_lives, 95)),
        fraction_below_10yr=float(np.mean(remaining_lives < 10.0)),
        fraction_below_20yr=float(np.mean(remaining_lives < 20.0)),
    )


@dataclass
class LHSResult:
    """Latin Hypercube Sweep results."""
    n_samples: int
    params: np.ndarray           # (n_samples, n_dims)
    param_names: List[str]
    remaining_life_years: np.ndarray
    failure_modes: List[str]
    spearman_correlations: Dict[str, float]


def run_lhs_sweep(
    geom: PipeGeometry,
    mat: MaterialProps,
    h2t: HydrogenTransport,
    he: HydrogenEmbrittlement,
    pit: PitGrowth,
    ops: OperatingConditions,
    sim: SimControl,
) -> LHSResult:
    """
    Latin Hypercube parametric sweep for sensitivity analysis.

    6 dimensions:
        0: p_H2 [3, 10] MPa
        1: pit_depth [0.5e-3, 3.0e-3] m
        2: D_L [5e-11, 5e-10] m2/s
        3: K_IC_seam [30, 90] MPa*sqrt(m)
        4: f_seam [1.5, 5.0]
        5: aspect_ratio [1.0, 4.0]
    """
    n = sim.n_lhs_samples
    n_dims = 6
    param_names = ["p_H2_MPa", "pit_depth_m", "D_L_m2s",
                   "K_IC_seam", "f_seam", "aspect_ratio"]

    # LHS in [0,1]
    lhs_raw = latin_hypercube_sample(n, n_dims, seed=sim.lhs_seed)

    # Map to physical ranges
    params = np.zeros_like(lhs_raw)
    params[:, 0] = 3.0 + lhs_raw[:, 0] * 7.0          # p_H2: [3, 10]
    params[:, 1] = 0.5e-3 + lhs_raw[:, 1] * 2.5e-3    # pit: [0.5, 3.0] mm
    params[:, 2] = 5e-11 * (10.0 ** (lhs_raw[:, 2] * 1.0))  # D_L: log-uniform
    params[:, 3] = 30.0 + lhs_raw[:, 3] * 60.0         # K_IC: [30, 90]
    params[:, 4] = 1.5 + lhs_raw[:, 4] * 3.5           # f_seam: [1.5, 5.0]
    params[:, 5] = 1.0 + lhs_raw[:, 5] * 3.0           # AR: [1.0, 4.0]

    # Reduced simulation for sweep
    sim_fast = SimControl(
        n_wall_nodes=30,
        dt_diffusion_s=200.0,
        max_cycles=100_000,
        da_increment_m=sim.da_increment_m,
        n_mc_samples=sim.n_mc_samples,
        mc_seed=sim.mc_seed,
        n_lhs_samples=n,
        lhs_seed=sim.lhs_seed,
    )

    remaining_lives = np.zeros(n)
    failure_modes = []

    for i in range(n):
        ops_i = OperatingConditions(
            p_H2_MPa=params[i, 0],
            p_min_MPa=params[i, 0] * 0.5,
            p_max_MPa=params[i, 0],
            R_ratio=0.5,
            cycles_per_year=ops.cycles_per_year,
            T_K=ops.T_K,
            design_factor_b3112=ops.design_factor_b3112,
        )

        h2t_i = HydrogenTransport(
            D_L_m2s=params[i, 2],
            D_L_activation_kJmol=h2t.D_L_activation_kJmol,
            sieverts_constant=h2t.sieverts_constant,
            V_H_m3mol=h2t.V_H_m3mol,
            E_trap_kJmol=h2t.E_trap_kJmol,
            T_ref_K=h2t.T_ref_K,
        )

        mat_i = MaterialProps(
            SMYS_MPa=mat.SMYS_MPa,
            SMTS_MPa=mat.SMTS_MPa,
            youngs_modulus_GPa=mat.youngs_modulus_GPa,
            poissons_ratio=mat.poissons_ratio,
            density_kg_m3=mat.density_kg_m3,
            K_IC_air_base_MPa_sqrtm=mat.K_IC_air_base_MPa_sqrtm,
            K_th_air_base_MPa_sqrtm=mat.K_th_air_base_MPa_sqrtm,
            K_IC_air_seam_MPa_sqrtm=params[i, 3],
            K_th_air_seam_MPa_sqrtm=mat.K_th_air_seam_MPa_sqrtm,
            C_paris_air=mat.C_paris_air,
            m_paris_air=mat.m_paris_air,
        )

        try:
            diff_result = solve_diffusion(geom, h2t_i, ops_i, sim_fast)
            result = run_life_prediction(
                geom, mat_i, h2t_i, he, pit, ops_i, sim_fast,
                pit_depth_override=params[i, 1],
                aspect_ratio_override=params[i, 5],
                diffusion_result=diff_result,
            )
            remaining_lives[i] = cycles_to_years(result.failure_cycle, ops_i)
            failure_modes.append(result.failure_mode)
        except Exception:
            remaining_lives[i] = 0.0
            failure_modes.append("error")

    # Spearman rank correlation for sensitivity
    from scipy.stats import spearmanr
    correlations = {}
    for j, name in enumerate(param_names):
        rho, _ = spearmanr(params[:, j], remaining_lives)
        correlations[name] = float(rho) if not np.isnan(rho) else 0.0

    return LHSResult(
        n_samples=n,
        params=params,
        param_names=param_names,
        remaining_life_years=remaining_lives,
        failure_modes=failure_modes,
        spearman_correlations=correlations,
    )
