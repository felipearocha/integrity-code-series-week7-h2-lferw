"""
INTEGRITY CODE SERIES -- Week 7
Hydrogen-Assisted Fatigue Crack Growth (HA-FCG) and Integrated Life Prediction

Governing equation:
    da/dN = C_paris * (dK)^m * f_H(C_H)

where the hydrogen enhancement function is:
    f_H(C_H) = 1 + alpha_H * (C_H / C_ref)^beta_H

    capped at f_H_max to prevent non-physical runaway.

Stress intensity factor for semi-elliptical surface crack in cylinder:
    K = F * sigma * sqrt(pi * a)

    where F is the Newman-Raju geometry factor (simplified as F=1.12 for
    surface crack with a/t < 0.5 and a/c < 1.0).

    Note: Full Newman-Raju solution depends on a/t, a/c, and angular position.
    [ASSUMED] simplified constant F=1.12 is used here. For production use,
    implement the full parametric Newman-Raju solution.

Failure criterion:
    K_max >= K_IC(C_H)

    K_IC(C_H) = K_IC_air * exp(-lambda_K * C_H / C_ref)
    with floor at K_IC_min.

Integrated simulation flow:
    1. Compute pre-existing pit depth from NG service
    2. Start hydrogen diffusion
    3. At each time step:
        a. Get C_H at pit/crack tip depth
        b. Check pit-to-crack transition
        c. If crack: advance via HA-FCG
        d. Check failure criterion
    4. Record remaining life
"""

import numpy as np
from dataclasses import dataclass
from typing import Tuple, Optional, List

from config import (
    PipeGeometry, MaterialProps, HydrogenTransport, HydrogenEmbrittlement,
    PitGrowth, OperatingConditions, SimControl, default_config
)
from hydrogen_diffusion import solve_diffusion, get_concentration_at_depth
from pit_to_crack import (
    pit_depth_after_ng_service, hoop_stress, evaluate_pit_state,
    degraded_threshold
)


# ============================================================
# STRESS INTENSITY FACTOR
# ============================================================
def newman_raju_simplified(sigma_MPa: float, a_m: float, t_m: float) -> float:
    """
    Simplified stress intensity factor for semi-elliptical surface crack.

    K = F * sigma * sqrt(pi * a)

    F = 1.12 for a/t < 0.5 (shallow crack).
    For deeper cracks, apply Folias-type correction:
        F = 1.12 * (1 + 0.12 * (1 - a/(2*c))^2) * f_w(a/t)
    where f_w = [sec(pi*a / (2*t))]^0.5 (finite width correction)

    [ASSUMED] aspect ratio a/c = 0.5 (equivalent to c/a = 2.0 from PitGrowth default)

    Parameters
    ----------
    sigma_MPa : float
    a_m : float
        Crack depth [m].
    t_m : float
        Wall thickness [m].

    Returns
    -------
    K : float [MPa*sqrt(m)]
    """
    a_over_t = a_m / t_m
    a_over_t = min(a_over_t, 0.95)  # prevent singularity

    # Finite width (thickness) correction
    f_w = np.sqrt(1.0 / np.cos(np.pi * a_over_t / 2.0))

    # Geometry factor with shallow crack correction
    F = 1.12 * f_w

    K = F * sigma_MPa * np.sqrt(np.pi * a_m)
    return K


# ============================================================
# HYDROGEN-DEGRADED PROPERTIES
# ============================================================
def degraded_toughness(
    K_IC_air: float,
    C_H: float,
    he: HydrogenEmbrittlement,
) -> float:
    """
    K_IC(C_H) = K_IC_air * exp(-lambda_K * C_H / C_ref)
    with floor at K_IC_min.
    """
    K_IC_H = K_IC_air * np.exp(-he.lambda_K * C_H / he.C_ref_mol_m3)
    return max(K_IC_H, he.K_IC_min_MPa_sqrtm)


def hydrogen_enhancement_factor(C_H: float, he: HydrogenEmbrittlement) -> float:
    """
    f_H(C_H) = 1 + alpha_H * (C_H / C_ref)^beta_H, capped at max_enhancement.
    """
    if C_H <= 0:
        return 1.0
    f_H = 1.0 + he.alpha_H * (C_H / he.C_ref_mol_m3) ** he.beta_H
    return min(f_H, he.max_enhancement)


def ha_fcg_rate(
    dK_MPa_sqrtm: float,
    C_H: float,
    mat: MaterialProps,
    he: HydrogenEmbrittlement,
) -> float:
    """
    Hydrogen-assisted fatigue crack growth rate.

    da/dN = C_paris * dK^m * f_H(C_H)

    Parameters
    ----------
    dK_MPa_sqrtm : float
        Stress intensity factor range [MPa*sqrt(m)].
    C_H : float
        Hydrogen concentration at crack tip [mol/m3].
    mat : MaterialProps
    he : HydrogenEmbrittlement

    Returns
    -------
    da_dN : float [m/cycle]
    """
    if dK_MPa_sqrtm <= 0:
        return 0.0

    f_H = hydrogen_enhancement_factor(C_H, he)
    da_dN = mat.C_paris_air * dK_MPa_sqrtm ** mat.m_paris_air * f_H

    return da_dN


# ============================================================
# INTEGRATED LIFE PREDICTION
# ============================================================
@dataclass
class LifeResult:
    """Result of integrated remaining life calculation."""
    initial_pit_depth_m: float
    aspect_ratio: float
    pit_transitioned: bool
    transition_cycle: int
    failure_cycle: int
    failure_crack_depth_m: float
    K_IC_at_failure: float
    K_max_at_failure: float
    C_H_at_failure: float
    crack_history_a: np.ndarray       # crack depth vs cycle
    crack_history_K: np.ndarray       # K_max vs cycle
    crack_history_C_H: np.ndarray     # C_H at crack tip vs cycle
    crack_history_cycles: np.ndarray  # cycle numbers
    converged: bool
    failure_mode: str                  # 'fracture', 'wall_penetration', 'no_failure'


def run_life_prediction(
    geom: PipeGeometry,
    mat: MaterialProps,
    h2t: HydrogenTransport,
    he: HydrogenEmbrittlement,
    pit: PitGrowth,
    ops: OperatingConditions,
    sim: SimControl,
    pit_depth_override: float = None,
    aspect_ratio_override: float = None,
    diffusion_result=None,
) -> LifeResult:
    """
    Run integrated pit-to-crack-to-failure life prediction.

    Sequence:
    1. Determine pit depth from NG service (or override)
    2. Pre-compute hydrogen diffusion through wall
    3. Check pit-to-crack transition
    4. If crack: integrate HA-FCG cycle by cycle
    5. Check failure at each cycle

    Parameters
    ----------
    geom, mat, h2t, he, pit, ops, sim : config objects
    pit_depth_override : float, optional
    aspect_ratio_override : float, optional
    diffusion_result : DiffusionResult, optional
        Pre-computed diffusion result to avoid re-solving.

    Returns
    -------
    LifeResult
    """
    # Step 1: Initial pit depth
    a0 = pit_depth_override if pit_depth_override is not None else pit_depth_after_ng_service(pit)
    a0 = min(a0, 0.8 * geom.wall_thickness_m)
    ar = aspect_ratio_override if aspect_ratio_override is not None else pit.aspect_ratio_mean

    # Step 2: Hydrogen diffusion (pre-compute if not provided)
    if diffusion_result is None:
        diffusion_result = solve_diffusion(geom, h2t, ops, sim)

    # Step 3: Operating stress
    sigma_max = hoop_stress(geom, ops.p_max_MPa)
    sigma_min = hoop_stress(geom, ops.p_min_MPa)
    dsigma = sigma_max - sigma_min

    # Time per cycle
    seconds_per_cycle = 365.25 * 24 * 3600 / ops.cycles_per_year
    t_max_diffusion = diffusion_result.t_grid[-1]

    # History storage (adaptive: store every Nth cycle)
    store_interval = max(1, sim.max_cycles // 5000)
    history_a = []
    history_K = []
    history_C_H = []
    history_cycles = []

    # Step 4: Integration loop
    a_current = a0
    transition_cycle = -1
    pit_transitioned = False
    failure_cycle = sim.max_cycles
    failure_mode = "no_failure"
    K_max_final = 0.0
    K_IC_final = 0.0
    C_H_final = 0.0
    converged = True

    for cycle in range(sim.max_cycles):
        t_current_s = cycle * seconds_per_cycle
        t_clamp = min(t_current_s, t_max_diffusion)

        # Get hydrogen concentration at current crack/pit depth
        C_H = get_concentration_at_depth(diffusion_result, a_current, t_clamp)

        # Current K at max load
        K_max = newman_raju_simplified(sigma_max, a_current, geom.wall_thickness_m)
        dK = newman_raju_simplified(dsigma, a_current, geom.wall_thickness_m)

        # Check pit-to-crack transition (only once)
        if not pit_transitioned:
            pit_state = evaluate_pit_state(
                geom, mat, pit, ops, he, C_H,
                pit_depth_m=a_current, aspect_ratio=ar,
            )
            if pit_state.is_crack:
                pit_transitioned = True
                transition_cycle = cycle
            else:
                # Pit still growing slowly (no cyclic growth yet, just record)
                if cycle % store_interval == 0:
                    history_a.append(a_current)
                    history_K.append(K_max)
                    history_C_H.append(C_H)
                    history_cycles.append(cycle)
                continue  # skip crack growth

        # Hydrogen-degraded toughness
        K_IC_H = degraded_toughness(mat.K_IC_air_seam_MPa_sqrtm, C_H, he)

        # Store history
        if cycle % store_interval == 0:
            history_a.append(a_current)
            history_K.append(K_max)
            history_C_H.append(C_H)
            history_cycles.append(cycle)

        # Check failure: fracture
        if K_max >= K_IC_H:
            failure_cycle = cycle
            failure_mode = "fracture"
            K_max_final = K_max
            K_IC_final = K_IC_H
            C_H_final = C_H
            break

        # Check failure: wall penetration
        if a_current >= 0.9 * geom.wall_thickness_m:
            failure_cycle = cycle
            failure_mode = "wall_penetration"
            K_max_final = K_max
            K_IC_final = K_IC_H
            C_H_final = C_H
            break

        # Check threshold
        K_th_H = degraded_threshold(mat.K_th_air_seam_MPa_sqrtm, C_H, he)
        if dK < K_th_H:
            # Below threshold, no crack growth this cycle
            continue

        # Crack growth
        da = ha_fcg_rate(dK, C_H, mat, he)
        a_current += da

    else:
        # Reached max_cycles without failure
        failure_mode = "no_failure"
        converged = True
        K_max_final = K_max if pit_transitioned else 0.0
        K_IC_final = K_IC_H if pit_transitioned else mat.K_IC_air_seam_MPa_sqrtm
        C_H_final = C_H if pit_transitioned else 0.0

    return LifeResult(
        initial_pit_depth_m=a0,
        aspect_ratio=ar,
        pit_transitioned=pit_transitioned,
        transition_cycle=transition_cycle,
        failure_cycle=failure_cycle,
        failure_crack_depth_m=a_current,
        K_IC_at_failure=K_IC_final,
        K_max_at_failure=K_max_final,
        C_H_at_failure=C_H_final,
        crack_history_a=np.array(history_a),
        crack_history_K=np.array(history_K),
        crack_history_C_H=np.array(history_C_H),
        crack_history_cycles=np.array(history_cycles),
        converged=converged,
        failure_mode=failure_mode,
    )


def cycles_to_years(cycles: int, ops: OperatingConditions) -> float:
    """Convert fatigue cycles to calendar years."""
    return cycles / ops.cycles_per_year


def maop_under_b3112(geom: PipeGeometry, mat: MaterialProps, ops: OperatingConditions) -> float:
    """
    MAOP under ASME B31.12 for LF-ERW pipe.

    P = 2 * SMYS * t * F * H_f / D

    where F = design factor (0.50 for LF-ERW in Class 1)
    and H_f = material performance factor from B31.12 Table PL-3.7.1

    [ASSUMED] H_f = 0.54 for X52 at 7 MPa H2 (from B31.12 Option A)
    This is a significant de-rating from NG service.
    """
    H_f = 0.54  # [ASSUMED] material performance factor for X52 in H2
    F = ops.design_factor_b3112
    P = 2.0 * mat.SMYS_MPa * geom.wall_thickness_m * F * H_f / geom.outer_diameter_m
    return P
