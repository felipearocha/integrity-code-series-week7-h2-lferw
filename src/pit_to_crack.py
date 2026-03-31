"""
INTEGRITY CODE SERIES -- Week 7
Pit Growth at ERW Seam and Pit-to-Crack Transition

Phase 1: Pit nucleation and growth during natural gas service
    a_pit(t) = k_pit * f_seam * t^n_pit

Phase 2: Stress intensity at pit tip (Murakami's sqrt(area) parameter)
    K_pit = 0.65 * sigma * sqrt(pi * sqrt(area_pit))

    For semi-elliptical surface pit:
        area_pit = (pi/4) * (2c) * a = (pi/2) * c * a
    where a = pit depth, c = pit half-length along surface

Phase 3: Pit-to-crack transition criterion
    Transition occurs when K_pit >= K_th(C_H)
    where K_th is degraded by local hydrogen concentration

    K_th(C_H) = K_th_air * exp(-lambda_th * C_H / C_ref)
    with floor at K_th_min

Alternative transition (El Haddad):
    Transition when pit depth a >= a_th (intrinsic defect size)
    a_th = (1/pi) * (K_th / (F * sigma))^2
    where F is geometry factor

Both criteria checked; transition at whichever is met first.
"""

import numpy as np
from dataclasses import dataclass
from typing import Tuple, Optional

from config import (
    PipeGeometry, MaterialProps, PitGrowth, OperatingConditions,
    HydrogenEmbrittlement
)


@dataclass
class PitState:
    """Current state of pit at ERW seam."""
    depth_m: float               # pit depth a [m]
    half_length_m: float         # pit half-length c [m]
    K_pit_MPa_sqrtm: float       # stress intensity at pit tip
    is_crack: bool               # has pit transitioned to crack?
    transition_year: float       # year of pit-to-crack transition (-1 if not yet)


def pit_depth_after_ng_service(pit: PitGrowth) -> float:
    """
    Compute pit depth at ERW seam after natural gas service.

    a_pit = k_pit * f_seam * t^n_pit

    Returns depth in meters.
    """
    return pit.k_pit_m * pit.f_seam * pit.ng_service_years ** pit.n_pit


def pit_geometry(depth_m: float, aspect_ratio: float) -> Tuple[float, float]:
    """
    Compute pit half-length and projected area.

    Parameters
    ----------
    depth_m : float
        Pit depth a [m].
    aspect_ratio : float
        c/a ratio.

    Returns
    -------
    half_length_m : float
        c [m].
    area_m2 : float
        Projected area for Murakami parameter [m2].
    """
    c = aspect_ratio * depth_m
    # Semi-elliptical surface pit: projected area = (pi/2) * c * a
    area = (np.pi / 2.0) * c * depth_m
    return c, area


def murakami_sif(sigma_MPa: float, area_m2: float) -> float:
    """
    Murakami's stress intensity factor for surface defect.

    K = 0.65 * sigma * sqrt(pi * sqrt(area))

    This is a well-established approximation for small surface defects.
    The 0.65 factor applies to surface-breaking defects (vs 0.50 for internal).

    Parameters
    ----------
    sigma_MPa : float
        Applied stress [MPa].
    area_m2 : float
        Projected defect area [m2].

    Returns
    -------
    K : float
        Stress intensity factor [MPa*sqrt(m)].
    """
    return 0.65 * sigma_MPa * np.sqrt(np.pi * np.sqrt(area_m2))


def hoop_stress(geom: PipeGeometry, p_MPa: float) -> float:
    """
    Barlow's formula for hoop stress.

    sigma_h = p * D / (2 * t)

    Parameters
    ----------
    geom : PipeGeometry
    p_MPa : float

    Returns
    -------
    sigma_h_MPa : float
    """
    return p_MPa * geom.outer_diameter_m / (2.0 * geom.wall_thickness_m)


def degraded_threshold(
    K_th_air: float,
    C_H: float,
    he: HydrogenEmbrittlement,
) -> float:
    """
    Hydrogen-degraded fatigue threshold.

    K_th(C_H) = K_th_air * exp(-lambda_th * C_H / C_ref)
    with floor at K_th_min.

    Parameters
    ----------
    K_th_air : float [MPa*sqrt(m)]
    C_H : float [mol/m3]
    he : HydrogenEmbrittlement

    Returns
    -------
    K_th_H : float [MPa*sqrt(m)]
    """
    K_th_H = K_th_air * np.exp(-he.lambda_th * C_H / he.C_ref_mol_m3)
    return max(K_th_H, he.K_th_min_MPa_sqrtm)


def el_haddad_threshold_depth(
    K_th_MPa_sqrtm: float,
    sigma_MPa: float,
    F_geom: float = 1.12,
) -> float:
    """
    El Haddad intrinsic defect size for short crack regime.

    a_th = (1/pi) * (K_th / (F * sigma))^2

    Parameters
    ----------
    K_th_MPa_sqrtm : float
    sigma_MPa : float
    F_geom : float
        Geometry factor for semi-elliptical surface crack. Default 1.12.

    Returns
    -------
    a_th : float [m]
    """
    if sigma_MPa <= 0:
        return np.inf
    return (1.0 / np.pi) * (K_th_MPa_sqrtm / (F_geom * sigma_MPa)) ** 2


def evaluate_pit_state(
    geom: PipeGeometry,
    mat: MaterialProps,
    pit: PitGrowth,
    ops: OperatingConditions,
    he: HydrogenEmbrittlement,
    C_H_at_pit: float,
    pit_depth_m: Optional[float] = None,
    aspect_ratio: Optional[float] = None,
) -> PitState:
    """
    Evaluate whether a pit at the ERW seam has transitioned to a crack.

    Checks two criteria:
    1. Murakami K_pit >= K_th(C_H)  [stress intensity criterion]
    2. a_pit >= a_th (El Haddad)     [intrinsic defect size criterion]

    Parameters
    ----------
    geom, mat, pit, ops, he : configuration objects
    C_H_at_pit : float
        Hydrogen concentration at pit depth [mol/m3].
    pit_depth_m : float, optional
        Override pit depth. Default: compute from NG service.
    aspect_ratio : float, optional
        Override aspect ratio. Default: pit.aspect_ratio_mean.

    Returns
    -------
    PitState
    """
    if pit_depth_m is None:
        pit_depth_m = pit_depth_after_ng_service(pit)
    if aspect_ratio is None:
        aspect_ratio = pit.aspect_ratio_mean

    # Cap pit depth at 80% of wall thickness (physical limit for surface pit)
    pit_depth_m = min(pit_depth_m, 0.8 * geom.wall_thickness_m)

    half_length, area = pit_geometry(pit_depth_m, aspect_ratio)
    sigma_h = hoop_stress(geom, ops.p_max_MPa)

    # Stress intensity at pit
    K_pit = murakami_sif(sigma_h, area)

    # Hydrogen-degraded threshold (at ERW seam)
    K_th_H = degraded_threshold(mat.K_th_air_seam_MPa_sqrtm, C_H_at_pit, he)

    # El Haddad depth
    a_th = el_haddad_threshold_depth(K_th_H, sigma_h)

    # Transition criteria
    criterion_1 = K_pit >= K_th_H
    criterion_2 = pit_depth_m >= a_th

    is_crack = criterion_1 or criterion_2

    return PitState(
        depth_m=pit_depth_m,
        half_length_m=half_length,
        K_pit_MPa_sqrtm=K_pit,
        is_crack=is_crack,
        transition_year=-1.0,  # to be set by caller
    )
