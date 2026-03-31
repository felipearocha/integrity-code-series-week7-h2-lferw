"""
INTEGRITY CODE SERIES -- Week 7
Configuration: Hydrogen Conversion of Aging LF-ERW Pipeline

All parameters with [ASSUMED] tag require experimental calibration
before deployment. Parameters without [ASSUMED] are from published
standard specifications or well-established physical constants.

Pipeline reference: 20-inch API 5L X52 LF-ERW, manufactured 1959
(same specification as Willow River, MN failure, January 16, 2026)
"""

import numpy as np
from dataclasses import dataclass, field


# ============================================================
# PHYSICAL CONSTANTS (exact or NIST values)
# ============================================================
FARADAY = 96485.0        # C/mol
R_GAS = 8.314            # J/(mol*K)
AVOGADRO = 6.022e23      # 1/mol


# ============================================================
# PIPELINE GEOMETRY (from PHMSA Corrective Action Order, Jan 2026)
# ============================================================
@dataclass
class PipeGeometry:
    """20-inch API 5L X52 LF-ERW pipeline, 1959 vintage."""
    outer_diameter_m: float = 0.508         # 20 inches
    wall_thickness_m: float = 0.00635       # 0.250 inches [ASSUMED typical for 20" X52]
    inner_diameter_m: float = 0.0           # computed in __post_init__

    def __post_init__(self):
        self.inner_diameter_m = self.outer_diameter_m - 2.0 * self.wall_thickness_m


# ============================================================
# MATERIAL PROPERTIES: API 5L X52 (from API 5L specification)
# ============================================================
@dataclass
class MaterialProps:
    """API 5L X52 carbon steel, ferritic-pearlitic microstructure."""
    # Specification values (API 5L)
    SMYS_MPa: float = 358.0          # 52 ksi, specified minimum yield strength
    SMTS_MPa: float = 455.0          # 66 ksi, specified minimum tensile strength
    youngs_modulus_GPa: float = 207.0
    poissons_ratio: float = 0.3
    density_kg_m3: float = 7850.0

    # Fracture toughness: BASE METAL in air
    K_IC_air_base_MPa_sqrtm: float = 120.0   # [ASSUMED] typical for X52 base metal
    K_th_air_base_MPa_sqrtm: float = 6.0     # [ASSUMED] fatigue threshold in air

    # Fracture toughness: ERW SEAM (degraded due to LF-ERW bond line defects)
    K_IC_air_seam_MPa_sqrtm: float = 60.0    # [ASSUMED] LF-ERW seam, significantly lower
    K_th_air_seam_MPa_sqrtm: float = 3.5     # [ASSUMED] LF-ERW seam threshold

    # Paris law in air (da/dN in m/cycle, dK in MPa*sqrt(m))
    C_paris_air: float = 3.0e-12     # [ASSUMED] typical ferritic steel
    m_paris_air: float = 3.0         # well-established exponent range


# ============================================================
# HYDROGEN TRANSPORT PROPERTIES
# ============================================================
@dataclass
class HydrogenTransport:
    """Hydrogen diffusion and solubility in ferritic steel."""
    # Lattice diffusivity at 25 degC
    D_L_m2s: float = 1.3e-10         # [ASSUMED] typical BCC iron at 298K
    D_L_activation_kJmol: float = 6.9  # [ASSUMED] activation energy for diffusion

    # Sievert's law: C_0 = S * sqrt(p_H2)
    # S in mol/(m^3 * MPa^0.5)
    sieverts_constant: float = 0.08   # [ASSUMED] requires experimental measurement

    # Partial molar volume of H in Fe
    V_H_m3mol: float = 2.0e-6        # literature consensus

    # Trap binding energy (irreversible traps at inclusions, dislocations)
    E_trap_kJmol: float = 30.0        # [ASSUMED] mixed trap population

    # Reference temperature
    T_ref_K: float = 298.15


# ============================================================
# HYDROGEN EMBRITTLEMENT PARAMETERS
# ============================================================
@dataclass
class HydrogenEmbrittlement:
    """Parameters governing H-degradation of mechanical properties.

    All degradation functions are phenomenological fits.
    [ASSUMED] parameters require in-situ testing in pressurized H2.
    """
    # K_IC degradation: K_IC(C_H) = K_IC_air * exp(-lambda_K * C_H / C_ref)
    lambda_K: float = 0.8            # [ASSUMED] dimensionless degradation rate
    C_ref_mol_m3: float = 1.0        # reference concentration for normalization

    # K_th degradation: K_th(C_H) = K_th_air * exp(-lambda_th * C_H / C_ref)
    lambda_th: float = 1.2           # [ASSUMED] threshold degrades faster than toughness

    # Minimum toughness floor (physical lower bound)
    K_IC_min_MPa_sqrtm: float = 20.0  # [ASSUMED] even fully embrittled steel has residual K_IC
    K_th_min_MPa_sqrtm: float = 1.5   # [ASSUMED] minimum threshold

    # HA-FCG enhancement: da/dN = C_air * dK^m * [1 + alpha_H * (C_H/C_ref)^beta_H]
    alpha_H: float = 15.0            # [ASSUMED] hydrogen acceleration prefactor
    beta_H: float = 0.7              # [ASSUMED] sublinear dependence on concentration

    # Maximum enhancement cap (physical limit)
    max_enhancement: float = 50.0     # [ASSUMED] da/dN_H2 <= 50 * da/dN_air


# ============================================================
# PIT GROWTH AT ERW SEAM (from natural gas service history)
# ============================================================
@dataclass
class PitGrowth:
    """Pre-existing pit damage from decades of natural gas service.

    Power-law pit growth: a_pit(t) = k * t^n
    where t is service years.
    """
    # Pit growth rate constant (m/year^n)
    # CALIBRATION NOTE: k_pit=2e-4 with f_seam=3.0 produces 4.9 mm pit after 67 years,
    # which is 77% of wall thickness and causes immediate failure even in air.
    # Reduced to k_pit=5e-5 (moderate external corrosion with CP partially effective)
    # and f_seam=2.0 (moderate seam preferential attack, not worst-case).
    # This produces ~0.82 mm pit after 67 years (13% wall), physically consistent
    # with a pipe that survived decades of NG service without rupture.
    # Monte Carlo samples the full range including worst-case.
    k_pit_m: float = 5.0e-5          # [ASSUMED] moderate corrosion with partial CP
    n_pit: float = 0.5               # [ASSUMED] diffusion-limited pit growth exponent

    # ERW seam enhancement factor (selective seam corrosion)
    f_seam: float = 2.0              # [ASSUMED] ERW seam corrodes 2x faster than base

    # Pit aspect ratio (c/a where c=half-length, a=depth)
    aspect_ratio_mean: float = 2.0   # [ASSUMED] semi-elliptical pit shape
    aspect_ratio_std: float = 0.5    # [ASSUMED] scatter for Monte Carlo

    # Service history
    ng_service_years: float = 67.0   # 1959 to 2026


# ============================================================
# OPERATING CONDITIONS
# ============================================================
@dataclass
class OperatingConditions:
    """Pipeline operating conditions under hydrogen service."""
    # Hydrogen pressure (for conversion from NG)
    p_H2_MPa: float = 7.0            # [ASSUMED] typical H2 transmission pressure
    p_min_MPa: float = 3.5           # [ASSUMED] minimum cyclic pressure
    p_max_MPa: float = 7.0           # same as p_H2_MPa

    # Stress ratio for fatigue
    R_ratio: float = 0.5             # p_min / p_max

    # Cyclic frequency (pressure cycles per year from demand variation)
    cycles_per_year: float = 365.0   # [ASSUMED] one full cycle per day

    # Temperature
    T_K: float = 298.15              # ambient, 25 degC

    # ASME B31.12 design factor for LF-ERW pipe
    # Class 1 location, LF-ERW: design factor = 0.50 (more conservative than NG service)
    design_factor_b3112: float = 0.50  # ASME B31.12, Table PL-3.7.1

    # MAOP under hydrogen (computed)
    def maop_h2(self, geom: PipeGeometry, mat: MaterialProps) -> float:
        """Maximum allowable operating pressure under B31.12."""
        return (2.0 * mat.SMYS_MPa * geom.wall_thickness_m
                * self.design_factor_b3112 / geom.outer_diameter_m)


# ============================================================
# SIMULATION CONTROL
# ============================================================
@dataclass
class SimControl:
    """Numerical solver parameters."""
    # Hydrogen diffusion grid
    n_wall_nodes: int = 50            # spatial nodes through wall thickness
    dt_diffusion_s: float = 100.0     # time step for diffusion solver

    # Crack growth integration
    max_cycles: int = 1_000_000       # maximum fatigue cycles to simulate
    da_increment_m: float = 1.0e-5    # crack advance per sub-step for adaptive stepping

    # Monte Carlo
    n_mc_samples: int = 10_000        # Monte Carlo realizations
    mc_seed: int = 42                 # reproducibility

    # Latin Hypercube Sampling for parametric sweep
    n_lhs_samples: int = 2_000
    lhs_seed: int = 7


# ============================================================
# CONVENIENCE FUNCTION
# ============================================================
def default_config():
    """Return all default configuration objects."""
    return {
        "geom": PipeGeometry(),
        "mat": MaterialProps(),
        "h2_transport": HydrogenTransport(),
        "h2_embrittlement": HydrogenEmbrittlement(),
        "pit": PitGrowth(),
        "ops": OperatingConditions(),
        "sim": SimControl(),
    }
