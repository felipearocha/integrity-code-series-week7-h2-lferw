"""
INTEGRITY CODE SERIES -- Week 7
Hydrogen Diffusion Through Pipe Wall: Stress-Assisted Fick's Law (Oriani)

Governing PDE (1D through-wall, thin-wall approximation):

    dC_L/dt = D_L * d2C_L/dx2 + D_L * (V_H / RT) * d/dx(C_L * dsigma_h/dx)

where:
    C_L     = lattice hydrogen concentration [mol/m3]
    D_L     = lattice diffusivity [m2/s]
    V_H     = partial molar volume of H in Fe [m3/mol]
    sigma_h = hydrostatic stress at position x [Pa]
    R       = gas constant [J/(mol*K)]
    T       = temperature [K]

Boundary conditions:
    C_L(x=0, t) = C_0 = S * sqrt(p_H2)   [Sievert's law, inner surface]
    C_L(x=w, t) = 0                        [outer surface, atmosphere/CP]

Initial condition:
    C_L(x, 0) = 0   [no hydrogen in pipe wall from NG service]

Stress field (elastic, Lame solution for thick cylinder under internal pressure):
    sigma_r(r)     = p*ri^2/(ro^2-ri^2) * (1 - ro^2/r^2)
    sigma_theta(r) = p*ri^2/(ro^2-ri^2) * (1 + ro^2/r^2)
    sigma_z        = p*ri^2/(ro^2-ri^2)  [closed-end condition]
    sigma_h        = (sigma_r + sigma_theta + sigma_z) / 3

Discretization: Method of Lines with central differences in space, RK45 in time.
"""

import numpy as np
from dataclasses import dataclass
from typing import Tuple

from config import (
    R_GAS, PipeGeometry, HydrogenTransport, OperatingConditions, SimControl
)


@dataclass
class DiffusionResult:
    """Container for diffusion solver output."""
    x_grid: np.ndarray          # spatial positions through wall [m]
    t_grid: np.ndarray          # time steps [s]
    C_field: np.ndarray         # concentration field [mol/m3], shape (n_t, n_x)
    sigma_h_field: np.ndarray   # hydrostatic stress [Pa], shape (n_x,)


def compute_stress_field(geom: PipeGeometry, p_MPa: float, n_nodes: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Lame solution for hydrostatic stress through thick-walled cylinder.

    Parameters
    ----------
    geom : PipeGeometry
    p_MPa : float
        Internal pressure in MPa.
    n_nodes : int
        Number of spatial nodes through wall thickness.

    Returns
    -------
    x_grid : ndarray
        Position through wall, 0 = inner surface, w = outer surface [m].
    sigma_h : ndarray
        Hydrostatic stress at each node [Pa].
    """
    p_Pa = p_MPa * 1.0e6
    ri = geom.inner_diameter_m / 2.0
    ro = geom.outer_diameter_m / 2.0
    w = geom.wall_thickness_m

    x_grid = np.linspace(0.0, w, n_nodes)
    r_grid = ri + x_grid  # radial position

    ri2 = ri ** 2
    ro2 = ro ** 2
    denom = ro2 - ri2

    sigma_r = p_Pa * ri2 / denom * (1.0 - ro2 / r_grid ** 2)
    sigma_theta = p_Pa * ri2 / denom * (1.0 + ro2 / r_grid ** 2)
    sigma_z = p_Pa * ri2 / denom  # closed-end axial stress

    sigma_h = (sigma_r + sigma_theta + sigma_z) / 3.0

    return x_grid, sigma_h


def sieverts_surface_concentration(h2t: HydrogenTransport, p_H2_MPa: float) -> float:
    """
    Sievert's law: C_0 = S * sqrt(p_H2).

    Parameters
    ----------
    h2t : HydrogenTransport
    p_H2_MPa : float

    Returns
    -------
    C_0 : float
        Surface hydrogen concentration [mol/m3].
    """
    return h2t.sieverts_constant * np.sqrt(p_H2_MPa)


def solve_diffusion(
    geom: PipeGeometry,
    h2t: HydrogenTransport,
    ops: OperatingConditions,
    sim: SimControl,
    t_total_s: float = None,
    n_snapshots: int = 50,
) -> DiffusionResult:
    """
    Solve 1D hydrogen diffusion through pipe wall using explicit finite differences.

    Uses forward Euler with stability check (Courant condition).
    Includes stress-assisted diffusion term (Oriani).

    Parameters
    ----------
    geom, h2t, ops, sim : configuration objects
    t_total_s : float, optional
        Total simulation time. Default: time for H to traverse 90% of wall.
    n_snapshots : int
        Number of time snapshots to store.

    Returns
    -------
    DiffusionResult
    """
    n = sim.n_wall_nodes
    w = geom.wall_thickness_m
    dx = w / (n - 1)
    D_L = h2t.D_L_m2s
    T = ops.T_K

    # Stability: dt <= dx^2 / (2*D_L)
    dt_max = 0.4 * dx ** 2 / D_L
    dt = min(sim.dt_diffusion_s, dt_max)

    # Default total time: characteristic diffusion time * 2
    if t_total_s is None:
        t_diff_char = w ** 2 / D_L
        t_total_s = 2.0 * t_diff_char

    n_steps = int(t_total_s / dt) + 1
    snapshot_interval = max(1, n_steps // n_snapshots)

    # Stress field (static, from max pressure)
    x_grid, sigma_h = compute_stress_field(geom, ops.p_max_MPa, n)

    # Stress gradient (central differences, one-sided at boundaries)
    dsigma_dx = np.gradient(sigma_h, dx)

    # Sievert's BC at inner surface
    C_0 = sieverts_surface_concentration(h2t, ops.p_H2_MPa)

    # Stress-coupling coefficient
    V_H = h2t.V_H_m3mol
    stress_coeff = V_H / (R_GAS * T)

    # Initialize
    C = np.zeros(n)
    C[0] = C_0  # inner surface BC

    # Storage
    t_stored = []
    C_stored = []
    t_stored.append(0.0)
    C_stored.append(C.copy())

    t_current = 0.0
    for step in range(1, n_steps + 1):
        C_new = C.copy()

        # Interior nodes: explicit finite differences
        for i in range(1, n - 1):
            # Standard diffusion term
            d2C_dx2 = (C[i + 1] - 2.0 * C[i] + C[i - 1]) / dx ** 2

            # Stress-assisted term: d/dx(C * dsigma/dx)
            # = dC/dx * dsigma/dx + C * d2sigma/dx2
            dC_dx = (C[i + 1] - C[i - 1]) / (2.0 * dx)
            d2sigma_dx2_i = (sigma_h[min(i + 1, n - 1)] - 2.0 * sigma_h[i]
                             + sigma_h[max(i - 1, 0)]) / dx ** 2

            stress_term = stress_coeff * (dC_dx * dsigma_dx[i] + C[i] * d2sigma_dx2_i)

            C_new[i] = C[i] + dt * D_L * (d2C_dx2 + stress_term)

        # Enforce BCs
        C_new[0] = C_0
        C_new[-1] = 0.0

        # Clamp non-physical negatives
        C_new = np.maximum(C_new, 0.0)

        C = C_new
        t_current += dt

        if step % snapshot_interval == 0:
            t_stored.append(t_current)
            C_stored.append(C.copy())

    return DiffusionResult(
        x_grid=x_grid,
        t_grid=np.array(t_stored),
        C_field=np.array(C_stored),
        sigma_h_field=sigma_h,
    )


def get_concentration_at_depth(result: DiffusionResult, depth_m: float, time_s: float) -> float:
    """
    Interpolate hydrogen concentration at given depth and time.

    Parameters
    ----------
    result : DiffusionResult
    depth_m : float
        Depth from inner surface [m].
    time_s : float
        Time since hydrogen exposure [s].

    Returns
    -------
    C_H : float
        Hydrogen concentration [mol/m3].
    """
    # Spatial interpolation
    x_idx = np.searchsorted(result.x_grid, depth_m)
    x_idx = min(x_idx, len(result.x_grid) - 1)

    # Temporal interpolation
    t_idx = np.searchsorted(result.t_grid, time_s)
    t_idx = min(t_idx, len(result.t_grid) - 1)

    if x_idx == 0 or t_idx == 0:
        return result.C_field[t_idx, x_idx]

    # Bilinear interpolation
    x0 = result.x_grid[max(x_idx - 1, 0)]
    x1 = result.x_grid[x_idx]
    t0 = result.t_grid[max(t_idx - 1, 0)]
    t1 = result.t_grid[t_idx]

    if x1 == x0 or t1 == t0:
        return result.C_field[t_idx, x_idx]

    fx = (depth_m - x0) / (x1 - x0)
    ft = (time_s - t0) / (t1 - t0)
    fx = np.clip(fx, 0.0, 1.0)
    ft = np.clip(ft, 0.0, 1.0)

    C00 = result.C_field[max(t_idx - 1, 0), max(x_idx - 1, 0)]
    C01 = result.C_field[max(t_idx - 1, 0), x_idx]
    C10 = result.C_field[t_idx, max(x_idx - 1, 0)]
    C11 = result.C_field[t_idx, x_idx]

    C = (C00 * (1 - fx) * (1 - ft) + C01 * fx * (1 - ft)
         + C10 * (1 - fx) * ft + C11 * fx * ft)

    return max(C, 0.0)
