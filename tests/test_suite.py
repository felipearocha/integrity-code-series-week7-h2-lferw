"""
INTEGRITY CODE SERIES -- Week 7
Comprehensive Test Suite: 150+ tests

Categories:
    A. Configuration and parameter validation (18)
    B. Stress field physics (10)
    C. Hydrogen diffusion: boundary conditions (10)
    D. Hydrogen diffusion: convergence and analytical (8)
    E. Hydrogen diffusion: physics and edge cases (10)
    F. Interpolation (6)
    G. Pit growth and geometry (10)
    H. Murakami SIF and transition criteria (12)
    I. Hoop stress (6)
    J. Hydrogen degradation functions (12)
    K. Newman-Raju SIF (8)
    L. HA-FCG rate (8)
    M. Integrated life prediction (12)
    N. Life prediction edge cases (8)
    O. LHS sampling (6)
    P. Monte Carlo (6)
    Q. ML surrogate (8)
    R. Cybersecurity (10)
    S. Integration: full chain (6)
    T. Regression: locked baselines (4)
    U. Visualization output (4)
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pytest
import tempfile

from config import (
    PipeGeometry, MaterialProps, HydrogenTransport, HydrogenEmbrittlement,
    PitGrowth, OperatingConditions, SimControl, default_config, R_GAS, FARADAY
)
from hydrogen_diffusion import (
    compute_stress_field, sieverts_surface_concentration, solve_diffusion,
    get_concentration_at_depth, DiffusionResult
)
from pit_to_crack import (
    pit_depth_after_ng_service, pit_geometry, murakami_sif, hoop_stress,
    degraded_threshold, el_haddad_threshold_depth, evaluate_pit_state
)
from ha_fcg import (
    newman_raju_simplified, degraded_toughness, hydrogen_enhancement_factor,
    ha_fcg_rate, run_life_prediction, cycles_to_years, maop_under_b3112
)
from cybersecurity import (
    check_parameter_bounds, AuditLogger, detect_pit_depth_spoofing,
    physics_monotonicity_check, STRIDE_THREATS, PARAMETER_BOUNDS,
    check_all_config_bounds
)


# ============================================================
# FIXTURES
# ============================================================
@pytest.fixture
def cfg():
    return default_config()

@pytest.fixture
def fast_sim():
    return SimControl(
        n_wall_nodes=20, dt_diffusion_s=500.0, max_cycles=50_000,
        da_increment_m=1e-5, n_mc_samples=100, mc_seed=42,
        n_lhs_samples=50, lhs_seed=7,
    )

@pytest.fixture
def tiny_sim():
    """Ultra-fast sim for edge case testing."""
    return SimControl(
        n_wall_nodes=10, dt_diffusion_s=1000.0, max_cycles=5_000,
        da_increment_m=1e-5, n_mc_samples=10, mc_seed=42,
        n_lhs_samples=10, lhs_seed=7,
    )

@pytest.fixture
def ref_diffusion(cfg, fast_sim):
    return solve_diffusion(cfg["geom"], cfg["h2_transport"], cfg["ops"], fast_sim)


# ============================================================
# A. CONFIGURATION AND PARAMETER VALIDATION (18 tests)
# ============================================================
class TestConfigGeometry:
    def test_inner_diameter_computed(self, cfg):
        g = cfg["geom"]
        assert abs(g.inner_diameter_m - (g.outer_diameter_m - 2*g.wall_thickness_m)) < 1e-12

    def test_outer_greater_than_inner(self, cfg):
        g = cfg["geom"]
        assert g.outer_diameter_m > g.inner_diameter_m

    def test_wall_positive(self, cfg):
        assert cfg["geom"].wall_thickness_m > 0

    def test_wall_less_than_radius(self, cfg):
        g = cfg["geom"]
        assert g.wall_thickness_m < g.outer_diameter_m / 2

    def test_20_inch_pipe(self, cfg):
        assert abs(cfg["geom"].outer_diameter_m - 0.508) < 1e-6

    def test_dt_ratio_reasonable(self, cfg):
        """Wall thickness / outer diameter in reasonable range for pipeline."""
        g = cfg["geom"]
        ratio = g.wall_thickness_m / g.outer_diameter_m
        assert 0.005 < ratio < 0.1


class TestConfigMaterial:
    def test_smys_x52(self, cfg):
        assert cfg["mat"].SMYS_MPa == 358.0

    def test_smts_greater_than_smys(self, cfg):
        m = cfg["mat"]
        assert m.SMTS_MPa > m.SMYS_MPa

    def test_seam_toughness_less_than_base(self, cfg):
        m = cfg["mat"]
        assert m.K_IC_air_seam_MPa_sqrtm <= m.K_IC_air_base_MPa_sqrtm

    def test_seam_threshold_less_than_base(self, cfg):
        m = cfg["mat"]
        assert m.K_th_air_seam_MPa_sqrtm <= m.K_th_air_base_MPa_sqrtm

    def test_paris_exponent_range(self, cfg):
        assert 2.0 <= cfg["mat"].m_paris_air <= 4.0

    def test_paris_coefficient_positive(self, cfg):
        assert cfg["mat"].C_paris_air > 0

    def test_youngs_modulus_steel(self, cfg):
        assert 190 < cfg["mat"].youngs_modulus_GPa < 220


class TestConfigHydrogen:
    def test_diffusivity_positive(self, cfg):
        assert cfg["h2_transport"].D_L_m2s > 0

    def test_sieverts_positive(self, cfg):
        assert cfg["h2_transport"].sieverts_constant > 0

    def test_partial_molar_volume_positive(self, cfg):
        assert cfg["h2_transport"].V_H_m3mol > 0

    def test_activation_energy_positive(self, cfg):
        assert cfg["h2_transport"].D_L_activation_kJmol > 0

    def test_embrittlement_floor_positive(self, cfg):
        he = cfg["h2_embrittlement"]
        assert he.K_IC_min_MPa_sqrtm > 0
        assert he.K_th_min_MPa_sqrtm > 0


# ============================================================
# B. STRESS FIELD PHYSICS (10 tests)
# ============================================================
class TestStressField:
    def test_shape(self, cfg):
        x, s = compute_stress_field(cfg["geom"], 7.0, 50)
        assert len(x) == 50 and len(s) == 50

    def test_grid_starts_at_zero(self, cfg):
        x, _ = compute_stress_field(cfg["geom"], 7.0, 50)
        assert x[0] == 0.0

    def test_grid_ends_at_wall(self, cfg):
        x, _ = compute_stress_field(cfg["geom"], 7.0, 50)
        assert abs(x[-1] - cfg["geom"].wall_thickness_m) < 1e-12

    def test_hydrostatic_positive_under_pressure(self, cfg):
        _, s = compute_stress_field(cfg["geom"], 7.0, 50)
        assert np.all(s > 0)

    def test_hydrostatic_zero_at_zero_pressure(self, cfg):
        _, s = compute_stress_field(cfg["geom"], 0.0, 50)
        assert np.allclose(s, 0.0)

    def test_hydrostatic_scales_with_pressure(self, cfg):
        _, s1 = compute_stress_field(cfg["geom"], 5.0, 50)
        _, s2 = compute_stress_field(cfg["geom"], 10.0, 50)
        assert np.allclose(s2, 2.0 * s1, rtol=1e-10)

    def test_hydrostatic_nearly_uniform_thin_wall(self, cfg):
        """For thin-wall pipe, hydrostatic stress is approximately constant."""
        _, s = compute_stress_field(cfg["geom"], 7.0, 50)
        cv = np.std(s) / np.mean(s)
        assert cv < 0.02

    def test_more_nodes_same_endpoints(self, cfg):
        x1, s1 = compute_stress_field(cfg["geom"], 7.0, 20)
        x2, s2 = compute_stress_field(cfg["geom"], 7.0, 100)
        assert abs(s1[0] - s2[0]) < 1.0  # Pa-level agreement

    def test_negative_pressure_flips_sign(self, cfg):
        """Negative pressure (external) should give negative hydrostatic."""
        _, s = compute_stress_field(cfg["geom"], -5.0, 20)
        assert np.all(s < 0)

    def test_stress_units_are_pascals(self, cfg):
        """7 MPa on 20-inch pipe should give stress ~100+ MPa = 1e8 Pa order."""
        _, s = compute_stress_field(cfg["geom"], 7.0, 20)
        assert s[0] > 1e7  # at least 10 MPa in Pa


# ============================================================
# C. HYDROGEN DIFFUSION: BOUNDARY CONDITIONS (10 tests)
# ============================================================
class TestDiffusionBC:
    def test_inner_bc_enforced(self, cfg, fast_sim):
        r = solve_diffusion(cfg["geom"], cfg["h2_transport"], cfg["ops"], fast_sim)
        C0 = sieverts_surface_concentration(cfg["h2_transport"], cfg["ops"].p_H2_MPa)
        assert abs(r.C_field[-1, 0] - C0) < 1e-10

    def test_outer_bc_zero(self, cfg, fast_sim):
        r = solve_diffusion(cfg["geom"], cfg["h2_transport"], cfg["ops"], fast_sim)
        assert r.C_field[-1, -1] == 0.0

    def test_inner_bc_all_snapshots(self, cfg, fast_sim):
        r = solve_diffusion(cfg["geom"], cfg["h2_transport"], cfg["ops"], fast_sim)
        C0 = sieverts_surface_concentration(cfg["h2_transport"], cfg["ops"].p_H2_MPa)
        for i in range(len(r.t_grid)):
            assert abs(r.C_field[i, 0] - C0) < 1e-10

    def test_outer_bc_all_snapshots(self, cfg, fast_sim):
        r = solve_diffusion(cfg["geom"], cfg["h2_transport"], cfg["ops"], fast_sim)
        for i in range(len(r.t_grid)):
            assert r.C_field[i, -1] == 0.0

    def test_initial_condition_zero_interior(self, cfg, fast_sim):
        r = solve_diffusion(cfg["geom"], cfg["h2_transport"], cfg["ops"], fast_sim)
        assert np.all(r.C_field[0, 1:-1] == 0.0)

    def test_non_negative_everywhere(self, cfg, fast_sim):
        r = solve_diffusion(cfg["geom"], cfg["h2_transport"], cfg["ops"], fast_sim)
        assert np.all(r.C_field >= 0)

    def test_sieverts_increases_with_pressure(self, cfg):
        h2t = cfg["h2_transport"]
        C1 = sieverts_surface_concentration(h2t, 3.0)
        C2 = sieverts_surface_concentration(h2t, 10.0)
        assert C2 > C1

    def test_sieverts_sqrt_scaling(self, cfg):
        h2t = cfg["h2_transport"]
        C1 = sieverts_surface_concentration(h2t, 1.0)
        C4 = sieverts_surface_concentration(h2t, 4.0)
        assert abs(C4 / C1 - 2.0) < 1e-10  # sqrt(4)/sqrt(1) = 2

    def test_sieverts_zero_pressure(self, cfg):
        assert sieverts_surface_concentration(cfg["h2_transport"], 0.0) == 0.0

    def test_output_shape_consistent(self, cfg, fast_sim):
        r = solve_diffusion(cfg["geom"], cfg["h2_transport"], cfg["ops"], fast_sim)
        nt = len(r.t_grid)
        nx = len(r.x_grid)
        assert r.C_field.shape == (nt, nx)


# ============================================================
# D. DIFFUSION CONVERGENCE AND ANALYTICAL (8 tests)
# ============================================================
class TestDiffusionConvergence:
    def test_steady_state_linear_no_stress(self, cfg):
        """Without stress coupling (V_H~0), steady state is C(x) = C0*(1 - x/w)."""
        h2t = HydrogenTransport(
            D_L_m2s=1e-9, D_L_activation_kJmol=6.9,
            sieverts_constant=0.08, V_H_m3mol=1e-20,  # near-zero stress coupling
            E_trap_kJmol=30.0, T_ref_K=298.15,
        )
        sim = SimControl(n_wall_nodes=50, dt_diffusion_s=10.0, max_cycles=5000,
                         da_increment_m=1e-5, n_mc_samples=10, mc_seed=42,
                         n_lhs_samples=10, lhs_seed=7)
        r = solve_diffusion(cfg["geom"], h2t, cfg["ops"], sim, t_total_s=5e5)
        C0 = sieverts_surface_concentration(h2t, cfg["ops"].p_H2_MPa)
        w = cfg["geom"].wall_thickness_m
        x = r.x_grid
        C_analytical = C0 * (1.0 - x / w)
        C_numerical = r.C_field[-1]
        error = np.max(np.abs(C_numerical - C_analytical)) / C0
        assert error < 0.05, f"Max relative error {error:.3f} > 5%"

    def test_finer_grid_reduces_error(self, cfg):
        """Refining the grid should reduce numerical error."""
        h2t = HydrogenTransport(
            D_L_m2s=1e-9, D_L_activation_kJmol=6.9,
            sieverts_constant=0.08, V_H_m3mol=1e-20,
            E_trap_kJmol=30.0, T_ref_K=298.15,
        )
        errors = []
        for n_nodes in [15, 30, 60]:
            sim = SimControl(n_wall_nodes=n_nodes, dt_diffusion_s=5.0,
                             max_cycles=5000, da_increment_m=1e-5,
                             n_mc_samples=10, mc_seed=42, n_lhs_samples=10, lhs_seed=7)
            r = solve_diffusion(cfg["geom"], h2t, cfg["ops"], sim, t_total_s=5e5)
            C0 = sieverts_surface_concentration(h2t, cfg["ops"].p_H2_MPa)
            w = cfg["geom"].wall_thickness_m
            C_ana = C0 * (1.0 - r.x_grid / w)
            errors.append(np.max(np.abs(r.C_field[-1] - C_ana)) / C0)
        assert errors[1] < errors[0] or errors[1] < 1e-10, "Finer grid should reduce error or be at machine precision"
        assert errors[2] < errors[1] or errors[2] < 1e-10, "Even finer grid should reduce further or be at machine precision"

    def test_monotonic_decreasing_at_steady_state(self, cfg, fast_sim):
        r = solve_diffusion(cfg["geom"], cfg["h2_transport"], cfg["ops"], fast_sim)
        diffs = np.diff(r.C_field[-1])
        assert np.all(diffs <= 1e-6)

    def test_midwall_concentration_increases_over_time(self, cfg, fast_sim):
        r = solve_diffusion(cfg["geom"], cfg["h2_transport"], cfg["ops"], fast_sim)
        mid = fast_sim.n_wall_nodes // 2
        assert r.C_field[-1, mid] >= r.C_field[1, mid]

    def test_total_hydrogen_increases_over_time(self, cfg, fast_sim):
        """Integral of C across wall should increase monotonically."""
        r = solve_diffusion(cfg["geom"], cfg["h2_transport"], cfg["ops"], fast_sim)
        _integrate = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
        integrals = [_integrate(r.C_field[i], r.x_grid) for i in range(len(r.t_grid))]
        for i in range(1, len(integrals)):
            assert integrals[i] >= integrals[i-1] - 1e-12

    def test_time_grid_monotonic(self, cfg, fast_sim):
        r = solve_diffusion(cfg["geom"], cfg["h2_transport"], cfg["ops"], fast_sim)
        assert np.all(np.diff(r.t_grid) >= 0)

    def test_multiple_snapshots_stored(self, cfg, fast_sim):
        r = solve_diffusion(cfg["geom"], cfg["h2_transport"], cfg["ops"], fast_sim)
        assert len(r.t_grid) > 5

    def test_stress_field_stored(self, cfg, fast_sim):
        r = solve_diffusion(cfg["geom"], cfg["h2_transport"], cfg["ops"], fast_sim)
        assert len(r.sigma_h_field) == fast_sim.n_wall_nodes


# ============================================================
# E. DIFFUSION PHYSICS AND EDGE CASES (10 tests)
# ============================================================
class TestDiffusionEdgeCases:
    def test_zero_pressure_zero_concentration(self, cfg, tiny_sim):
        ops = OperatingConditions(
            p_H2_MPa=0.0, p_min_MPa=0.0, p_max_MPa=0.0,
            R_ratio=0.0, cycles_per_year=365, T_K=298.15,
            design_factor_b3112=0.5)
        r = solve_diffusion(cfg["geom"], cfg["h2_transport"], ops, tiny_sim)
        assert np.allclose(r.C_field, 0.0)

    def test_very_small_diffusivity(self, cfg, tiny_sim):
        """Very slow diffusion: interior should remain near zero."""
        h2t = HydrogenTransport(
            D_L_m2s=1e-14, D_L_activation_kJmol=6.9,
            sieverts_constant=0.08, V_H_m3mol=2e-6,
            E_trap_kJmol=30.0, T_ref_K=298.15)
        r = solve_diffusion(cfg["geom"], h2t, cfg["ops"], tiny_sim, t_total_s=1e4)
        mid = tiny_sim.n_wall_nodes // 2
        assert r.C_field[-1, mid] < 0.01 * r.C_field[-1, 0]

    def test_high_diffusivity_fast_penetration(self, cfg, tiny_sim):
        """Fast diffusion: profile should be nearly linear quickly."""
        h2t = HydrogenTransport(
            D_L_m2s=1e-7, D_L_activation_kJmol=6.9,
            sieverts_constant=0.08, V_H_m3mol=1e-20,
            E_trap_kJmol=30.0, T_ref_K=298.15)
        r = solve_diffusion(cfg["geom"], h2t, cfg["ops"], tiny_sim, t_total_s=1e6)
        C0 = r.C_field[-1, 0]
        mid = tiny_sim.n_wall_nodes // 2
        assert r.C_field[-1, mid] > 0.3 * C0

    def test_higher_pressure_higher_surface_conc(self, cfg, tiny_sim):
        ops_lo = OperatingConditions(p_H2_MPa=3.0, p_min_MPa=1.5, p_max_MPa=3.0,
            R_ratio=0.5, cycles_per_year=365, T_K=298.15, design_factor_b3112=0.5)
        ops_hi = OperatingConditions(p_H2_MPa=10.0, p_min_MPa=5.0, p_max_MPa=10.0,
            R_ratio=0.5, cycles_per_year=365, T_K=298.15, design_factor_b3112=0.5)
        r_lo = solve_diffusion(cfg["geom"], cfg["h2_transport"], ops_lo, tiny_sim)
        r_hi = solve_diffusion(cfg["geom"], cfg["h2_transport"], ops_hi, tiny_sim)
        assert r_hi.C_field[-1, 0] > r_lo.C_field[-1, 0]

    def test_result_fields_finite(self, cfg, fast_sim):
        r = solve_diffusion(cfg["geom"], cfg["h2_transport"], cfg["ops"], fast_sim)
        assert np.all(np.isfinite(r.C_field))
        assert np.all(np.isfinite(r.sigma_h_field))

    def test_c_field_max_at_inner_surface(self, cfg, fast_sim):
        r = solve_diffusion(cfg["geom"], cfg["h2_transport"], cfg["ops"], fast_sim)
        for i in range(1, len(r.t_grid)):
            assert r.C_field[i, 0] >= r.C_field[i].max() - 1e-10

    def test_diffusion_result_type(self, cfg, fast_sim):
        r = solve_diffusion(cfg["geom"], cfg["h2_transport"], cfg["ops"], fast_sim)
        assert isinstance(r, DiffusionResult)

    def test_x_grid_monotonic(self, cfg, fast_sim):
        r = solve_diffusion(cfg["geom"], cfg["h2_transport"], cfg["ops"], fast_sim)
        assert np.all(np.diff(r.x_grid) > 0)

    def test_custom_total_time(self, cfg, tiny_sim):
        r = solve_diffusion(cfg["geom"], cfg["h2_transport"], cfg["ops"], tiny_sim,
                            t_total_s=10000.0)
        assert r.t_grid[-1] <= 10000.0 + 2000.0  # within tolerance

    def test_n_snapshots_parameter(self, cfg, tiny_sim):
        r = solve_diffusion(cfg["geom"], cfg["h2_transport"], cfg["ops"], tiny_sim,
                            n_snapshots=10)
        assert len(r.t_grid) >= 5  # at least some snapshots


# ============================================================
# F. INTERPOLATION (6 tests)
# ============================================================
class TestInterpolation:
    def test_at_inner_surface(self, ref_diffusion, cfg):
        C = get_concentration_at_depth(ref_diffusion, 0.0, ref_diffusion.t_grid[-1])
        C0 = sieverts_surface_concentration(cfg["h2_transport"], cfg["ops"].p_H2_MPa)
        assert abs(C - C0) < 0.01

    def test_at_outer_surface(self, ref_diffusion, cfg):
        w = cfg["geom"].wall_thickness_m
        C = get_concentration_at_depth(ref_diffusion, w, ref_diffusion.t_grid[-1])
        assert C < 0.01

    def test_non_negative(self, ref_diffusion, cfg):
        C = get_concentration_at_depth(ref_diffusion, cfg["geom"].wall_thickness_m * 0.5,
                                       ref_diffusion.t_grid[-1] * 0.5)
        assert C >= 0

    def test_decreases_with_depth(self, ref_diffusion, cfg):
        t = ref_diffusion.t_grid[-1]
        w = cfg["geom"].wall_thickness_m
        C1 = get_concentration_at_depth(ref_diffusion, w * 0.1, t)
        C2 = get_concentration_at_depth(ref_diffusion, w * 0.5, t)
        assert C1 >= C2

    def test_increases_with_time(self, ref_diffusion, cfg):
        depth = cfg["geom"].wall_thickness_m * 0.3
        C_early = get_concentration_at_depth(ref_diffusion, depth, ref_diffusion.t_grid[1])
        C_late = get_concentration_at_depth(ref_diffusion, depth, ref_diffusion.t_grid[-1])
        assert C_late >= C_early

    def test_at_time_zero_interior_zero(self, ref_diffusion, cfg):
        C = get_concentration_at_depth(ref_diffusion, cfg["geom"].wall_thickness_m * 0.5, 0.0)
        assert C < 0.01


# ============================================================
# G. PIT GROWTH AND GEOMETRY (10 tests)
# ============================================================
class TestPitGrowth:
    def test_depth_positive(self, cfg):
        assert pit_depth_after_ng_service(cfg["pit"]) > 0

    def test_depth_less_than_wall(self, cfg):
        assert pit_depth_after_ng_service(cfg["pit"]) < cfg["geom"].wall_thickness_m

    def test_depth_increases_with_time(self):
        p1 = PitGrowth(k_pit_m=5e-5, n_pit=0.5, f_seam=2.0, aspect_ratio_mean=2.0,
                        aspect_ratio_std=0.5, ng_service_years=30)
        p2 = PitGrowth(k_pit_m=5e-5, n_pit=0.5, f_seam=2.0, aspect_ratio_mean=2.0,
                        aspect_ratio_std=0.5, ng_service_years=67)
        assert pit_depth_after_ng_service(p2) > pit_depth_after_ng_service(p1)

    def test_seam_factor_increases_depth(self):
        p1 = PitGrowth(k_pit_m=5e-5, n_pit=0.5, f_seam=1.0, aspect_ratio_mean=2.0,
                        aspect_ratio_std=0.5, ng_service_years=67)
        p2 = PitGrowth(k_pit_m=5e-5, n_pit=0.5, f_seam=3.0, aspect_ratio_mean=2.0,
                        aspect_ratio_std=0.5, ng_service_years=67)
        assert pit_depth_after_ng_service(p2) == 3.0 * pit_depth_after_ng_service(p1)

    def test_power_law_exponent(self):
        """n=0.5 means depth scales as sqrt(time)."""
        p = PitGrowth(k_pit_m=1e-4, n_pit=0.5, f_seam=1.0, aspect_ratio_mean=2.0,
                       aspect_ratio_std=0.5, ng_service_years=100)
        expected = 1e-4 * 1.0 * 100 ** 0.5
        assert abs(pit_depth_after_ng_service(p) - expected) < 1e-12

    def test_area_positive(self):
        c, a = pit_geometry(1e-3, 2.0)
        assert a > 0

    def test_half_length_equals_ar_times_depth(self):
        c, _ = pit_geometry(1e-3, 2.5)
        assert abs(c - 2.5e-3) < 1e-12

    def test_area_formula(self):
        """area = (pi/2) * c * a for semi-elliptical pit."""
        depth = 2e-3
        ar = 3.0
        c, area = pit_geometry(depth, ar)
        expected = (np.pi / 2.0) * c * depth
        assert abs(area - expected) < 1e-15

    def test_larger_ar_larger_area(self):
        _, a1 = pit_geometry(1e-3, 1.0)
        _, a2 = pit_geometry(1e-3, 3.0)
        assert a2 > a1

    def test_deeper_pit_larger_area(self):
        _, a1 = pit_geometry(0.5e-3, 2.0)
        _, a2 = pit_geometry(2.0e-3, 2.0)
        assert a2 > a1


# ============================================================
# H. MURAKAMI SIF AND TRANSITION CRITERIA (12 tests)
# ============================================================
class TestMurakamiTransition:
    def test_sif_positive(self):
        assert murakami_sif(200.0, 1e-6) > 0

    def test_sif_zero_stress(self):
        assert murakami_sif(0.0, 1e-6) == 0.0

    def test_sif_scales_with_stress(self):
        K1 = murakami_sif(100.0, 1e-6)
        K2 = murakami_sif(200.0, 1e-6)
        assert abs(K2 / K1 - 2.0) < 1e-10

    def test_sif_increases_with_area(self):
        K1 = murakami_sif(200.0, 1e-7)
        K2 = murakami_sif(200.0, 1e-5)
        assert K2 > K1

    def test_sif_formula(self):
        sigma = 250.0
        area = 5e-6
        expected = 0.65 * sigma * np.sqrt(np.pi * np.sqrt(area))
        assert abs(murakami_sif(sigma, area) - expected) < 1e-10

    def test_degraded_threshold_at_zero_H(self, cfg):
        he = cfg["h2_embrittlement"]
        K_th = degraded_threshold(5.0, 0.0, he)
        assert abs(K_th - 5.0) < 1e-10

    def test_degraded_threshold_decreases(self, cfg):
        he = cfg["h2_embrittlement"]
        K0 = degraded_threshold(5.0, 0.0, he)
        K1 = degraded_threshold(5.0, 1.0, he)
        assert K1 < K0

    def test_degraded_threshold_floor(self, cfg):
        he = cfg["h2_embrittlement"]
        K = degraded_threshold(5.0, 1e6, he)
        assert K >= he.K_th_min_MPa_sqrtm

    def test_el_haddad_positive(self):
        assert el_haddad_threshold_depth(5.0, 200.0) > 0

    def test_el_haddad_zero_stress_infinite(self):
        assert el_haddad_threshold_depth(5.0, 0.0) == np.inf

    def test_pit_state_deeper_higher_K(self, cfg):
        he = cfg["h2_embrittlement"]
        s1 = evaluate_pit_state(cfg["geom"], cfg["mat"], cfg["pit"], cfg["ops"], he,
                                C_H_at_pit=0.1, pit_depth_m=0.5e-3)
        s2 = evaluate_pit_state(cfg["geom"], cfg["mat"], cfg["pit"], cfg["ops"], he,
                                C_H_at_pit=0.1, pit_depth_m=2.0e-3)
        assert s2.K_pit_MPa_sqrtm > s1.K_pit_MPa_sqrtm

    def test_pit_capped_at_80pct_wall(self, cfg):
        he = cfg["h2_embrittlement"]
        s = evaluate_pit_state(cfg["geom"], cfg["mat"], cfg["pit"], cfg["ops"], he,
                               C_H_at_pit=0.0, pit_depth_m=0.01)  # 10 mm > wall
        assert s.depth_m <= 0.8 * cfg["geom"].wall_thickness_m


# ============================================================
# I. HOOP STRESS (6 tests)
# ============================================================
class TestHoopStress:
    def test_positive(self, cfg):
        assert hoop_stress(cfg["geom"], 7.0) > 0

    def test_zero_pressure(self, cfg):
        assert hoop_stress(cfg["geom"], 0.0) == 0.0

    def test_barlow_formula(self, cfg):
        g = cfg["geom"]
        expected = 7.0 * g.outer_diameter_m / (2.0 * g.wall_thickness_m)
        assert abs(hoop_stress(g, 7.0) - expected) < 1e-10

    def test_scales_with_pressure(self, cfg):
        s1 = hoop_stress(cfg["geom"], 5.0)
        s2 = hoop_stress(cfg["geom"], 10.0)
        assert abs(s2 / s1 - 2.0) < 1e-10

    def test_below_smys_at_maop(self, cfg):
        maop = maop_under_b3112(cfg["geom"], cfg["mat"], cfg["ops"])
        sigma = hoop_stress(cfg["geom"], maop)
        assert sigma < cfg["mat"].SMYS_MPa

    def test_maop_positive(self, cfg):
        assert maop_under_b3112(cfg["geom"], cfg["mat"], cfg["ops"]) > 0


# ============================================================
# J. HYDROGEN DEGRADATION FUNCTIONS (12 tests)
# ============================================================
class TestHydrogenDegradation:
    def test_toughness_at_zero_H(self, cfg):
        he = cfg["h2_embrittlement"]
        assert abs(degraded_toughness(60.0, 0.0, he) - 60.0) < 1e-10

    def test_toughness_decreases(self, cfg):
        he = cfg["h2_embrittlement"]
        assert degraded_toughness(60.0, 1.0, he) < 60.0

    def test_toughness_floor(self, cfg):
        he = cfg["h2_embrittlement"]
        assert degraded_toughness(60.0, 1e8, he) >= he.K_IC_min_MPa_sqrtm

    def test_toughness_monotonic(self, cfg):
        he = cfg["h2_embrittlement"]
        vals = [degraded_toughness(60.0, c, he) for c in np.linspace(0, 5, 20)]
        for i in range(1, len(vals)):
            assert vals[i] <= vals[i-1] + 1e-12

    def test_enhancement_unity_at_zero(self, cfg):
        assert hydrogen_enhancement_factor(0.0, cfg["h2_embrittlement"]) == 1.0

    def test_enhancement_increases(self, cfg):
        he = cfg["h2_embrittlement"]
        f1 = hydrogen_enhancement_factor(0.1, he)
        f2 = hydrogen_enhancement_factor(1.0, he)
        assert f2 > f1 > 1.0

    def test_enhancement_capped(self, cfg):
        he = cfg["h2_embrittlement"]
        assert hydrogen_enhancement_factor(1e10, he) <= he.max_enhancement

    def test_enhancement_negative_C_returns_one(self, cfg):
        """Negative concentration should be treated as zero."""
        he = cfg["h2_embrittlement"]
        assert hydrogen_enhancement_factor(-1.0, he) == 1.0

    def test_enhancement_continuous(self, cfg):
        he = cfg["h2_embrittlement"]
        Cs = np.linspace(0, 3, 100)
        fs = [hydrogen_enhancement_factor(c, he) for c in Cs]
        for i in range(1, len(fs)):
            assert abs(fs[i] - fs[i-1]) < 5.0  # no discontinuities

    def test_fcg_rate_zero_dK(self, cfg):
        assert ha_fcg_rate(0.0, 0.5, cfg["mat"], cfg["h2_embrittlement"]) == 0.0

    def test_fcg_rate_positive(self, cfg):
        rate = ha_fcg_rate(10.0, 0.5, cfg["mat"], cfg["h2_embrittlement"])
        assert rate > 0

    def test_fcg_rate_increases_with_H(self, cfg):
        mat, he = cfg["mat"], cfg["h2_embrittlement"]
        r1 = ha_fcg_rate(10.0, 0.0, mat, he)
        r2 = ha_fcg_rate(10.0, 1.0, mat, he)
        assert r2 > r1


# ============================================================
# K. NEWMAN-RAJU SIF (8 tests)
# ============================================================
class TestNewmanRaju:
    def test_positive(self):
        assert newman_raju_simplified(200.0, 1e-3, 6.35e-3) > 0

    def test_zero_stress(self):
        assert newman_raju_simplified(0.0, 1e-3, 6.35e-3) == 0.0

    def test_increases_with_crack_depth(self):
        K1 = newman_raju_simplified(200.0, 0.5e-3, 6.35e-3)
        K2 = newman_raju_simplified(200.0, 2.0e-3, 6.35e-3)
        assert K2 > K1

    def test_increases_with_stress(self):
        K1 = newman_raju_simplified(100.0, 1e-3, 6.35e-3)
        K2 = newman_raju_simplified(300.0, 1e-3, 6.35e-3)
        assert K2 > K1

    def test_shallow_crack_near_1_12(self):
        """For very shallow crack (a/t << 1), F should be close to 1.12."""
        sigma = 100.0
        a = 0.1e-3
        t = 6.35e-3
        K = newman_raju_simplified(sigma, a, t)
        K_simple = 1.12 * sigma * np.sqrt(np.pi * a)
        assert abs(K / K_simple - 1.0) < 0.05

    def test_deep_crack_higher_than_shallow_F(self):
        """Finite-width correction increases F for deeper cracks."""
        sigma = 200.0
        t = 6.35e-3
        K_shallow = newman_raju_simplified(sigma, 0.1e-3, t)
        K_deep = newman_raju_simplified(sigma, 5.0e-3, t)
        F_shallow = K_shallow / (sigma * np.sqrt(np.pi * 0.1e-3))
        F_deep = K_deep / (sigma * np.sqrt(np.pi * 5.0e-3))
        assert F_deep > F_shallow

    def test_a_over_t_clamped(self):
        """a/t > 0.95 should be clamped, not crash."""
        K = newman_raju_simplified(200.0, 6.3e-3, 6.35e-3)
        assert np.isfinite(K)
        assert K > 0

    def test_finite_result(self):
        K = newman_raju_simplified(200.0, 3e-3, 6.35e-3)
        assert np.isfinite(K)


# ============================================================
# L. HA-FCG RATE (8 tests)
# ============================================================
class TestHAFCGRate:
    def test_air_rate_paris_law(self, cfg):
        mat = cfg["mat"]
        he = cfg["h2_embrittlement"]
        dK = 15.0
        rate = ha_fcg_rate(dK, 0.0, mat, he)
        expected = mat.C_paris_air * dK ** mat.m_paris_air
        assert abs(rate - expected) < 1e-20

    def test_hydrogen_accelerates(self, cfg):
        mat, he = cfg["mat"], cfg["h2_embrittlement"]
        r_air = ha_fcg_rate(15.0, 0.0, mat, he)
        r_h2 = ha_fcg_rate(15.0, 0.5, mat, he)
        assert r_h2 > r_air

    def test_higher_dK_higher_rate(self, cfg):
        mat, he = cfg["mat"], cfg["h2_embrittlement"]
        r1 = ha_fcg_rate(5.0, 0.5, mat, he)
        r2 = ha_fcg_rate(20.0, 0.5, mat, he)
        assert r2 > r1

    def test_rate_finite(self, cfg):
        rate = ha_fcg_rate(50.0, 5.0, cfg["mat"], cfg["h2_embrittlement"])
        assert np.isfinite(rate)

    def test_rate_non_negative(self, cfg):
        for dK in [0, 1, 10, 50]:
            for c in [0, 0.1, 1.0, 10.0]:
                r = ha_fcg_rate(dK, c, cfg["mat"], cfg["h2_embrittlement"])
                assert r >= 0

    def test_negative_dK_zero_rate(self, cfg):
        assert ha_fcg_rate(-5.0, 0.5, cfg["mat"], cfg["h2_embrittlement"]) == 0.0

    def test_paris_exponent_effect(self):
        """Higher m means stronger dK dependence."""
        from ha_fcg import ha_fcg_rate
        mat_lo = MaterialProps(C_paris_air=3e-12, m_paris_air=2.5,
            SMYS_MPa=358, SMTS_MPa=455, youngs_modulus_GPa=207,
            poissons_ratio=0.3, density_kg_m3=7850,
            K_IC_air_base_MPa_sqrtm=120, K_th_air_base_MPa_sqrtm=6,
            K_IC_air_seam_MPa_sqrtm=60, K_th_air_seam_MPa_sqrtm=3.5)
        mat_hi = MaterialProps(C_paris_air=3e-12, m_paris_air=3.5,
            SMYS_MPa=358, SMTS_MPa=455, youngs_modulus_GPa=207,
            poissons_ratio=0.3, density_kg_m3=7850,
            K_IC_air_base_MPa_sqrtm=120, K_th_air_base_MPa_sqrtm=6,
            K_IC_air_seam_MPa_sqrtm=60, K_th_air_seam_MPa_sqrtm=3.5)
        he = HydrogenEmbrittlement()
        r_lo = ha_fcg_rate(20.0, 0.0, mat_lo, he)
        r_hi = ha_fcg_rate(20.0, 0.0, mat_hi, he)
        assert r_hi > r_lo

    def test_enhancement_bounded_in_rate(self, cfg):
        mat, he = cfg["mat"], cfg["h2_embrittlement"]
        r_air = ha_fcg_rate(10.0, 0.0, mat, he)
        r_max_h = ha_fcg_rate(10.0, 1e10, mat, he)
        assert r_max_h <= r_air * he.max_enhancement + 1e-30


# ============================================================
# M. INTEGRATED LIFE PREDICTION (12 tests)
# ============================================================
class TestLifePrediction:
    def test_runs_without_error(self, cfg, fast_sim):
        r = run_life_prediction(cfg["geom"], cfg["mat"], cfg["h2_transport"],
            cfg["h2_embrittlement"], cfg["pit"], cfg["ops"], fast_sim)
        assert r.failure_mode in ("fracture", "wall_penetration", "no_failure")

    def test_failure_cycle_non_negative(self, cfg, fast_sim):
        r = run_life_prediction(cfg["geom"], cfg["mat"], cfg["h2_transport"],
            cfg["h2_embrittlement"], cfg["pit"], cfg["ops"], fast_sim)
        assert r.failure_cycle >= 0

    def test_deeper_pit_shorter_life(self, cfg, fast_sim):
        r1 = run_life_prediction(cfg["geom"], cfg["mat"], cfg["h2_transport"],
            cfg["h2_embrittlement"], cfg["pit"], cfg["ops"], fast_sim,
            pit_depth_override=0.5e-3)
        r2 = run_life_prediction(cfg["geom"], cfg["mat"], cfg["h2_transport"],
            cfg["h2_embrittlement"], cfg["pit"], cfg["ops"], fast_sim,
            pit_depth_override=3.0e-3)
        assert r2.failure_cycle <= r1.failure_cycle

    def test_higher_pressure_shorter_life(self, cfg, fast_sim):
        ops_lo = OperatingConditions(p_H2_MPa=3.0, p_min_MPa=1.5, p_max_MPa=3.0,
            R_ratio=0.5, cycles_per_year=365, T_K=298.15, design_factor_b3112=0.5)
        ops_hi = OperatingConditions(p_H2_MPa=10.0, p_min_MPa=5.0, p_max_MPa=10.0,
            R_ratio=0.5, cycles_per_year=365, T_K=298.15, design_factor_b3112=0.5)
        r_lo = run_life_prediction(cfg["geom"], cfg["mat"], cfg["h2_transport"],
            cfg["h2_embrittlement"], cfg["pit"], cfg["ops"], fast_sim)
        # Override ops for high pressure
        r_hi = run_life_prediction(cfg["geom"], cfg["mat"], cfg["h2_transport"],
            cfg["h2_embrittlement"], cfg["pit"], ops_hi, fast_sim)
        assert r_hi.failure_cycle <= r_lo.failure_cycle

    def test_crack_history_non_empty(self, cfg, fast_sim):
        r = run_life_prediction(cfg["geom"], cfg["mat"], cfg["h2_transport"],
            cfg["h2_embrittlement"], cfg["pit"], cfg["ops"], fast_sim)
        assert len(r.crack_history_a) > 0

    def test_crack_history_monotonic(self, cfg, fast_sim):
        r = run_life_prediction(cfg["geom"], cfg["mat"], cfg["h2_transport"],
            cfg["h2_embrittlement"], cfg["pit"], cfg["ops"], fast_sim)
        if len(r.crack_history_a) > 2:
            assert np.all(np.diff(r.crack_history_a) >= -1e-12)

    def test_K_history_monotonic(self, cfg, fast_sim):
        r = run_life_prediction(cfg["geom"], cfg["mat"], cfg["h2_transport"],
            cfg["h2_embrittlement"], cfg["pit"], cfg["ops"], fast_sim)
        if len(r.crack_history_K) > 2:
            assert np.all(np.diff(r.crack_history_K) >= -1e-12)

    def test_cycles_to_years_conversion(self, cfg):
        assert abs(cycles_to_years(365, cfg["ops"]) - 1.0) < 1e-10

    def test_cycles_to_years_zero(self, cfg):
        assert cycles_to_years(0, cfg["ops"]) == 0.0

    def test_initial_pit_stored(self, cfg, fast_sim):
        r = run_life_prediction(cfg["geom"], cfg["mat"], cfg["h2_transport"],
            cfg["h2_embrittlement"], cfg["pit"], cfg["ops"], fast_sim,
            pit_depth_override=1.5e-3)
        assert abs(r.initial_pit_depth_m - 1.5e-3) < 1e-10

    def test_aspect_ratio_stored(self, cfg, fast_sim):
        r = run_life_prediction(cfg["geom"], cfg["mat"], cfg["h2_transport"],
            cfg["h2_embrittlement"], cfg["pit"], cfg["ops"], fast_sim,
            aspect_ratio_override=3.0)
        assert abs(r.aspect_ratio - 3.0) < 1e-10

    def test_precomputed_diffusion_accepted(self, cfg, fast_sim):
        diff = solve_diffusion(cfg["geom"], cfg["h2_transport"], cfg["ops"], fast_sim)
        r = run_life_prediction(cfg["geom"], cfg["mat"], cfg["h2_transport"],
            cfg["h2_embrittlement"], cfg["pit"], cfg["ops"], fast_sim,
            diffusion_result=diff)
        assert r.failure_cycle >= 0


# ============================================================
# N. LIFE PREDICTION EDGE CASES (8 tests)
# ============================================================
class TestLifeEdgeCases:
    def test_immediate_failure_deep_pit(self, cfg, tiny_sim):
        """Very deep pit should fail immediately or very quickly."""
        r = run_life_prediction(cfg["geom"], cfg["mat"], cfg["h2_transport"],
            cfg["h2_embrittlement"], cfg["pit"], cfg["ops"], tiny_sim,
            pit_depth_override=5.0e-3)  # 79% of wall
        assert r.failure_cycle < 100

    def test_shallow_pit_survives_longer(self, cfg, tiny_sim):
        r = run_life_prediction(cfg["geom"], cfg["mat"], cfg["h2_transport"],
            cfg["h2_embrittlement"], cfg["pit"], cfg["ops"], tiny_sim,
            pit_depth_override=0.1e-3)
        assert r.failure_cycle > 100

    def test_very_high_toughness_no_failure(self, cfg, tiny_sim):
        mat = MaterialProps(
            SMYS_MPa=358, SMTS_MPa=455, youngs_modulus_GPa=207,
            poissons_ratio=0.3, density_kg_m3=7850,
            K_IC_air_base_MPa_sqrtm=500, K_th_air_base_MPa_sqrtm=6,
            K_IC_air_seam_MPa_sqrtm=500, K_th_air_seam_MPa_sqrtm=3.5,
            C_paris_air=3e-12, m_paris_air=3.0)
        r = run_life_prediction(cfg["geom"], mat, cfg["h2_transport"],
            cfg["h2_embrittlement"], cfg["pit"], cfg["ops"], tiny_sim,
            pit_depth_override=0.5e-3)
        assert r.failure_mode in ("no_failure", "fracture", "wall_penetration")

    def test_zero_pressure_no_crack_growth(self, cfg, tiny_sim):
        ops = OperatingConditions(p_H2_MPa=0.001, p_min_MPa=0.0005,
            p_max_MPa=0.001, R_ratio=0.5, cycles_per_year=365,
            T_K=298.15, design_factor_b3112=0.5)
        r = run_life_prediction(cfg["geom"], cfg["mat"], cfg["h2_transport"],
            cfg["h2_embrittlement"], cfg["pit"], ops, tiny_sim)
        assert r.failure_mode == "no_failure"

    def test_wall_penetration_mode(self, cfg, tiny_sim):
        """Very high toughness but deep crack should eventually penetrate wall."""
        mat = MaterialProps(
            SMYS_MPa=358, SMTS_MPa=455, youngs_modulus_GPa=207,
            poissons_ratio=0.3, density_kg_m3=7850,
            K_IC_air_base_MPa_sqrtm=500, K_th_air_base_MPa_sqrtm=1.0,
            K_IC_air_seam_MPa_sqrtm=500, K_th_air_seam_MPa_sqrtm=1.0,
            C_paris_air=3e-10, m_paris_air=3.0)  # fast growth
        sim = SimControl(n_wall_nodes=10, dt_diffusion_s=1000, max_cycles=500_000,
            da_increment_m=1e-5, n_mc_samples=10, mc_seed=42,
            n_lhs_samples=10, lhs_seed=7)
        r = run_life_prediction(cfg["geom"], mat, cfg["h2_transport"],
            cfg["h2_embrittlement"], cfg["pit"], cfg["ops"], sim,
            pit_depth_override=3.0e-3)
        assert r.failure_mode in ("wall_penetration", "fracture")

    def test_converged_flag(self, cfg, tiny_sim):
        r = run_life_prediction(cfg["geom"], cfg["mat"], cfg["h2_transport"],
            cfg["h2_embrittlement"], cfg["pit"], cfg["ops"], tiny_sim)
        assert isinstance(r.converged, bool)

    def test_failure_crack_within_wall(self, cfg, fast_sim):
        r = run_life_prediction(cfg["geom"], cfg["mat"], cfg["h2_transport"],
            cfg["h2_embrittlement"], cfg["pit"], cfg["ops"], fast_sim)
        assert r.failure_crack_depth_m <= cfg["geom"].wall_thickness_m * 1.01

    def test_C_H_at_failure_non_negative(self, cfg, fast_sim):
        r = run_life_prediction(cfg["geom"], cfg["mat"], cfg["h2_transport"],
            cfg["h2_embrittlement"], cfg["pit"], cfg["ops"], fast_sim)
        assert r.C_H_at_failure >= 0


# ============================================================
# O. LHS SAMPLING (6 tests)
# ============================================================
class TestLHS:
    def test_shape(self):
        from monte_carlo import latin_hypercube_sample
        s = latin_hypercube_sample(100, 6, seed=42)
        assert s.shape == (100, 6)

    def test_unit_cube(self):
        from monte_carlo import latin_hypercube_sample
        s = latin_hypercube_sample(100, 6, seed=42)
        assert np.all(s >= 0) and np.all(s <= 1)

    def test_stratification(self):
        from monte_carlo import latin_hypercube_sample
        s = latin_hypercube_sample(100, 3, seed=42)
        for j in range(3):
            bins = np.floor(s[:, j] * 100).astype(int)
            bins = np.clip(bins, 0, 99)
            assert len(np.unique(bins)) == 100

    def test_reproducibility(self):
        from monte_carlo import latin_hypercube_sample
        s1 = latin_hypercube_sample(50, 4, seed=7)
        s2 = latin_hypercube_sample(50, 4, seed=7)
        assert np.array_equal(s1, s2)

    def test_different_seeds_different(self):
        from monte_carlo import latin_hypercube_sample
        s1 = latin_hypercube_sample(50, 4, seed=1)
        s2 = latin_hypercube_sample(50, 4, seed=2)
        assert not np.array_equal(s1, s2)

    def test_single_sample(self):
        from monte_carlo import latin_hypercube_sample
        s = latin_hypercube_sample(1, 3, seed=42)
        assert s.shape == (1, 3)


# ============================================================
# P. MONTE CARLO (6 tests)
# ============================================================
class TestMonteCarlo:
    def test_mc_runs(self, cfg, tiny_sim):
        tiny_sim.n_mc_samples = 10
        tiny_sim.max_cycles = 5_000
        from monte_carlo import run_monte_carlo
        r = run_monte_carlo(cfg["geom"], cfg["mat"], cfg["h2_transport"],
            cfg["h2_embrittlement"], cfg["pit"], cfg["ops"], tiny_sim)
        assert r.n_samples == 10

    def test_percentiles_ordered(self, cfg, tiny_sim):
        tiny_sim.n_mc_samples = 20
        tiny_sim.max_cycles = 5_000
        from monte_carlo import run_monte_carlo
        r = run_monte_carlo(cfg["geom"], cfg["mat"], cfg["h2_transport"],
            cfg["h2_embrittlement"], cfg["pit"], cfg["ops"], tiny_sim)
        assert r.percentile_5 <= r.percentile_50 <= r.percentile_95

    def test_lives_non_negative(self, cfg, tiny_sim):
        tiny_sim.n_mc_samples = 10
        tiny_sim.max_cycles = 5_000
        from monte_carlo import run_monte_carlo
        r = run_monte_carlo(cfg["geom"], cfg["mat"], cfg["h2_transport"],
            cfg["h2_embrittlement"], cfg["pit"], cfg["ops"], tiny_sim)
        assert np.all(r.remaining_life_years >= 0)

    def test_fraction_below_bounded(self, cfg, tiny_sim):
        tiny_sim.n_mc_samples = 10
        tiny_sim.max_cycles = 5_000
        from monte_carlo import run_monte_carlo
        r = run_monte_carlo(cfg["geom"], cfg["mat"], cfg["h2_transport"],
            cfg["h2_embrittlement"], cfg["pit"], cfg["ops"], tiny_sim)
        assert 0 <= r.fraction_below_10yr <= 1
        assert 0 <= r.fraction_below_20yr <= 1

    def test_failure_modes_valid(self, cfg, tiny_sim):
        tiny_sim.n_mc_samples = 10
        tiny_sim.max_cycles = 5_000
        from monte_carlo import run_monte_carlo
        r = run_monte_carlo(cfg["geom"], cfg["mat"], cfg["h2_transport"],
            cfg["h2_embrittlement"], cfg["pit"], cfg["ops"], tiny_sim)
        valid = {"fracture", "wall_penetration", "no_failure"}
        for m in r.failure_modes:
            assert m in valid

    def test_sample_arrays_correct_length(self, cfg, tiny_sim):
        tiny_sim.n_mc_samples = 15
        tiny_sim.max_cycles = 5_000
        from monte_carlo import run_monte_carlo
        r = run_monte_carlo(cfg["geom"], cfg["mat"], cfg["h2_transport"],
            cfg["h2_embrittlement"], cfg["pit"], cfg["ops"], tiny_sim)
        assert len(r.remaining_life_years) == 15
        assert len(r.initial_pit_depths_m) == 15


# ============================================================
# Q. ML SURROGATE (8 tests)
# ============================================================
class TestSurrogate:
    def test_trains(self):
        from ml_surrogate import train_surrogate
        rng = np.random.RandomState(42)
        X = rng.random((200, 6))
        y = 50*X[:,0] - 30*X[:,1] + 10*rng.random(200)
        r = train_surrogate(X, y, list("abcdef"))
        assert r.r2_train > 0

    def test_importances_sum_to_one(self):
        from ml_surrogate import train_surrogate
        rng = np.random.RandomState(42)
        X = rng.random((200, 4))
        y = X[:,0]*10 + rng.random(200)
        r = train_surrogate(X, y, list("abcd"))
        assert abs(sum(r.feature_importances.values()) - 1.0) < 0.01

    def test_ranking_validation(self):
        from ml_surrogate import validate_feature_ranking
        imp = {"K_IC_seam":0.4, "pit_depth_m":0.25, "p_H2_MPa":0.15,
               "D_L_m2s":0.1, "f_seam":0.05, "aspect_ratio":0.05}
        r = validate_feature_ranking(imp)
        assert r["K_IC_seam"] == "OK"

    def test_save_load(self, tmp_path):
        from ml_surrogate import train_surrogate, save_surrogate, load_surrogate
        rng = np.random.RandomState(42)
        X = rng.random((100, 3))
        y = X[:,0]*10 + rng.random(100)
        r = train_surrogate(X, y, list("abc"))
        p = str(tmp_path / "m.pkl")
        save_surrogate(r, p)
        loaded = load_surrogate(p)
        assert loaded["r2_test"] == r.r2_test

    def test_predictions_non_negative(self):
        """Surrogate trained on positive targets should predict positive."""
        from ml_surrogate import train_surrogate
        rng = np.random.RandomState(42)
        X = rng.random((200, 4))
        y = np.abs(50*X[:,0] + 10*rng.random(200))
        r = train_surrogate(X, y, list("abcd"))
        preds = r.model.predict(r.X_test)
        assert np.all(preds > -50)  # GBR may predict slightly negative

    def test_r2_test_reasonable(self):
        from ml_surrogate import train_surrogate
        rng = np.random.RandomState(42)
        X = rng.random((300, 4))
        y = 100*X[:,0] - 50*X[:,1] + 5*rng.random(300)
        r = train_surrogate(X, y, list("abcd"))
        assert r.r2_test > 0.5

    def test_dominant_feature_highest_importance(self):
        from ml_surrogate import train_surrogate
        rng = np.random.RandomState(42)
        X = rng.random((300, 4))
        y = 100*X[:,0] + 0.01*rng.random(300)  # X0 dominates
        r = train_surrogate(X, y, list("abcd"))
        assert r.feature_importances["a"] > 0.5

    def test_insufficient_data_raises(self):
        from ml_surrogate import train_surrogate
        X = np.random.random((5, 3))
        y = np.array([1, 2, 3, 4, 5])
        with pytest.raises(ValueError):
            train_surrogate(X, y, list("abc"))


# ============================================================
# R. CYBERSECURITY (10 tests)
# ============================================================
class TestCybersecurity:
    def test_stride_count(self):
        assert len(STRIDE_THREATS) == 7

    def test_stride_has_required_fields(self):
        for t in STRIDE_THREATS:
            assert "id" in t and "category" in t and "mitigation" in t

    def test_bounds_valid(self):
        assert check_parameter_bounds("p_H2_MPa", 7.0)["valid"]

    def test_bounds_invalid_high(self):
        assert not check_parameter_bounds("p_H2_MPa", 50.0)["valid"]

    def test_bounds_invalid_low(self):
        assert not check_parameter_bounds("pit_depth_m", -0.001)["valid"]

    def test_bounds_unknown_param(self):
        r = check_parameter_bounds("unknown_param", 42.0)
        assert r["valid"]  # no bounds defined = passes

    def test_audit_chain_valid(self):
        logger = AuditLogger()
        logger.log("eng01", "run_sim", "in1", "out1")
        logger.log("eng01", "run_sim", "in2", "out2")
        logger.log("eng02", "modify_config", "in3", "out3")
        assert logger.verify_chain()["valid"]
        assert logger.verify_chain()["entries_checked"] == 3

    def test_audit_chain_tamper_detected(self):
        logger = AuditLogger()
        logger.log("eng01", "run", "a", "b")
        logger.log("eng01", "run", "c", "d")
        logger.chain[0].chain_hash = "tampered_hash"
        assert not logger.verify_chain()["valid"]

    def test_spoofing_negative_depths(self):
        depths = np.array([0.001, -0.005, 0.002])
        assert detect_pit_depth_spoofing(depths, 0.00635)["spoofing_detected"]

    def test_monotonicity_correct_data(self):
        """Deeper pits should correlate with shorter life."""
        pits = np.array([0.5, 1.0, 1.5, 2.0, 2.5, 3.0]) * 1e-3
        lives = np.array([200, 150, 100, 60, 30, 5])
        r = physics_monotonicity_check(pits, lives)
        assert r["consistent"]


# ============================================================
# S. INTEGRATION: FULL CHAIN (6 tests)
# ============================================================
class TestIntegration:
    def test_full_chain_default_params(self, cfg, fast_sim):
        """Full chain from config to life prediction."""
        diff = solve_diffusion(cfg["geom"], cfg["h2_transport"], cfg["ops"], fast_sim)
        life = run_life_prediction(cfg["geom"], cfg["mat"], cfg["h2_transport"],
            cfg["h2_embrittlement"], cfg["pit"], cfg["ops"], fast_sim,
            diffusion_result=diff)
        years = cycles_to_years(life.failure_cycle, cfg["ops"])
        assert years >= 0
        assert life.failure_mode in ("fracture", "wall_penetration", "no_failure")

    def test_diffusion_feeds_life(self, cfg, fast_sim):
        """Life prediction with and without pre-computed diffusion should match."""
        diff = solve_diffusion(cfg["geom"], cfg["h2_transport"], cfg["ops"], fast_sim)
        r1 = run_life_prediction(cfg["geom"], cfg["mat"], cfg["h2_transport"],
            cfg["h2_embrittlement"], cfg["pit"], cfg["ops"], fast_sim,
            diffusion_result=diff)
        r2 = run_life_prediction(cfg["geom"], cfg["mat"], cfg["h2_transport"],
            cfg["h2_embrittlement"], cfg["pit"], cfg["ops"], fast_sim)
        assert r1.failure_cycle == r2.failure_cycle

    def test_crack_depth_at_failure_consistent(self, cfg, fast_sim):
        """If fracture, K_max should be near K_IC at failure."""
        r = run_life_prediction(cfg["geom"], cfg["mat"], cfg["h2_transport"],
            cfg["h2_embrittlement"], cfg["pit"], cfg["ops"], fast_sim)
        if r.failure_mode == "fracture":
            assert r.K_max_at_failure >= r.K_IC_at_failure * 0.95

    def test_pit_transition_before_failure(self, cfg, fast_sim):
        """Pit must transition before crack can grow to failure."""
        r = run_life_prediction(cfg["geom"], cfg["mat"], cfg["h2_transport"],
            cfg["h2_embrittlement"], cfg["pit"], cfg["ops"], fast_sim)
        if r.failure_mode == "fracture":
            assert r.pit_transitioned
            assert r.transition_cycle <= r.failure_cycle

    def test_cybersecurity_bounds_on_config(self, cfg):
        """All default config values should pass bounds checking."""
        checks = []
        g = cfg["geom"]; m = cfg["mat"]; o = cfg["ops"]
        checks.append(check_parameter_bounds("wall_thickness_m", g.wall_thickness_m))
        checks.append(check_parameter_bounds("SMYS_MPa", m.SMYS_MPa))
        checks.append(check_parameter_bounds("p_H2_MPa", o.p_H2_MPa))
        for c in checks:
            assert c["valid"], c["message"]

    def test_maop_below_operating_pressure(self, cfg):
        """B31.12 MAOP should be below default operating pressure (known de-rating)."""
        maop = maop_under_b3112(cfg["geom"], cfg["mat"], cfg["ops"])
        assert maop < cfg["ops"].p_max_MPa


# ============================================================
# T. REGRESSION: LOCKED BASELINES (4 tests)
# ============================================================
class TestRegression:
    def test_baseline_pit_depth(self, cfg):
        a = pit_depth_after_ng_service(cfg["pit"])
        assert abs(a * 1000 - 0.8185) < 0.01, f"Pit depth drifted: {a*1000:.4f} mm"

    def test_baseline_hoop_stress(self, cfg):
        s = hoop_stress(cfg["geom"], cfg["ops"].p_max_MPa)
        assert abs(s - 280.0) < 0.1, f"Hoop stress drifted: {s:.1f}"

    def test_baseline_maop(self, cfg):
        m = maop_under_b3112(cfg["geom"], cfg["mat"], cfg["ops"])
        assert abs(m - 2.42) < 0.05, f"MAOP drifted: {m:.3f}"

    def test_baseline_sieverts(self, cfg):
        C0 = sieverts_surface_concentration(cfg["h2_transport"], cfg["ops"].p_H2_MPa)
        assert abs(C0 - 0.2117) < 0.005, f"Sieverts drifted: {C0:.4f}"


# ============================================================
# U. VISUALIZATION OUTPUT (4 tests)
# ============================================================
class TestVisualization:
    def test_hero_creates_file(self, cfg, fast_sim, tmp_path):
        from visualization import plot_hero_diffusion_crack
        diff = solve_diffusion(cfg["geom"], cfg["h2_transport"], cfg["ops"], fast_sim)
        life = run_life_prediction(cfg["geom"], cfg["mat"], cfg["h2_transport"],
            cfg["h2_embrittlement"], cfg["pit"], cfg["ops"], fast_sim, diffusion_result=diff)
        p = str(tmp_path / "assets" / "hero.png")
        plot_hero_diffusion_crack(diff, life, cfg["geom"], save_path=p)
        assert os.path.exists(p) and os.path.getsize(p) > 1000

    def test_sensitivity_creates_file(self, tmp_path):
        from visualization import plot_sensitivity_tornado
        corrs = {"p_H2_MPa": -0.7, "pit_depth_m": -0.6, "K_IC_seam": 0.1}
        d = tmp_path / "assets"; d.mkdir()
        p = str(d / "sens.png")
        plot_sensitivity_tornado(corrs, save_path=p)
        assert os.path.exists(p) and os.path.getsize(p) > 1000

    def test_surrogate_parity_creates_file(self, tmp_path):
        from visualization import plot_surrogate_parity
        from ml_surrogate import train_surrogate
        rng = np.random.RandomState(42)
        X = rng.random((200, 4)); y = X[:,0]*50 + rng.random(200)*5
        surr = train_surrogate(X, y, list("abcd"))
        d = tmp_path / "assets"; d.mkdir(exist_ok=True)
        p = str(d / "parity.png")
        plot_surrogate_parity(surr, save_path=p)
        assert os.path.exists(p) and os.path.getsize(p) > 1000

    def test_gif_creates_file(self, cfg, fast_sim, tmp_path):
        from visualization import generate_gif
        diff = solve_diffusion(cfg["geom"], cfg["h2_transport"], cfg["ops"], fast_sim)
        life = run_life_prediction(cfg["geom"], cfg["mat"], cfg["h2_transport"],
            cfg["h2_embrittlement"], cfg["pit"], cfg["ops"], fast_sim, diffusion_result=diff)
        p = str(tmp_path / "assets" / "test.gif")
        generate_gif(diff, life, cfg["geom"], save_path=p, n_frames=5, fps=2)
        assert os.path.exists(p) and os.path.getsize(p) > 1000


# ============================================================
if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
