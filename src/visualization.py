"""
INTEGRITY CODE SERIES -- Week 7
Visualization Package v2: Engineering Publication Standard

Outputs:
    1. Hero: Spatiotemporal heatmap (H2 diffusion) + crack growth trajectory
    2. Monte Carlo CDF + histogram
    3. Sensitivity tornado (Spearman)
    4. Surrogate parity + feature importance
    5. Iso-risk contour: remaining life vs (p_H2, pit_depth) with 10yr boundary
    6. Failure Assessment Diagram (FAD) per API 579-1 / BS 7910
    7. Animated GIF with color-mapped wall concentration
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import os
from typing import Optional, Dict

_D = chr(36)

# ============================================================
# STYLE
# ============================================================
def _apply_style():
    plt.rcParams.update({
        "figure.facecolor": "white", "axes.facecolor": "white",
        "axes.edgecolor": "#2a2a2a", "axes.linewidth": 0.7,
        "axes.labelcolor": "#1a1a1a", "axes.labelsize": 11,
        "axes.titlesize": 11.5, "axes.titleweight": "normal",
        "axes.grid": True, "axes.axisbelow": True,
        "grid.color": "#d4d4d4", "grid.linewidth": 0.35, "grid.linestyle": "-",
        "xtick.direction": "out", "ytick.direction": "out",
        "xtick.major.size": 4, "ytick.major.size": 4,
        "xtick.minor.size": 2, "ytick.minor.size": 2,
        "xtick.major.width": 0.55, "ytick.major.width": 0.55,
        "xtick.labelsize": 9.5, "ytick.labelsize": 9.5,
        "xtick.color": "#2a2a2a", "ytick.color": "#2a2a2a",
        "legend.frameon": False, "legend.fontsize": 8.5,
        "legend.handlelength": 1.8,
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial"],
        "font.size": 10, "lines.linewidth": 1.2, "lines.markersize": 3.5,
        "savefig.dpi": 300, "savefig.bbox": "tight", "savefig.pad_inches": 0.06,
    })

_NAV  = "#1b3a5c"; _BLUE = "#345d8a"; _STL  = "#4c80b0"; _TEAL = "#2e7d7b"
_DRED = "#8c2318"; _BRCK = "#b03828"; _CHAR = "#333333"; _G60  = "#666666"
_G80  = "#bbbbbb"; _ORNG = "#c06000"; _GOLD = "#a87b00"

_PARAM_LABELS = {
    "p_H2_MPa": r"$p_{\mathrm{H_2}}$  [MPa]",
    "pit_depth_m": r"$a_{\mathrm{pit}}$  [m]",
    "D_L_m2s": r"$D_L$  [m$^2$ s$^{-1}$]",
    "K_IC_seam": r"$K_{\mathrm{IC,seam}}$  [MPa$\sqrt{\mathrm{m}}$]",
    "f_seam": r"$f_{\mathrm{seam}}$",
    "aspect_ratio": r"$c/a$",
}

def _ensure_dir(path):
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)


# ============================================================
# 1. HERO: SPATIOTEMPORAL HEATMAP + CRACK GROWTH
# ============================================================
def plot_hero_diffusion_crack(diffusion_result, life_result, geom,
                              save_path="assets/hero_h2_crack.png"):
    _apply_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 5.2))
    fig.subplots_adjust(wspace=0.38)

    x_mm = diffusion_result.x_grid * 1000.0
    t_hours = diffusion_result.t_grid / 3600.0
    C = diffusion_result.C_field

    # Left: spatiotemporal heatmap
    X, T = np.meshgrid(x_mm, t_hours)
    c_max = max(C.max(), 1e-6)
    im = ax1.pcolormesh(X, T, C, cmap="YlOrRd", shading="gouraud",
                        vmin=0, vmax=c_max, rasterized=True)
    cb = fig.colorbar(im, ax=ax1, pad=0.02, aspect=30)
    cb.set_label(r"$C_{\mathrm{H}}$ [mol m$^{-3}$]", fontsize=9.5)
    cb.ax.tick_params(labelsize=8.5)

    # Pit depth marker on heatmap (visible reference line)
    pit_mm = life_result.initial_pit_depth_m * 1000.0
    if pit_mm > 0:
        ax1.axvline(pit_mm, color="white", linewidth=1.5, linestyle="-", zorder=5)
        ax1.axvline(pit_mm, color=_DRED, linewidth=0.8, linestyle="--", zorder=6)
        ax1.text(pit_mm + 0.08, t_hours[-1]*0.95,
                 f"$a_0$ = {pit_mm:.2f} mm", fontsize=8, color="white",
                 va="top", zorder=7,
                 bbox=dict(facecolor=_DRED, alpha=0.7, edgecolor="none",
                           boxstyle="round,pad=0.15"))

    ax1.set_xlabel("Depth from inner surface [mm]")
    ax1.set_ylabel("Time [hours]")
    ax1.set_title("(a)  Hydrogen concentration field", loc="left")
    ax1.set_xlim(0, x_mm[-1])
    ax1.set_ylim(0, t_hours[-1])

    # Right: crack depth vs cycles
    if len(life_result.crack_history_cycles) > 2:
        a_mm = life_result.crack_history_a * 1000.0
        cyc = life_result.crack_history_cycles
        ax2.plot(cyc, a_mm, color=_NAV, lw=1.3)
        ax2.axhline(geom.wall_thickness_m*1000*0.9, color=_DRED,
                    ls="-", lw=0.6, alpha=0.6, label="90% wall")
        if life_result.pit_transitioned and life_result.transition_cycle > 0:
            ax2.axvline(life_result.transition_cycle, color=_ORNG,
                        ls=":", lw=0.6, label="Transition")
        ax2.set_xlabel("Fatigue cycles, $N$")
        ax2.set_ylabel("Crack depth, $a$ [mm]")
        ax2.set_title("(b)  HA-FCG at ERW seam", loc="left")
        ax2.legend(fontsize=7.5, loc="upper left")
        ax2.minorticks_on()

        ax2b = ax2.twinx()
        ax2b.plot(cyc, life_result.crack_history_C_H, color=_TEAL,
                  lw=0.7, ls="--", alpha=0.65)
        ax2b.set_ylabel(r"$C_{\mathrm{H}}$ at tip [mol m$^{-3}$]",
                        fontsize=9, color=_TEAL)
        ax2b.tick_params(axis="y", labelcolor=_TEAL, labelsize=8.5)
        ax2b.spines["right"].set_color(_TEAL)
        ax2b.spines["right"].set_linewidth(0.5)
    else:
        ax2.text(0.5, 0.5,
                 "Immediate failure at cycle 0\n"
                 r"($K_{\max}$ > $K_{\mathrm{IC}}$)",
                 ha="center", va="center", fontsize=11, color=_CHAR,
                 transform=ax2.transAxes)
        ax2.set_title("(b)  Crack growth history", loc="left")

    _ensure_dir(save_path)
    fig.savefig(save_path); plt.close(fig)
    return save_path


# ============================================================
# 2. CDF + HISTOGRAM
# ============================================================
def plot_monte_carlo_cdf(mc_result, save_path="assets/mc_cdf.png"):
    _apply_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 5.2))
    fig.subplots_adjust(wspace=0.32)

    rl = mc_result.remaining_life_years
    srl = np.sort(rl)
    cdf = np.arange(1, len(srl)+1) / len(srl)

    # Detect right-censoring threshold (cluster of samples at max value)
    max_life = srl[-1]
    censor_tol = max_life * 0.02  # within 2% of max = censored
    n_censored = int(np.sum(rl >= max_life - censor_tol))
    frac_censored = n_censored / len(rl)
    has_censoring = frac_censored > 0.10  # flag if >10% censored

    # CDF
    ax1.step(srl, cdf, where="post", color=_NAV, lw=1.3)
    for pv, lb, cl in [(mc_result.percentile_5, "P$_5$", _DRED),
                       (mc_result.percentile_50, "P$_{50}$", _ORNG),
                       (mc_result.percentile_95, "P$_{95}$", _TEAL)]:
        ax1.axvline(pv, color=cl, ls="--", lw=0.7)
        ax1.text(pv + max_life*0.01, 0.15, f"{lb} = {pv:.0f} yr",
                 fontsize=8, color=cl, rotation=90, va="bottom")

    if has_censoring:
        ax1.annotate(f"{frac_censored:.0%} right-censored\n(simulation cap)",
                     xy=(max_life, 1.0 - frac_censored), xytext=(max_life*0.55, 0.5),
                     fontsize=8, color=_G60, ha="center",
                     arrowprops=dict(arrowstyle="->", color=_G60, lw=0.7))

    ax1.set_xlabel("Remaining life [years]")
    ax1.set_ylabel("Cumulative probability")
    ax1.set_title(f"(a)  Remaining life CDF ($n$ = {mc_result.n_samples})", loc="left")
    ax1.set_ylim(0, 1.08); ax1.set_xlim(left=0); ax1.minorticks_on()

    # Histogram: separate censored from uncensored
    if has_censoring:
        rl_uncensored = rl[rl < max_life - censor_tol]
        bin_max = max_life - censor_tol if len(rl_uncensored) > 0 else max_life
        bins = np.linspace(0, bin_max * 1.05, 35)

        ax2.hist(rl_uncensored, bins=bins, color=_STL,
                 edgecolor="white", lw=0.3, alpha=0.8, label="Observed failures")

        # Censored bar: placed at the right edge, hatched
        bar_width = bins[1] - bins[0] if len(bins) > 1 else 5
        bar_x = bin_max * 1.05 + bar_width * 0.5
        ax2.bar(bar_x, n_censored, width=bar_width, color=_G80,
                edgecolor=_CHAR, linewidth=0.5, hatch="//", alpha=0.7,
                label=f"Censored (>{max_life:.0f} yr): {n_censored}")
        ax2.legend(fontsize=7.5, loc="upper center")
    else:
        bins = np.linspace(0, max(srl)*1.05, 40)
        ax2.hist(rl, bins=bins, color=_STL, edgecolor="white", lw=0.3, alpha=0.8)

    ym = ax2.get_ylim()[1]
    ax2.axvline(10, color=_DRED, ls="-", lw=0.9)
    ax2.text(12, ym*0.92, f"10 yr: {mc_result.fraction_below_10yr:.0%}",
             fontsize=7.5, color=_DRED, va="top")
    ax2.axvline(20, color=_ORNG, ls="-", lw=0.9)
    ax2.text(22, ym*0.80, f"20 yr: {mc_result.fraction_below_20yr:.0%}",
             fontsize=7.5, color=_ORNG, va="top")
    ax2.set_xlabel("Remaining life [years]")
    ax2.set_ylabel("Count")
    ax2.set_title("(b)  Remaining life histogram", loc="left")
    ax2.set_xlim(left=0); ax2.minorticks_on()

    _ensure_dir(save_path)
    fig.savefig(save_path); plt.close(fig)
    return save_path


# ============================================================
# 3. SENSITIVITY TORNADO
# ============================================================
def plot_sensitivity_tornado(spearman_corrs, save_path="assets/sensitivity_tornado.png"):
    _apply_style()
    fig, ax = plt.subplots(figsize=(7.5, 4.2))

    sp = sorted(spearman_corrs.items(), key=lambda x: abs(x[1]), reverse=True)
    names = [_PARAM_LABELS.get(p[0], p[0]) for p in sp]
    vals = [p[1] for p in sp]
    cols = [_DRED if v < 0 else _BLUE for v in vals]
    y = np.arange(len(names))

    ax.barh(y, vals, color=cols, edgecolor="white", height=0.52, lw=0.3)
    ax.set_yticks(y); ax.set_yticklabels(names, fontsize=9.5)
    ax.set_xlabel(r"Spearman rank correlation, $\rho_s$")
    ax.set_title("Sensitivity analysis: LHS parametric sweep", loc="left")
    ax.axvline(0, color=_CHAR, lw=0.5)
    ax.set_xlim(-1.0, 1.0); ax.minorticks_on()
    for i, v in enumerate(vals):
        off = 0.025 * np.sign(v)
        ax.text(v+off, i, f"{v:+.3f}", va="center",
                ha="left" if v>=0 else "right", fontsize=8, color=_CHAR)

    _ensure_dir(save_path)
    fig.savefig(save_path); plt.close(fig)
    return save_path


# ============================================================
# 4. SURROGATE PARITY
# ============================================================
def plot_surrogate_parity(surrogate_result, save_path="assets/surrogate_parity.png"):
    _apply_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 5.2))
    fig.subplots_adjust(wspace=0.35)

    yt = surrogate_result.y_test; yp = surrogate_result.y_pred_test
    lm = max(yt.max(), yp.max()) * 1.08; lims = [0, lm]

    ax1.scatter(yt, yp, s=14, c=_STL, alpha=0.45, edgecolors="none", zorder=3)
    ax1.plot(lims, lims, color=_CHAR, lw=0.6, ls="--", zorder=2)
    xr = np.linspace(0, lm, 50)
    ax1.fill_between(xr, xr*0.8, xr*1.2, color=_G80, alpha=0.18, zorder=1)
    ax1.set_xlabel("Physics solver [years]")
    ax1.set_ylabel("GBR surrogate [years]")
    ax1.set_title(f"(a)  Parity (R$^2$ = {surrogate_result.r2_test:.3f},"
                  f"  MAE = {surrogate_result.mae_test:.1f} yr)", loc="left")
    ax1.set_xlim(lims); ax1.set_ylim(lims)
    ax1.set_aspect("equal"); ax1.minorticks_on()

    sfi = sorted(surrogate_result.feature_importances.items(), key=lambda x: x[1], reverse=True)
    fn = [_PARAM_LABELS.get(f[0], f[0]) for f in sfi]
    fv = [f[1] for f in sfi]; yp2 = np.arange(len(fn))
    ax2.barh(yp2, fv, color=_NAV, edgecolor="white", height=0.52, lw=0.3)
    ax2.set_yticks(yp2); ax2.set_yticklabels(fn, fontsize=9.5)
    ax2.set_xlabel("Feature importance")
    ax2.set_title("(b)  GBR feature importances", loc="left")
    ax2.invert_yaxis(); ax2.minorticks_on()
    ax2.set_xlim(0, max(fv)*1.18)
    for i, v in enumerate(fv):
        ax2.text(v+0.004, i, f"{v:.3f}", va="center", fontsize=8, color=_CHAR)

    _ensure_dir(save_path)
    fig.savefig(save_path); plt.close(fig)
    return save_path


# ============================================================
# 5. ISO-RISK CONTOUR: remaining life vs (p_H2, pit_depth)
# ============================================================
def plot_iso_risk_contour(lhs_params, lhs_lives, param_names,
                          save_path="assets/iso_risk_contour.png",
                          surrogate_model=None):
    """
    2D filled contour of remaining life as function of the two
    dominant parameters (p_H2 and pit_depth).

    If a trained surrogate model is provided, predictions are made
    on a clean 2D grid with other parameters at their median values.
    This avoids interpolation artifacts from projecting 6D LHS data.

    If no surrogate, falls back to linear griddata (less clean).
    """
    _apply_style()
    from scipy.interpolate import griddata

    idx_p = param_names.index("p_H2_MPa")
    idx_a = param_names.index("pit_depth_m")

    p_range = (lhs_params[:, idx_p].min(), lhs_params[:, idx_p].max())
    a_range = (lhs_params[:, idx_a].min(), lhs_params[:, idx_a].max())

    n_grid = 80
    pi = np.linspace(p_range[0], p_range[1], n_grid)
    ai = np.linspace(a_range[0], a_range[1], n_grid)
    PI, AI = np.meshgrid(pi, ai)

    if surrogate_model is not None:
        # Clean grid: other params at median
        medians = np.median(lhs_params, axis=0)
        X_pred = np.zeros((PI.size, len(param_names)))
        for j in range(len(param_names)):
            if j == idx_p:
                X_pred[:, j] = PI.ravel()
            elif j == idx_a:
                X_pred[:, j] = AI.ravel()
            else:
                X_pred[:, j] = medians[j]
        LI = surrogate_model.predict(X_pred).reshape(PI.shape)
        LI = np.clip(LI, 0, None)
        subtitle = "(other parameters at LHS median, GBR surrogate)"
    else:
        # Fallback: linear interpolation of raw data
        p_vals = lhs_params[:, idx_p]
        a_vals = lhs_params[:, idx_a]
        LI = griddata((p_vals, a_vals), lhs_lives, (PI, AI), method="linear")
        LI_near = griddata((p_vals, a_vals), lhs_lives, (PI, AI), method="nearest")
        mask = np.isnan(LI)
        LI[mask] = LI_near[mask]
        LI = np.clip(LI, 0, None)
        subtitle = "(projected from 6D LHS, linear interpolation)"

    ai_mm = ai * 1000.0
    AI_mm = AI * 1000.0

    fig, ax = plt.subplots(figsize=(8, 5.5))

    life_max = min(LI.max(), 150)
    levels = np.linspace(0, life_max, 25)
    cf = ax.contourf(PI, AI_mm, LI, levels=levels, cmap="RdYlGn", extend="max")
    cb = fig.colorbar(cf, ax=ax, pad=0.02, aspect=30)
    cb.set_label("Remaining life [years]", fontsize=10)
    cb.ax.tick_params(labelsize=8.5)

    # Threshold contours
    try:
        cs10 = ax.contour(PI, AI_mm, LI, levels=[10], colors=[_DRED],
                          linewidths=2.0, linestyles="-")
        ax.clabel(cs10, fmt="10 yr", fontsize=8.5, colors=[_DRED])
    except ValueError:
        pass

    try:
        cs20 = ax.contour(PI, AI_mm, LI, levels=[20], colors=[_ORNG],
                          linewidths=1.4, linestyles="--")
        ax.clabel(cs20, fmt="20 yr", fontsize=8.5, colors=[_ORNG])
    except ValueError:
        pass

    try:
        cs50 = ax.contour(PI, AI_mm, LI, levels=[50], colors=["white"],
                          linewidths=0.8, linestyles=":")
        ax.clabel(cs50, fmt="50 yr", fontsize=7.5, colors=["white"])
    except ValueError:
        pass

    ax.set_xlabel(r"Hydrogen pressure, $p_{\mathrm{H_2}}$ [MPa]")
    ax.set_ylabel(r"Initial pit depth, $a_{\mathrm{pit}}$ [mm]")
    ax.set_title("Iso-risk map: remaining life vs. pressure and pit depth", loc="left")
    ax.minorticks_on()

    _ensure_dir(save_path)
    fig.savefig(save_path); plt.close(fig)
    return save_path


# ============================================================
# 6. FAILURE ASSESSMENT DIAGRAM (FAD) per API 579-1 / BS 7910
# ============================================================
def plot_fad(life_result, geom, mat, he,
             save_path="assets/fad.png"):
    """
    Failure Assessment Diagram.

    Axes:
        K_r = K_applied / K_IC(C_H)    (fracture axis)
        L_r = sigma_ref / sigma_yield   (collapse axis)

    FAD envelope (Option 1, BS 7910 Level 2A):
        K_r = f(L_r) = (1 - 0.14 * L_r^2) * [0.3 + 0.7 * exp(-0.65 * L_r^6)]
        valid for 0 <= L_r <= L_r_max where L_r_max = (SMYS + SMTS) / (2 * SMYS)

    Assessment point trajectory plotted over fatigue cycles.
    """
    _apply_style()
    from ha_fcg import newman_raju_simplified, degraded_toughness
    from pit_to_crack import hoop_stress

    fig, ax = plt.subplots(figsize=(7, 6.5))

    # FAD envelope (Option 1)
    L_r_max = (mat.SMYS_MPa + mat.SMTS_MPa) / (2.0 * mat.SMYS_MPa)
    L_r_cut = min(L_r_max, 1.6)

    Lr_env = np.linspace(0, L_r_cut, 200)
    Kr_env = (1.0 - 0.14 * Lr_env**2) * (0.3 + 0.7 * np.exp(-0.65 * Lr_env**6))

    # Fill unsafe region
    ax.fill_between(Lr_env, Kr_env, 1.5, color="#f5e6e6", alpha=0.5, zorder=1)
    ax.fill_betweenx([0, 1.5], L_r_cut, 1.8, color="#f5e6e6", alpha=0.5, zorder=1)
    ax.plot(Lr_env, Kr_env, color=_DRED, lw=1.6, zorder=4, label="FAD envelope (Option 1)")
    ax.axvline(L_r_cut, color=_DRED, lw=1.0, ls="--", zorder=4)

    # Assessment trajectory from crack history
    if len(life_result.crack_history_a) > 2:
        n_pts = len(life_result.crack_history_a)
        Kr_traj = np.zeros(n_pts)
        Lr_traj = np.zeros(n_pts)

        for i in range(n_pts):
            a_i = life_result.crack_history_a[i]
            C_H_i = life_result.crack_history_C_H[i]
            sigma_max = hoop_stress(geom, 7.0)  # operating pressure

            K_app = newman_raju_simplified(sigma_max, a_i, geom.wall_thickness_m)
            K_IC_H = degraded_toughness(mat.K_IC_air_seam_MPa_sqrtm, C_H_i, he)

            Kr_traj[i] = K_app / K_IC_H if K_IC_H > 0 else 1.5
            # Reference stress for surface crack (net section yield)
            # sigma_ref = sigma * t / (t - a) for through-wall bending
            t = geom.wall_thickness_m
            sigma_ref = sigma_max * t / max(t - a_i, 0.001 * t)
            Lr_traj[i] = sigma_ref / mat.SMYS_MPa

        # Color by cycle progress
        cmap = plt.cm.viridis
        # Plot as colored scatter
        cycles = life_result.crack_history_cycles
        norm = plt.Normalize(vmin=cycles[0], vmax=cycles[-1])
        sc = ax.scatter(Lr_traj, Kr_traj, c=cycles, cmap=cmap, norm=norm,
                       s=8, zorder=5, edgecolors="none", alpha=0.7)
        cb = fig.colorbar(sc, ax=ax, pad=0.02, aspect=30, shrink=0.85)
        cb.set_label("Fatigue cycles, $N$", fontsize=9.5)
        cb.ax.tick_params(labelsize=8.5)

        # Start and end markers
        ax.plot(Lr_traj[0], Kr_traj[0], "o", color=_TEAL, ms=7, zorder=6,
                markeredgecolor="white", markeredgewidth=0.8, label="Start")
        ax.plot(Lr_traj[-1], Kr_traj[-1], "s", color=_DRED, ms=7, zorder=6,
                markeredgecolor="white", markeredgewidth=0.8, label="End")

        # Direction arrow at midpoint
        mid = n_pts // 2
        if mid > 0 and mid < n_pts - 1:
            dx = Lr_traj[mid+1] - Lr_traj[mid]
            dy = Kr_traj[mid+1] - Kr_traj[mid]
            ax.annotate("", xy=(Lr_traj[mid]+dx*2, Kr_traj[mid]+dy*2),
                       xytext=(Lr_traj[mid], Kr_traj[mid]),
                       arrowprops=dict(arrowstyle="->", color=_CHAR, lw=1.0),
                       zorder=7)

    # Safe/unsafe labels
    ax.text(0.12, 0.12, "SAFE", fontsize=14, color=_TEAL, alpha=0.5,
            fontweight="bold", transform=ax.transAxes)
    ax.text(0.65, 0.85, "UNSAFE", fontsize=14, color=_DRED, alpha=0.4,
            fontweight="bold", transform=ax.transAxes)

    # Set axis limits to capture full trajectory
    if len(life_result.crack_history_a) > 2:
        x_max = min(max(Lr_traj.max() * 1.1, L_r_cut * 1.15), 3.5)
        y_max = min(max(Kr_traj.max() * 1.1, 1.2), 2.0)
    else:
        x_max = L_r_cut * 1.15
        y_max = 1.3

    ax.set_xlabel(r"$L_r = \sigma_{\mathrm{ref}} \,/\, \sigma_{\mathrm{y}}$")
    ax.set_ylabel(r"$K_r = K_{\mathrm{app}} \,/\, K_{\mathrm{IC}}(C_{\mathrm{H}})$")
    ax.set_title("Failure Assessment Diagram (BS 7910 Option 1)", loc="left")
    ax.legend(fontsize=8, loc="upper left")
    ax.set_xlim(0, x_max); ax.set_ylim(0, y_max)
    ax.minorticks_on()

    _ensure_dir(save_path)
    fig.savefig(save_path); plt.close(fig)
    return save_path


# ============================================================
# 7. ANIMATED GIF: color-mapped wall concentration + crack
# ============================================================
def generate_gif(diffusion_result, life_result, geom,
                 save_path="assets/h2_crack_evolution.gif",
                 n_frames=40, fps=4):
    """
    Animated GIF with engineering-style presentation.
    Left: H2 concentration profile with filled area.
    Right: Pipe wall cross-section with color-mapped concentration
           and advancing crack at ERW seam.
    """
    _apply_style()
    import io
    from PIL import Image
    from matplotlib.patches import Rectangle
    from matplotlib.colors import Normalize
    from matplotlib.cm import ScalarMappable

    x_mm = diffusion_result.x_grid * 1000.0
    wall_mm = geom.wall_thickness_m * 1000.0
    n_av = len(diffusion_result.t_grid)
    fidxs = np.linspace(0, n_av-1, n_frames, dtype=int)
    c_max = max(diffusion_result.C_field.max() * 1.12, 0.01)
    n_x = len(x_mm)

    frames = []

    for fi, f_idx in enumerate(fidxs):
        fig = plt.figure(figsize=(12.5, 5.2))
        gs = fig.add_gridspec(1, 3, width_ratios=[1.3, 0.05, 0.8], wspace=0.08)
        ax1 = fig.add_subplot(gs[0, 0])
        ax_cb = fig.add_subplot(gs[0, 1])
        ax2 = fig.add_subplot(gs[0, 2])

        t_d = diffusion_result.t_grid[f_idx] / 86400.0
        C_p = diffusion_result.C_field[f_idx]

        frac = fi / max(n_frames-1, 1)
        if len(life_result.crack_history_a) > 2:
            cidx = min(int(frac*(len(life_result.crack_history_a)-1)),
                       len(life_result.crack_history_a)-1)
            cmm = life_result.crack_history_a[cidx] * 1000.0
            cn = life_result.crack_history_cycles[cidx]
        else:
            cmm = life_result.initial_pit_depth_m * 1000.0; cn = 0

        # Left: concentration profile
        ax1.fill_between(x_mm, 0, C_p, color=_STL, alpha=0.18)
        ax1.plot(x_mm, C_p, color=_NAV, lw=1.3)
        ax1.axvline(cmm, color=_DRED, ls="--", lw=0.7)
        ax1.set_xlim(0, wall_mm); ax1.set_ylim(0, c_max)
        ax1.set_xlabel("Depth [mm]")
        ax1.set_ylabel(r"$C_{\mathrm{H}}$ [mol m$^{-3}$]")
        ax1.set_title(f"t = {t_d:.0f} d  |  N = {cn:,}  |  a = {cmm:.2f} mm",
                      loc="left", fontsize=9.5)
        ax1.minorticks_on()

        # Right: color-mapped wall cross-section
        # Build a 2D column where each row is colored by C_H at that depth
        n_cells = n_x
        cell_height = wall_mm / n_cells
        cmap = plt.cm.YlOrRd
        norm = Normalize(vmin=0, vmax=c_max)

        ax2.set_xlim(-0.3, 3.0)
        ax2.set_ylim(-wall_mm*0.12, wall_mm*1.2)

        # Draw wall as stack of colored cells
        strip_x = 0.3
        strip_w = 1.4
        for ci in range(n_cells):
            y_bot = ci * cell_height
            c_val = C_p[min(ci, len(C_p)-1)]
            color = cmap(norm(c_val))
            rect = Rectangle((strip_x, y_bot), strip_w, cell_height,
                              facecolor=color, edgecolor="none")
            ax2.add_patch(rect)

        # Wall outline
        ax2.plot([strip_x, strip_x+strip_w, strip_x+strip_w, strip_x, strip_x],
                 [0, 0, wall_mm, wall_mm, 0], color=_CHAR, lw=1.0, zorder=3)

        # ERW seam indicator
        seam_x = strip_x + strip_w*0.4
        seam_w = strip_w * 0.2
        ax2.plot([seam_x, seam_x], [0, wall_mm], color="white", lw=0.5,
                 ls=":", alpha=0.6, zorder=4)
        ax2.plot([seam_x+seam_w, seam_x+seam_w], [0, wall_mm], color="white",
                 lw=0.5, ls=":", alpha=0.6, zorder=4)
        ax2.text(seam_x + seam_w/2, wall_mm*1.03, "ERW", fontsize=7,
                 ha="center", color=_G60, va="bottom", zorder=5)

        # Crack as dark wedge
        if cmm > 0.01:
            cx = strip_x + strip_w * 0.5
            cw_base = strip_w * 0.08 + strip_w * 0.08 * (cmm/wall_mm)
            tri_x = [cx-cw_base, cx, cx+cw_base]
            tri_y = [0, cmm, 0]
            ax2.fill(tri_x, tri_y, color=_DRED, alpha=0.85, zorder=6)
            ax2.plot(tri_x, tri_y, color="white", lw=0.4, zorder=7)

        # Depth annotations
        if cmm > 0.05:
            ann_x = strip_x + strip_w + 0.15
            ax2.annotate("", xy=(ann_x, 0), xytext=(ann_x, cmm),
                         arrowprops=dict(arrowstyle="<->", color=_DRED, lw=0.7),
                         zorder=5)
            ax2.text(ann_x + 0.1, cmm/2, f"{cmm:.2f}", fontsize=8,
                     color=_DRED, va="center", zorder=5)

        # Labels
        ax2.text(strip_x + strip_w/2, -wall_mm*0.07, r"Inner (H$_2$)",
                 ha="center", fontsize=7.5, color=_CHAR)
        ax2.text(strip_x + strip_w/2, wall_mm*1.12, "Outer",
                 ha="center", fontsize=7.5, color=_CHAR)

        ax2.set_aspect("equal"); ax2.axis("off")
        ax2.set_title("Wall cross-section", loc="left", fontsize=9.5)

        # Colorbar for wall
        sm = ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cb = fig.colorbar(sm, cax=ax_cb)
        cb.set_label(r"$C_{\mathrm{H}}$", fontsize=8)
        cb.ax.tick_params(labelsize=7)

        fig.suptitle("ICS2 Week 7 :  H$_2$ conversion of aging LF-ERW pipeline",
                     fontsize=10.5, y=0.99, color=_CHAR)

        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=150)
        buf.seek(0)
        frames.append(Image.open(buf).copy())
        buf.close(); plt.close(fig)

    _ensure_dir(save_path)
    frames[0].save(save_path, save_all=True, append_images=frames[1:],
                   duration=int(1000/fps), loop=0)
    return save_path
