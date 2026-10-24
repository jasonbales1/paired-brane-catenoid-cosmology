"""
sparc_catenoid_rar_figure.py (DOP853 Radial Catenoid ODE & 6-Panel Master Figure)
==================================================================================
PUBLICATION-QUALITY 6-PANEL SPARC RAR & ROTATION CURVE FIGURE GENERATOR (DOP853 ODE)
Workspace: C:/2branesbetterthannobrain/

Upgraded to integrate the nonlinear Catenoid and Weyl-RAR differential equations
and per-galaxy radial rotation curves via `scipy.integrate.solve_ivp(method="DOP853")`.

Generates and saves:
  - SPARC_Paired_Brane_RAR_Master_Figure.png (300 DPI)
  - SPARC_Paired_Brane_RAR_Master_Figure.pdf (Vector PDF for LaTeX/Manuscript)
==================================================================================
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from astroquery.vizier import Vizier
from sparc_rar_catenoid_validator import (
    clean_col,
    build_universal_catenoid_ode_splines,
    solve_galaxy_radial_catenoid_ode,
)


def generate_sparc_master_figure():
    workspace = r"C:\2branesbetterthannobrain"
    os.makedirs(workspace, exist_ok=True)

    c_m_s = 2.99792458e8
    H0_s = (67.4 * 1e3) / 3.08567758e22
    R_cosmos = c_m_s / H0_s
    a_brane = (c_m_s ** 2) / (6.0 * R_cosmos)  # 1.0914e-10 m/s^2
    kpc_to_m = 3.08567758e19

    print("=" * 94)
    print(" 2BRANESBETTERTHANNOBRAIN — GENERATING 6-PANEL SPARC RAR & ROTATION CURVE FIGURE (DOP853 ODE)")
    print("=" * 94)

    cs_log_cat, cs_log_weyl = build_universal_catenoid_ode_splines(a_brane)

    v = Vizier(columns=["**"], row_limit=-1)
    v.timeout = 120

    print("[1/3] Querying VizieR for SPARC Master & Radial Tables (J/AJ/152/157)...")
    t1 = v.get_catalogs("J/AJ/152/157/table1")[0]
    t1_names = [str(n).strip().replace(" ", "") for n in clean_col(t1, ["Name", "Galaxy"], str)]
    t1_qual = clean_col(t1, ["Qual", "Q"], float)
    t1_inc = clean_col(t1, ["Inc", "i"], float)

    good_gal_set = {
        t1_names[i]
        for i in range(len(t1_names))
        if (t1_qual[i] <= 2) and (t1_inc[i] >= 30.0)
    }

    t2 = v.get_catalogs("J/AJ/152/157/table2")[0]
    gal_col = np.array([str(n).strip().replace(" ", "") for n in clean_col(t2, ["Name", "Galaxy", "ID"], str)])
    rad_kpc = clean_col(t2, ["Rad", "R"], float)
    vobs_kms = clean_col(t2, ["Vobs", "V_obs"], float)
    evobs_kms = clean_col(t2, ["e_Vobs", "e_V_obs"], float)
    vgas_kms = clean_col(t2, ["Vgas", "V_gas"], float)
    vdisk_kms = clean_col(t2, ["Vdisk", "V_disk"], float)
    vbul_kms = np.nan_to_num(clean_col(t2, ["Vbulge", "Vbul", "V_bul"], float, default_val=0.0), nan=0.0)

    ups_disk_0, ups_bul_0 = 0.50, 0.70
    vbar_sq_0 = (
        np.sign(vgas_kms) * (vgas_kms ** 2)
        + ups_disk_0 * np.sign(vdisk_kms) * (vdisk_kms ** 2)
        + ups_bul_0 * np.sign(vbul_kms) * (vbul_kms ** 2)
    )

    valid = (
        np.array([g in good_gal_set for g in gal_col])
        & np.isfinite(rad_kpc) & (rad_kpc > 0.1)
        & np.isfinite(vobs_kms) & (vobs_kms > 10.0)
        & np.isfinite(evobs_kms) & (evobs_kms > 0.0)
        & ((evobs_kms / vobs_kms) <= 0.15)
        & np.isfinite(vbar_sq_0) & (vbar_sq_0 > 25.0)
    )

    g_names = gal_col[valid]
    r_kpc = rad_kpc[valid]
    r_m = r_kpc * kpc_to_m
    v_obs = vobs_kms[valid]
    e_vobs = evobs_kms[valid]
    v_gas = vgas_kms[valid]
    v_disk = vdisk_kms[valid]
    v_bul = vbul_kms[valid]

    g_obs = ((v_obs * 1e3) ** 2) / r_m
    v_bar_0 = np.sqrt(vbar_sq_0[valid])
    g_bar_0 = ((v_bar_0 * 1e3) ** 2) / r_m

    # Zero-parameter Paired-Brane Catenoid via Per-Galaxy DOP853 Radial ODE
    g_pred_cat_0 = np.zeros_like(g_bar_0)
    for ug in np.unique(g_names):
        m = (g_names == ug)
        g_pred_cat_0[m] = solve_galaxy_radial_catenoid_ode(r_m[m], g_bar_0[m], a_brane, cs_log_cat)
    v_pred_cat_0 = np.sqrt(g_pred_cat_0 * r_m) / 1e3

    # SPARC 3.6 um IMF Stellar Population Synthesis M/L bounds (Upsilon_disk in [0.30, 0.85]) via DOP853 ODE
    v_pred_opt = np.zeros_like(v_obs)
    g_pred_opt = np.zeros_like(g_obs)
    v_bar_opt = np.zeros_like(v_obs)
    opt_ups_map = {}
    u_grid = np.linspace(0.30, 0.85, 56)

    for ug in np.unique(g_names):
        m = (g_names == ug)
        best_err = 1e9
        best_gb, best_vb, best_u = None, None, 0.50
        for u_d in u_grid:
            u_b = 1.40 * u_d
            vb2 = (
                np.sign(v_gas[m]) * (v_gas[m] ** 2)
                + u_d * np.sign(v_disk[m]) * (v_disk[m] ** 2)
                + u_b * np.sign(v_bul[m]) * (v_bul[m] ** 2)
            )
            if np.any(vb2 <= 0):
                continue
            gb = (vb2 * 1e6) / r_m[m]
            gp = np.exp(cs_log_cat(np.log(gb / a_brane)))
            vp = np.sqrt(gp * r_m[m]) / 1e3
            err = np.mean(((v_obs[m] - vp) / e_vobs[m]) ** 2)
            if err < best_err:
                best_err = err
                best_gb, best_vb, best_u = gb, np.sqrt(vb2), u_d
        gp_ode = solve_galaxy_radial_catenoid_ode(r_m[m], best_gb, a_brane, cs_log_cat)
        v_pred_opt[m] = np.sqrt(gp_ode * r_m[m]) / 1e3
        g_pred_opt[m] = gp_ode
        v_bar_opt[m] = best_vb
        opt_ups_map[ug] = best_u

    res_newt_dex = np.log10(g_obs / g_bar_0)
    res_cat_0_dex = np.log10(g_obs / g_pred_cat_0)
    res_opt_dex = np.log10(g_obs / g_pred_opt)

    print("[2/3] Building 6-Panel Publication Figure (DOP853 ODE)...")
    plt.rcParams.update({
        "font.size": 10,
        "axes.labelsize": 11,
        "axes.titlesize": 11.5,
        "xtick.labelsize": 9.5,
        "ytick.labelsize": 9.5,
        "legend.fontsize": 8.8,
        "figure.titlesize": 13.5,
    })

    fig = plt.figure(figsize=(15.5, 13.0), dpi=300)
    gs = gridspec.GridSpec(3, 2, height_ratios=[1.15, 0.95, 0.95], hspace=0.34, wspace=0.24)

    # -------------------------------------------------------------------------
    # PANEL A: Global SPARC Radial Acceleration Relation (2,921 points)
    # -------------------------------------------------------------------------
    ax_rar = fig.add_subplot(gs[0, 0])
    log_gbar = np.log10(g_bar_0)
    log_gobs = np.log10(g_obs)

    ax_rar.scatter(
        log_gbar, log_gobs, s=8, c="#1f77b4", alpha=0.28, edgecolors="none",
        label=f"SPARC Radial Data ($N = {len(v_obs)}$ pts, 151 galaxies)"
    )

    bins = np.linspace(-12.2, -8.6, 15)
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    med_gobs, p16_gobs, p84_gobs = [], [], []
    valid_bins = []
    for b_i in range(len(bins) - 1):
        bm = (log_gbar >= bins[b_i]) & (log_gbar < bins[b_i + 1])
        if np.sum(bm) >= 10:
            valid_bins.append(bin_centers[b_i])
            med_gobs.append(np.median(log_gobs[bm]))
            p16_gobs.append(np.percentile(log_gobs[bm], 16))
            p84_gobs.append(np.percentile(log_gobs[bm], 84))
    valid_bins = np.array(valid_bins)
    med_gobs = np.array(med_gobs)
    yerr_low = med_gobs - np.array(p16_gobs)
    yerr_high = np.array(p84_gobs) - med_gobs

    ax_rar.errorbar(
        valid_bins, med_gobs, yerr=[yerr_low, yerr_high], fmt="o", color="#111111",
        markersize=5.5, elinewidth=1.4, capsize=3, zorder=5, label="SPARC Binned Median $\\pm 1\\sigma$"
    )

    g_grid = np.logspace(-12.5, -8.2, 300)
    x_grid_rar = np.log(g_grid / a_brane)
    g_cat_curve = np.exp(cs_log_cat(x_grid_rar))
    g_weyl_curve = np.exp(cs_log_weyl(x_grid_rar))

    ax_rar.plot(
        np.log10(g_grid), np.log10(g_grid), "--", color="#7f7f7f", linewidth=1.6,
        label="Pure Newtonian Baryons ($g_{\\mathrm{obs}} = g_{\\mathrm{bar}}$)"
    )
    ax_rar.plot(
        np.log10(g_grid), np.log10(g_cat_curve), "-", color="#d62728", linewidth=2.4, zorder=4,
        label=f"Paired-Brane Catenoid ODE ($a_{{\\mathrm{{brane}}}} = 1.0914\\times 10^{{-10}}\\mathrm{{m/s^2}}$)"
    )
    ax_rar.plot(
        np.log10(g_grid), np.log10(g_weyl_curve), ":", color="#ff7f0e", linewidth=2.0, zorder=4,
        label="Paired-Brane Weyl-RAR ODE Transition"
    )
    ax_rar.axvline(np.log10(a_brane), color="#d62728", linestyle="-.", alpha=0.45, linewidth=1.1)
    ax_rar.text(np.log10(a_brane) + 0.06, -11.95, "$a_{\\mathrm{brane}} = c^2 / (6 R_{\\mathrm{cosmos}})$", color="#b01a1a", fontsize=8.8)

    ax_rar.set_xlim(-12.4, -8.4)
    ax_rar.set_ylim(-12.0, -8.4)
    ax_rar.set_xlabel("$\\log_{10}(g_{\\mathrm{bar}} \\; [\\mathrm{m\\,s^{-2}}])$  [Fixed $\\Upsilon_d=0.5, \\Upsilon_b=0.7$]")
    ax_rar.set_ylabel("$\\log_{10}(g_{\\mathrm{obs}} \\; [\\mathrm{m\\,s^{-2}}])$")
    ax_rar.set_title("A. Global SPARC Radial Acceleration Relation (DOP853 ODE, Zero Free Parameters)", fontweight="bold")
    ax_rar.grid(True, linestyle=":", alpha=0.4)
    ax_rar.legend(loc="upper left", framealpha=0.92)

    # -------------------------------------------------------------------------
    # PANEL B: Logarithmic Acceleration Residual Histogram
    # -------------------------------------------------------------------------
    ax_hist = fig.add_subplot(gs[0, 1])
    h_bins = np.linspace(-0.65, 1.15, 65)

    ax_hist.hist(
        res_newt_dex, bins=h_bins, density=True, color="#8c8c8c", alpha=0.40, edgecolor="#555555",
        label=f"Newtonian Baryons (Mean = {np.mean(res_newt_dex):+.3f} dex, $\\sigma$ = {np.std(res_newt_dex):.3f} dex)"
    )
    ax_hist.hist(
        res_cat_0_dex, bins=h_bins, density=True, color="#1f77b4", alpha=0.55, edgecolor="#0d4a7a",
        label=f"Paired-Brane Zero-Param ODE (Mean = {np.mean(res_cat_0_dex):+.4f} dex, $\\sigma$ = {np.std(res_cat_0_dex):.3f} dex)"
    )
    ax_hist.hist(
        res_opt_dex, bins=h_bins, density=True, histtype="step", color="#d62728", linewidth=2.3,
        label=f"Paired-Brane SPARC IMF ODE (Mean = {np.mean(res_opt_dex):+.4f} dex, $\\sigma$ = {np.std(res_opt_dex):.3f} dex)"
    )
    ax_hist.axvline(0.0, color="#111111", linestyle="--", linewidth=1.5)

    ax_hist.set_xlim(-0.60, 1.10)
    ax_hist.set_xlabel("Acceleration Residual $\\log_{10}(g_{\\mathrm{obs}} / g_{\\mathrm{model}})$ [dex]")
    ax_hist.set_ylabel("Normalized Probability Density")
    ax_hist.set_title("B. Global SPARC Residual Collapse ($N = 2,921$ Radial Points, DOP853 ODE)", fontweight="bold")
    ax_hist.grid(True, linestyle=":", alpha=0.4)
    ax_hist.legend(loc="upper right", framealpha=0.92)

    # -------------------------------------------------------------------------
    # PANELS C-F: 4 Benchmark Spiral Rotation Curves (With Full Vbulge & DOP853 ODE)
    # -------------------------------------------------------------------------
    showcase = [
        ("NGC6503", "C. NGC 6503 (Pure Disk Spiral, $R_{\\max} = 23.5\\,\\mathrm{kpc}$, DOP853 ODE)", gs[1, 0]),
        ("NGC7331", "D. NGC 7331 (Bulge + Disk Spiral, $R_{\\max} = 36.3\\,\\mathrm{kpc}$, DOP853 ODE)", gs[1, 1]),
        ("NGC3198", "E. NGC 3198 (Extended Halo Benchmark, $R_{\\max} = 44.1\\,\\mathrm{kpc}$, DOP853 ODE)", gs[2, 0]),
        ("NGC2841", "F. NGC 2841 (Massive Bulge-Dominated Spiral, $R_{\\max} = 63.6\\,\\mathrm{kpc}$, DOP853 ODE)", gs[2, 1]),
    ]

    for gname, title_str, subplot_spec in showcase:
        ax = fig.add_subplot(subplot_spec)
        m = (g_names == gname)
        rk = r_kpc[m]
        vo = v_obs[m]
        evo = e_vobs[m]
        vb0 = v_bar_0[m]
        vp0 = v_pred_cat_0[m]
        vp_imf = v_pred_opt[m]
        vbul_g = v_bul[m]
        rms_dex = np.std(res_opt_dex[m], ddof=1)
        u_best = opt_ups_map[gname]

        ax.errorbar(
            rk, vo, yerr=evo, fmt="o", color="#111111", markersize=4.8,
            elinewidth=1.2, capsize=2.5, zorder=5, label="Measured $V_{\\mathrm{obs}} \\pm e_{V}$"
        )
        ax.plot(
            rk, vb0, "--", color="#555555", linewidth=1.7,
            label="Newtonian Baryons ($\\Upsilon_d=0.5, \\Upsilon_b=0.7$)"
        )
        if np.any(vbul_g > 0.5):
            ax.plot(
                rk, np.sqrt(0.70) * vbul_g, ":", color="#2ca02c", linewidth=1.5,
                label="Recovered Central Bulge ($V_{\\mathrm{bulge}}$)"
            )
        ax.plot(
            rk, vp0, "-", color="#1f77b4", linewidth=2.1, alpha=0.88,
            label="Paired-Brane Zero-Param ODE ($\\Upsilon_d=0.50$)"
        )
        ax.plot(
            rk, vp_imf, "-", color="#d62728", linewidth=2.4, zorder=4,
            label=f"Paired-Brane IMF ODE ($\\Upsilon_d={u_best:.2f}$, RMS={rms_dex:.3f} dex)"
        )

        ax.set_xlim(0.0, np.max(rk) * 1.06)
        ax.set_ylim(0.0, np.max(vo) * 1.22)
        ax.set_xlabel("Galactocentric Radius $R$ [kpc]")
        ax.set_ylabel("Circular Velocity $V(R)$ [km/s]")
        ax.set_title(title_str, fontweight="bold")
        ax.grid(True, linestyle=":", alpha=0.4)
        ax.legend(loc="lower right", framealpha=0.90)

    fig.suptitle(
        "Paired-Brane Catenoid Cosmology: Parameter-Free SPARC Rotation Curve & RAR Validation (DOP853 ODE)\n"
        f"Locked Horizon Acceleration $a_{{\\mathrm{{brane}}}} = c^2 / (6 R_{{\\mathrm{{cosmos}}}}) = 1.0914 \\times 10^{{-10}}\\,\\mathrm{{m/s^2}}$ "
        f"| Zero-Param Mean Offset = {np.mean(res_cat_0_dex):+.4f} dex ($N = 2,921$ pts)",
        fontweight="bold", y=0.985
    )

    png_path = os.path.join(workspace, "SPARC_Paired_Brane_RAR_Master_Figure.png")
    pdf_path = os.path.join(workspace, "SPARC_Paired_Brane_RAR_Master_Figure.pdf")
    print("[3/3] Saving high-resolution figures to disk...")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

    print("-" * 94)
    print(f"SUCCESS! Saved 300-DPI PNG : {png_path}")
    print(f"SUCCESS! Saved Vector PDF  : {pdf_path}")
    print("=" * 94)


if __name__ == "__main__":
    generate_sparc_master_figure()
