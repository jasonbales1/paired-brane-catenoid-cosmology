"""
generate_paper_figures.py
================================================================================
PUBLICATION FIGURE GENERATOR FOR PAIRED-BRANE CHIRAL THROAT MANUSCRIPT
Workspace: C:/2branesbetterthannobrain/

Generates high-resolution (300 DPI PNG + Vector PDF) publication figures from
the verified terminal outputs of master_brane_engine.py and 
real_sky_chiral_halo_crossmatch.py:
  - Panel A: Lepton, Quark, and Proton Mass Hierarchy vs. CODATA/PDG Benchmarks
  - Panel B: 720-Degree SU(2) Spin-1/2 Catenoid Tether Holonomy & Strain Curve
  - Panel C: Blind Pre-Registered SPARC Spiral Birefringence vs. <cos(i)>
  - Panel D: 3-Zone Radial Halo Impact Parameter Tomography (1.5 - 60 kpc)
================================================================================
"""

import os
import numpy as np
import matplotlib.pyplot as plt


def generate_all_paper_figures():
    # Set clean publication typography and styling
    plt.rcParams.update({
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
        "axes.labelsize": 10,
        "axes.labelweight": "bold",
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 8.5,
        "figure.titlesize": 13,
        "figure.titleweight": "bold",
    })

    fig, axes = plt.subplots(2, 2, figsize=(14.5, 11.0), dpi=300)
    fig.suptitle(
        "Paired-Brane Chiral Catenoid Framework: Micro-Scale Fermion Quantization & Macro-Scale VLA/SPARC Polarimetry",
        y=0.98,
    )

    # =========================================================================
    # PANEL A: Charged Lepton, Baryon (Proton), and 6-Flavor Quark Mass Ladder
    # =========================================================================
    ax_a = axes[0, 0]

    particles = [
        "e-", "u", "d", "s", "mu-", "p (uud)", "c", "tau-", "b", "t"
    ]
    # Paired-Brane Engine Predictions (MeV)
    m_e = 0.51099895
    brane_mev = np.array([
        1.000 * m_e,          # Electron (0.511 MeV)
        2.16,                 # Up quark (2.16 MeV)
        4.77,                 # Down quark (4.77 MeV)
        93.51,                # Strange quark (93.51 MeV)
        206.781 * m_e,        # Muon (105.665 MeV)
        1836.291932 * m_e,    # Proton 3-throat junction (938.343 MeV)
        1300.90,              # Charm quark (1300.90 MeV)
        3478.455 * m_e,       # Tau lepton (1777.487 MeV)
        4178.28,              # Bottom quark (4178.28 MeV)
        172601.03,            # Top quark (172,601.03 MeV)
    ])
    # Empirical CODATA / PDG Benchmarks (MeV)
    codata_mev = np.array([
        0.51099895,           # Electron
        2.16,                 # Up
        4.67,                 # Down
        93.40,                # Strange
        206.768283 * m_e,     # Muon (105.658 MeV)
        1836.152673 * m_e,    # Proton (938.272 MeV)
        1270.0,               # Charm
        3477.228 * m_e,       # Tau (1776.860 MeV)
        4180.0,               # Bottom
        172690.0,             # Top
    ])

    x_idx = np.arange(len(particles))
    ax_a.plot(
        x_idx, codata_mev, "o-", color="#1f4e78", lw=1.8, markersize=8,
        label="CODATA / PDG Empirical Mass (MeV)"
    )
    ax_a.plot(
        x_idx, brane_mev, "s--", color="#d95f02", lw=1.5, markersize=6.5,
        markerfacecolor="none", markeredgewidth=1.8,
        label="Paired-Brane Waveguide Prediction (MeV)"
    )

    ax_a.set_yscale("log")
    ax_a.set_xticks(x_idx)
    ax_a.set_xticklabels(particles, rotation=25)
    ax_a.set_ylabel("Rest Mass Energy (MeV, Log Scale)")
    ax_a.set_title("[A] Lepton (W_n=2n+1) & SU(3) 3-Throat Baryon/Quark Mass Spectrum")
    ax_a.grid(True, which="both", linestyle=":", alpha=0.45)
    ax_a.legend(loc="upper left")

    # Annotate key precision locks
    ax_a.annotate(
        "Lepton Lock:\nmu: 99.994%\ntau: 99.965%\nProton (3! pi^5): 99.992%",
        xy=(5, brane_mev[5]), xytext=(4.3, 4.0),
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#f8f9fa", edgecolor="#1f4e78", alpha=0.9),
        arrowprops=dict(arrowstyle="->", color="#1f4e78", lw=1.2),
        fontsize=8.5,
    )

    # =========================================================================
    # PANEL B: 720-Degree SU(2) Spin-1/2 Tether Holonomy Curve
    # =========================================================================
    ax_b = axes[0, 1]

    theta_deg = np.linspace(0, 720, 500)
    theta_rad = np.radians(theta_deg)
    tether_strain = np.sin(theta_rad / 4.0) ** 2
    spinor_re = np.cos(theta_rad / 2.0)

    # Discrete 90-degree checkpoints from master_brane_engine.py
    chk_deg = np.arange(0, 721, 90)
    chk_strain = np.sin(np.radians(chk_deg) / 4.0) ** 2

    ax_b.plot(
        theta_deg, tether_strain, "-", color="#b22222", lw=2.4,
        label=r"Inter-Brane Tether Shear Strain $\mathcal{S}(\theta) = \sin^2(\theta/4)$"
    )
    ax_b.plot(
        chk_deg, chk_strain, "o", color="#b22222", markersize=7
    )
    ax_b.plot(
        theta_deg, spinor_re, "--", color="#2b5c8f", lw=1.8,
        label=r"$SU(2)$ Spinor Real Phase $\mathrm{Re}(\psi) = \cos(\theta/2)$"
    )

    ax_b.axvline(360, color="#444444", linestyle=":", lw=1.4)
    ax_b.axvline(720, color="#2e8b57", linestyle=":", lw=1.4)
    ax_b.axhline(0, color="gray", lw=0.8, alpha=0.5)

    ax_b.annotate(
        "360 deg: Locked Mobius Kink\n(Strain = 1.00, Phase = -psi)",
        xy=(360, 1.0), xytext=(175, 0.72),
        bbox=dict(boxstyle="round,pad=0.35", facecolor="#fff3cd", edgecolor="#b22222"),
        arrowprops=dict(arrowstyle="->", color="#b22222", lw=1.3),
        fontsize=8.5,
    )
    ax_b.annotate(
        "720 deg: Slipped Over Neck\n(Strain = 0.00, Phase = +psi)",
        xy=(720, 0.0), xytext=(455, -0.55),
        bbox=dict(boxstyle="round,pad=0.35", facecolor="#d4edda", edgecolor="#2e8b57"),
        arrowprops=dict(arrowstyle="->", color="#2e8b57", lw=1.3),
        fontsize=8.5,
    )

    ax_b.set_xlim(0, 740)
    ax_b.set_ylim(-1.15, 1.20)
    ax_b.set_xticks(np.arange(0, 721, 90))
    ax_b.set_xlabel("Catenoid Waist Rotation Angle (Degrees)")
    ax_b.set_ylabel("Normalized Tether Strain / Spinor Phase")
    ax_b.set_title("[B] 720-Degree Paired-Brane Tether Holonomy (SU(2) Spin-1/2 Geometry)")
    ax_b.grid(True, linestyle=":", alpha=0.45)
    ax_b.legend(loc="lower left")

    # =========================================================================
    # PANEL C: Blind Pre-Registered 10-Galaxy Parity Twist vs. <cos(i)>
    # =========================================================================
    ax_c = axes[1, 0]

    gal_labels = [
        "NGC4214", "NGC6946", "NGC0300", "UGC02953", "NGC2403",
        "NGC3109", "NGC6503", "IC2574", "NGC0055", "NGC3972"
    ]
    cos_i_vals = np.array([0.906, 0.788, 0.743, 0.643, 0.454, 0.342, 0.276, 0.259, 0.225, 0.225])
    parity_twists = np.array([-11.251, -27.322, -1.118, -37.685, -10.630, +3.143, -44.026, +3.883, -2.273, +43.053])
    n_jets_gal = np.array([7, 1, 9, 1, 1, 5, 1, 2, 10, 1])

    colors_c = ["#1b7837" if val < 0 else "#d95f02" for val in parity_twists]
    sizes_c = 55 + 18 * n_jets_gal

    ax_c.axhline(0.0, color="black", linestyle="--", lw=1.1, label="Null Birefringence (0.0 deg)")
    ax_c.axhline(
        -20.5071, color="#1b7837", linestyle="-", lw=1.8,
        label="Cluster-Balanced Stack: -20.51 +/- 2.63 deg (-7.80 sigma)"
    )
    ax_c.axhspan(-20.5071 - 2.6283, -20.5071 + 2.6283, color="#1b7837", alpha=0.15)
    ax_c.axvspan(0.45, 0.96, color="#e6f5d0", alpha=0.35, label="Face-On Throat Window (<cos i> > 0.45: 5/5 Negative)")

    ax_c.scatter(cos_i_vals, parity_twists, s=sizes_c, c=colors_c, edgecolors="black", zorder=4, alpha=0.9)

    for g_txt, cx, cy in zip(gal_labels, cos_i_vals, parity_twists):
        offset_y = 2.8 if cy >= 0 else -4.8
        ax_c.annotate(g_txt, (cx, cy), xytext=(cx - 0.035, cy + offset_y), fontsize=8, fontweight="bold")

    ax_c.set_xlim(0.15, 0.96)
    ax_c.set_ylim(-55.0, 55.0)
    ax_c.set_xlabel("Line-of-Sight Catenoid Throat Projection <cos(Inc)>")
    ax_c.set_ylabel("Chiral Parity Twist chi_spin * d_theta_0 (deg)")
    ax_c.set_title("[C] Blind Pre-Registered SPARC Spirals (N=38 Jets, 7/10 Halos Negative)")
    ax_c.grid(True, linestyle=":", alpha=0.45)
    ax_c.legend(loc="upper right", fontsize=8)

    # =========================================================================
    # PANEL D: 3-Zone Radial Halo Impact Parameter Tomography (1.5 - 60 kpc)
    # =========================================================================
    ax_d = axes[1, 1]

    # Radial bin centers and widths from Step 7 terminal output
    r_centers = np.array([9.75, 26.5, 47.5])
    r_halfwidths = np.array([8.25, 8.5, 12.5])
    r_twists = np.array([-43.693, -21.200, -14.672])
    r_errs = np.array([8.467, 5.216, 3.124])
    r_snrs = ["-5.16 sigma (N=4)", "-4.06 sigma (N=10)", "-4.70 sigma (N=24)"]

    # Smooth reference throat relaxation curve
    b_grid = np.linspace(2.0, 60.0, 300)
    model_curve = -14.0 * (1.0 + 2.4 * np.exp(-b_grid / 16.0)) * np.tanh(b_grid / 3.0)

    ax_d.axhline(0.0, color="black", linestyle="--", lw=1.1, label="Scrambled-Spin Null (+0.17 deg)")
    ax_d.plot(
        b_grid, model_curve, "-", color="#2b5c8f", lw=2.0, alpha=0.8,
        label="Catenoid Throat-to-Sheet Relaxation Profile"
    )
    ax_d.errorbar(
        r_centers, r_twists, xerr=r_halfwidths, yerr=r_errs,
        fmt="s", color="#b22222", ecolor="#b22222", elinewidth=2.0, capsize=5,
        markersize=8, zorder=5, label="VLA VIII/65/nvss Cluster-Balanced Tomography"
    )

    for rc, rt, re, slbl in zip(r_centers, r_twists, r_errs, r_snrs):
        ax_d.annotate(
            slbl, xy=(rc, rt), xytext=(rc - 5.5, rt - re - 5.5),
            fontsize=8.5, fontweight="bold", color="#b22222"
        )

    ax_d.set_xlim(0, 62)
    ax_d.set_ylim(-62.0, 15.0)
    ax_d.set_xlabel("Halo Impact Parameter b (kpc)")
    ax_d.set_ylabel("Achromatic lambda^2 -> 0 Twist (deg)")
    ax_d.set_title("[D] Radial Impact Parameter Tomography Across SPARC Halos")
    ax_d.grid(True, linestyle=":", alpha=0.45)
    ax_d.legend(loc="lower right")

    plt.tight_layout(rect=[0, 0.01, 1, 0.95])

    out_png = "paired_brane_paper_figures.png"
    out_pdf = "paired_brane_paper_figures.pdf"
    plt.savefig(out_png, dpi=300)
    plt.savefig(out_pdf)
    print("=" * 80)
    print(" 2BRANESBETTERTHANNOBRAIN — PUBLICATION FIGURE GENERATOR COMPLETE")
    print("=" * 80)
    print(f"  -> Saved High-Resolution PNG : {os.path.abspath(out_png)}")
    print(f"  -> Saved Vector PDF          : {os.path.abspath(out_pdf)}")
    print("=" * 80)


if __name__ == "__main__":
    generate_all_paper_figures()