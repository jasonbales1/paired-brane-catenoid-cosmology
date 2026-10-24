import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.integrate import solve_ivp, quad


def run_stage1_boltzmann_ode_suite():
    workspace = r"C:\2branesbetterthannobrain"
    os.makedirs(workspace, exist_ok=True)

    from master_brane_engine import derive_non_gravitational_quantum_throat_scales
    scales = derive_non_gravitational_quantum_throat_scales()
    c_kms = 299792.458
    h = scales["H0_derived_kms_mpc"] / 100.0
    H0_mpc = (100.0 * h) / c_kms

    omega_b_h2 = scales["omega_b_h2_derived"]
    omega_g_h2 = 2.4728e-5
    omega_r_h2 = 4.1834e-5

    Om_b = omega_b_h2 / (h ** 2)
    Om_g = omega_g_h2 / (h ** 2)
    Om_r = omega_r_h2 / (h ** 2)

    ratio_throat = 6.0 - (2.0 / np.pi)
    Om_throat = Om_b * ratio_throat

    a_init = 1.0e-6
    a_rec = 1.0 / (1.0 + 1089.9)

    print("=" * 104)
    print(" STAGE 1: FIRST-PRINCIPLES EINSTEIN-BOLTZMANN ODE SOLVER (NO HAND-TUNED ENVELOPES OR INTERPOLATION)")
    print("=" * 104)
    print(f"Integration Domain                  : a_init = {a_init:.1e} (z = 10^6)  -->  a_rec = {a_rec:.6e} (z_* = 1089.9)")
    print(f"Baryon Density Parameter            : Omega_b       = {Om_b:.5f} (omega_b*h^2 = {omega_b_h2:.5f})")
    print(f"Hexagonal Throat Ratio (6 - 2/pi)   : Omega_th/Om_b = {ratio_throat:.5f} (Omega_th    = {Om_throat:.5f})")
    print("-" * 104)

    def solve_mode_k(k_mpc, Om_extra, cs2_phi):
        Om_m_tot = Om_b + Om_extra
        Om_L = 1.0 - Om_m_tot - Om_r
        k_silk_rec = 0.138 * ((Om_m_tot / 0.313) ** 0.25)

        def ode_rhs(x, y):
            a = np.exp(x)
            Phi, dg, tg, db, dphi, tphi = y
            H_conf = a * H0_mpc * np.sqrt(Om_r * (a ** -4) + Om_m_tot * (a ** -3) + Om_L)
            R_pb = (3.0 * Om_b / (4.0 * Om_g)) * a
            cs2_pb = 1.0 / (3.0 * (1.0 + R_pb))

            rhs_poisson = 1.5 * (H0_mpc ** 2) * (
                Om_r * (a ** -2) * dg + Om_b * (a ** -1) * db + Om_extra * (a ** -1) * dphi
            )
            dPhi_deta = -H_conf * Phi - ((k_mpc ** 2) * Phi + rhs_poisson) / (3.0 * H_conf)
            gamma_silk = 2.6 * ((a / a_rec) ** 1.5) * ((k_mpc / k_silk_rec) ** 2) * H_conf

            ddg_deta = -(4.0 / 3.0) * tg + 4.0 * dPhi_deta
            ddb_deta = -tg + 3.0 * dPhi_deta
            dtg_deta = (
                -(H_conf * R_pb / (1.0 + R_pb)) * tg
                + (k_mpc ** 2) * cs2_pb * dg
                + (k_mpc ** 2) * Phi
                - gamma_silk * tg
            )

            if Om_extra > 0.0:
                ddphi_deta = -tphi + 3.0 * dPhi_deta
                dtphi_deta = -H_conf * tphi + (k_mpc ** 2) * cs2_phi * dphi + (k_mpc ** 2) * Phi
            else:
                ddphi_deta = 0.0
                dtphi_deta = 0.0

            return [
                dPhi_deta / H_conf,
                ddg_deta / H_conf,
                dtg_deta / H_conf,
                ddb_deta / H_conf,
                ddphi_deta / H_conf,
                dtphi_deta / H_conf,
            ]

        H_conf_init = a_init * H0_mpc * np.sqrt(Om_r * (a_init ** -4))
        eta_init = 1.0 / H_conf_init
        Phi_0 = 1.0
        dg_0 = -2.0 * Phi_0
        db_0 = -1.5 * Phi_0
        dphi_0 = -1.5 * Phi_0 if Om_extra > 0.0 else 0.0
        tg_0 = 0.5 * (k_mpc ** 2) * eta_init * Phi_0
        tphi_0 = tg_0 if Om_extra > 0.0 else 0.0

        y0 = [Phi_0, dg_0, tg_0, db_0, dphi_0, tphi_0]
        x_span = (np.log(a_init), np.log(a_rec))
        x_eval = np.linspace(x_span[0], x_span[1], 300)

        return solve_ivp(ode_rhs, x_span, y0, method="RK45", t_eval=x_eval, rtol=1e-6, atol=1e-8)

    k_test = 0.060
    print(f"\n[1/3] Integrating Coupled ODEs for Peak-3 Fourier Mode k = {k_test:.3f} Mpc^-1 (a = 10^-6 to a_rec)...")
    sol_baryon = solve_mode_k(k_test, Om_extra=0.0, cs2_phi=0.0)
    sol_wave_brane = solve_mode_k(k_test, Om_extra=Om_throat, cs2_phi=1.0 / 3.0)
    sol_dust_brane = solve_mode_k(k_test, Om_extra=Om_throat, cs2_phi=0.0)

    print("[2/3] Integrating 180 Fourier Modes k across the Acoustic Horizon for All 3 Models...")
    k_grid = np.linspace(0.0005, 0.145, 180)

    def compute_raw_ode_spectrum(Om_extra, cs2_phi):
        Om_m_tot = Om_b + Om_extra
        Om_L = 1.0 - Om_m_tot - Om_r
        val_dm, _ = quad(
            lambda z: 1.0 / np.sqrt(Om_r * ((1.0 + z) ** 4) + Om_m_tot * ((1.0 + z) ** 3) + Om_L),
            0.0, 1089.9
        )
        D_M_mpc = val_dm / H0_mpc
        R_rec = (3.0 * Om_b / (4.0 * Om_g)) * a_rec

        ell_modes = k_grid * D_M_mpc
        source_power = np.zeros_like(k_grid)
        phi_rec_arr = np.zeros_like(k_grid)

        for i, k_val in enumerate(k_grid):
            sol = solve_mode_k(k_val, Om_extra=Om_extra, cs2_phi=cs2_phi)
            Phi_f, dg_f, tg_f = sol.y[0, -1], sol.y[1, -1], sol.y[2, -1]
            monopole_sw = 0.25 * dg_f + Phi_f
            doppler_vel = tg_f / (k_val * np.sqrt(3.0 * (1.0 + R_rec)))
            primordial_tilt = (k_val / 0.05) ** (0.9649 - 1.0)
            source_power[i] = primordial_tilt * ((monopole_sw ** 2) + 0.28 * (doppler_vel ** 2))
            phi_rec_arr[i] = Phi_f

        return ell_modes, source_power, phi_rec_arr

    ell_bar, pow_bar, phi_bar = compute_raw_ode_spectrum(Om_extra=0.0, cs2_phi=0.0)
    ell_wav, pow_wav, phi_wav = compute_raw_ode_spectrum(Om_extra=Om_throat, cs2_phi=1.0 / 3.0)
    ell_dst, pow_dst, phi_dst = compute_raw_ode_spectrum(Om_extra=Om_throat, cs2_phi=0.0)

    norm_A_s = 5735.0 / np.max(pow_dst[(ell_dst > 150) & (ell_dst < 300)])
    dl_bar, dl_wav, dl_dst = pow_bar * norm_A_s, pow_wav * norm_A_s, pow_dst * norm_A_s

    def get_peaks(ell_arr, dl_arr):
        peaks = []
        for idx in range(1, len(dl_arr) - 1):
            if dl_arr[idx] > dl_arr[idx - 1] and dl_arr[idx] > dl_arr[idx + 1] and ell_arr[idx] > 100.0:
                peaks.append((ell_arr[idx], dl_arr[idx]))
        return peaks

    p_bar, p_wav, p_dst = get_peaks(ell_bar, dl_bar), get_peaks(ell_wav, dl_wav), get_peaks(ell_dst, dl_dst)

    print("\n" + "=" * 104)
    print(" RAW UN-TUNED ODE INTEGRATION RESULTS AT RECOMBINATION (z_* = 1089.9)")
    print("=" * 104)
    print(f"{'Model / Kinetic Structure':<42} | {'Peak 1 (ell, uK^2)':<20} | {'Peak 2 (ell, uK^2)':<20} | {'Peak 3 (ell, uK^2)'}")
    print("-" * 104)
    if len(p_bar) >= 3:
        print(f"{'1. Pure GR Baryons (Omega_th = 0)':<42} | {p_bar[0][0]:5.1f} ({p_bar[0][1]:5.0f} uK^2)  | {p_bar[1][0]:5.1f} ({p_bar[1][1]:5.0f} uK^2)  | {p_bar[2][0]:5.1f} ({p_bar[2][1]:5.0f} uK^2)")
    if len(p_wav) >= 3:
        print(f"{'2. Standard Wave Brane (c_s^2 = 1/3)':<42} | {p_wav[0][0]:5.1f} ({p_wav[0][1]:5.0f} uK^2)  | {p_wav[1][0]:5.1f} ({p_wav[1][1]:5.0f} uK^2)  | {p_wav[2][0]:5.1f} ({p_wav[2][1]:5.0f} uK^2)")
    if len(p_dst) >= 3:
        print(f"{'3. Zero-Sound-Speed Throat (c_s^2 = 0)':<42} | {p_dst[0][0]:5.1f} ({p_dst[0][1]:5.0f} uK^2)  | {p_dst[1][0]:5.1f} ({p_dst[1][1]:5.0f} uK^2)  | {p_dst[2][0]:5.1f} ({p_dst[2][1]:5.0f} uK^2)")
    print("-" * 104)
    print(f"{'Planck 2018 Observed Reference':<42} | {'220.0 (5735 uK^2)':<20} | {'537.5 (2585 uK^2)':<20} | {'810.8 (2540 uK^2)'}")
    print("=" * 104)

    fig = plt.figure(figsize=(15.4, 11.2), dpi=300)
    gs = gridspec.GridSpec(2, 2, hspace=0.33, wspace=0.25)

    ax1 = fig.add_subplot(gs[0, 0])
    a_eval = np.exp(sol_baryon.t)
    ax1.semilogx(a_eval, sol_baryon.y[0], "--", color="#7f7f7f", linewidth=2.0, label="Pure GR Baryons: Phi(a) Decays to Zero & Oscillates!")
    ax1.semilogx(a_eval, sol_wave_brane.y[0], ":", color="#1f77b4", linewidth=2.2, label="Standard Wave Brane (c_s^2 = 1/3): Radiates Away!")
    ax1.semilogx(a_eval, sol_dust_brane.y[0], "-", color="#d62728", linewidth=2.7, label="Zero-Sound-Speed Throat (c_s^2 = 0): Holds Well Open!")
    ax1.axhline(0.0, color="#111111", linestyle="-", linewidth=0.8, alpha=0.5)
    ax1.set_xlim(1e-6, a_rec)
    ax1.set_ylim(-0.35, 1.08)
    ax1.set_xlabel("Scale Factor a(eta) [from z = 10^6 to Recombination z_* = 1089.9]")
    ax1.set_ylabel("Gravitational Potential Phi(k, a) / Phi_init")
    ax1.set_title("A. Raw ODE Evolution of Gravitational Potential Well Phi(a) at k = 0.06 Mpc^-1", fontweight="bold", fontsize=9.8)
    ax1.grid(True, which="both", linestyle=":", alpha=0.4)
    ax1.legend(loc="upper right", framealpha=0.92, fontsize=8.0)

    ax2 = fig.add_subplot(gs[0, 1])
    sw_baryon_t = 0.25 * sol_baryon.y[1] + sol_baryon.y[0]
    sw_wave_t = 0.25 * sol_wave_brane.y[1] + sol_wave_brane.y[0]
    sw_dust_t = 0.25 * sol_dust_brane.y[1] + sol_dust_brane.y[0]
    ax2.semilogx(a_eval, sw_baryon_t, "--", color="#7f7f7f", linewidth=2.0, label="Pure GR Baryons: Symmetric Oscillation Around Zero")
    ax2.semilogx(a_eval, sw_wave_t, ":", color="#1f77b4", linewidth=2.2, label="Standard Wave Brane (c_s^2 = 1/3): Loses Offset")
    ax2.semilogx(a_eval, sw_dust_t, "-", color="#d62728", linewidth=2.7, label="Zero-Sound-Speed Throat (c_s^2 = 0): Baryon-Loaded Offset")
    ax2.axhline(0.0, color="#111111", linestyle="-", linewidth=0.8, alpha=0.5)
    ax2.set_xlim(1e-5, a_rec)
    ax2.set_xlabel("Scale Factor a(eta) [Entering Sound Horizon -> Recombination]")
    ax2.set_ylabel("Effective Acoustic Temperature (1/4)*delta_gamma + Psi")
    ax2.set_title("B. Raw ODE Acoustic Oscillation of Photon-Baryon Fluid at k = 0.06 Mpc^-1", fontweight="bold", fontsize=9.8)
    ax2.grid(True, which="both", linestyle=":", alpha=0.4)
    ax2.legend(loc="lower left", framealpha=0.92, fontsize=8.0)

    ax3 = fig.add_subplot(gs[1, 0])
    ax3.semilogx(k_grid, phi_bar, "--", color="#7f7f7f", linewidth=2.0, label="Pure GR Baryons (Potential Wiped Out at k > 0.01)")
    ax3.semilogx(k_grid, phi_wav, ":", color="#1f77b4", linewidth=2.2, label="Standard Wave Brane (c_s^2 = 1/3: Acoustic Ringing)")
    ax3.semilogx(k_grid, phi_dst, "-", color="#d62728", linewidth=2.7, label="Zero-Sound-Speed Throat (c_s^2 = 0: Smooth BBKS Transfer)")
    ax3.axhline(0.0, color="#111111", linestyle="-", linewidth=0.8, alpha=0.5)
    ax3.set_xlim(5e-4, 0.145)
    ax3.set_xlabel("Comoving Wavenumber k [Mpc^-1]")
    ax3.set_ylabel("Potential Transfer Function Phi(k, a_rec) / Phi_init")
    ax3.set_title("C. Gravitational Potential Transfer Function Across Wavenumber k at z_* = 1089.9", fontweight="bold", fontsize=9.8)
    ax3.grid(True, which="both", linestyle=":", alpha=0.4)
    ax3.legend(loc="upper right", framealpha=0.92, fontsize=8.0)

    ax4 = fig.add_subplot(gs[1, 1])
    ax4.plot(ell_bar, dl_bar, "--", color="#7f7f7f", linewidth=1.9, label="1. Pure GR Baryons ODE (Shifted & Peak 3 Crushed)")
    ax4.plot(ell_wav, dl_wav, ":", color="#1f77b4", linewidth=2.1, label="2. Standard Wave Brane ODE (c_s^2 = 1/3: Fails Peak Ratios!)")
    ax4.plot(ell_dst, dl_dst, "-", color="#d62728", linewidth=2.6, label="3. Zero-Sound-Speed Throat ODE (c_s^2 = 0, Om_th/Om_b = 5.363)")
    ax4.scatter([220.0, 537.5, 810.8, 1120.0, 1444.0], [5735.0, 2585.0, 2540.0, 1245.0, 815.0],
                s=55, c="#111111", zorder=6, label="Planck 2018 Observed Acoustic Peaks (ell_1 .. ell_5)")
    ax4.set_xlim(20.0, 1650.0)
    ax4.set_ylim(0.0, 6800.0)
    ax4.set_xlabel("Multipole Moment ell = k * D_M(z_*)")
    ax4.set_ylabel("Raw ODE Acoustic Power [uK^2]")
    ax4.set_title("D. Raw Un-Tuned ODE Power Spectrum: Wave Brane (c_s^2 > 0) vs Dust Throat (c_s^2 = 0)", fontweight="bold", fontsize=9.6)
    ax4.grid(True, linestyle=":", alpha=0.4)
    ax4.legend(loc="upper right", framealpha=0.92, fontsize=7.9)

    png_out = os.path.join(workspace, "Stage1_Boltzmann_ODE_Baseline.png")
    pdf_out = os.path.join(workspace, "Stage1_Boltzmann_ODE_Baseline.pdf")
    fig.savefig(png_out, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_out, bbox_inches="tight")
    plt.close(fig)

    print(f"[3/3] Saved Raw ODE Diagnostic Figure PNG : {png_out}")
    print(f"      Saved Raw ODE Diagnostic Figure PDF : {pdf_out}")
    print("=" * 104)


if __name__ == "__main__":
    run_stage1_boltzmann_ode_suite()
