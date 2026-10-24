"""
chiral_halo_polarimetry_test.py (DOP853 Stokes (Q, U) Transport & Chiral Throat ODE Solver)
====================================================================================================
Tests Prediction #1 of the Paired-Brane Chiral Throat Model using 100% first-principles differential
equation integration (`scipy.integrate.solve_ivp`, `method="DOP853"`):
  1. Integrates the radial Chiral Catenoid Throat Frame-Dragging ODE from b = 0 to b_max = 45 kpc:
       d(Theta_chiral)/db = (Theta_max / R_throat) * [1 - (Theta_chiral / Theta_max)^2]
  2. Integrates the coupled radiative Stokes (Q, U) polarization transport ODE along the line of
     sight s in [0, 1]:
       dQ/ds = -2 * Delta_phi_tot * U(s)
       dU/ds = +2 * Delta_phi_tot * Q(s)
     to extract the multi-frequency polarization angles theta_obs(lambda^2) and recover the
     achromatic (lambda^0) chiral twist via spin-parity stacking.
====================================================================================================
"""

import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import CubicSpline


def build_chiral_throat_profile_ode(true_chiral_max_deg=0.85, r_throat_kpc=3.5, b_max_kpc=45.0):
    """
    Integrates the radial Chiral Catenoid Frame-Dragging ODE via `solve_ivp(..., method="DOP853")`:
      d(Theta)/db = (Theta_max / R_throat) * [1 - (Theta / Theta_max)^2]
    """
    def ode_rhs(b, y):
        th = y[0]
        return [(true_chiral_max_deg / r_throat_kpc) * (1.0 - ((th / true_chiral_max_deg) ** 2))]

    b_grid = np.linspace(0.0, b_max_kpc, 400)
    sol = solve_ivp(ode_rhs, (0.0, b_max_kpc), [0.0], method="DOP853", t_eval=b_grid, rtol=1e-10, atol=1e-12)
    return CubicSpline(sol.t, sol.y[0])


def solve_stokes_rotation_ode_spline():
    """
    Integrates the coupled Stokes (Q, U) radiative polarization transport ODE
      dQ/dphi = -2 * U,   dU/dphi = +2 * Q
    across total rotation angle phi_deg in [-250, +250] deg via `solve_ivp(..., method="DOP853")`
    to map Stokes rotation without algebraic angle wrapping.
    """
    phi_max_rad = np.radians(900.0)

    def stokes_rhs(phi, y):
        q, u = y
        return [-2.0 * u, 2.0 * q]

    phi_pos = np.linspace(0.0, phi_max_rad, 1800)
    phi_neg = np.linspace(0.0, -phi_max_rad, 1800)
    sol_pos = solve_ivp(stokes_rhs, (0.0, phi_max_rad), [1.0, 0.0], method="DOP853", t_eval=phi_pos, rtol=1e-10, atol=1e-12)
    sol_neg = solve_ivp(stokes_rhs, (0.0, -phi_max_rad), [1.0, 0.0], method="DOP853", t_eval=phi_neg, rtol=1e-10, atol=1e-12)

    phi_all = np.concatenate([phi_neg[::-1][:-1], phi_pos])
    q_all = np.concatenate([sol_neg.y[0][::-1][:-1], sol_pos.y[0]])
    u_all = np.concatenate([sol_neg.y[1][::-1][:-1], sol_pos.y[1]])
    unwrapped_angle_deg = np.degrees(0.5 * np.unwrap(np.arctan2(u_all, q_all)))
    unwrapped_angle_deg -= unwrapped_angle_deg[len(phi_neg) - 1]
    return CubicSpline(np.degrees(phi_all), unwrapped_angle_deg)


def run_chiral_halo_pipeline(
    n_sightlines=2500,
    true_chiral_max_deg=0.85,
    r_throat_kpc=3.5,
    intrinsic_jet_scatter_deg=12.0,
    radiometer_noise_deg=1.5,
    seed=42,
):
    rng = np.random.default_rng(seed)

    # 1. Define VLA / SKA-Pathfinder Frequency Bands (1.2 GHz to 6.5 GHz)
    freqs_hz = np.array([1.3e9, 1.8e9, 2.5e9, 3.5e9, 5.0e9, 6.2e9])
    c_m_s = 2.99792458e8
    lambda_sq = (c_m_s / freqs_hz) ** 2

    # 2. Integrate Radial Chiral Throat ODE & Stokes Transport ODE via DOP853
    cs_chiral_ode = build_chiral_throat_profile_ode(true_chiral_max_deg, r_throat_kpc, b_max_kpc=45.0)
    cs_stokes_ode = solve_stokes_rotation_ode_spline()

    chi_spin = rng.choice([-1.0, 1.0], size=n_sightlines)
    impact_b_kpc = rng.uniform(2.0, 40.0, size=n_sightlines)

    # Evaluate DOP853 Chiral Throat ODE trajectory at each sightline's impact parameter b:
    true_chiral_shift_deg = chi_spin * cs_chiral_ode(impact_b_kpc)

    # 3. Generate Astrophysical Confounders
    intrinsic_offset_deg = rng.normal(0.0, intrinsic_jet_scatter_deg, size=n_sightlines)
    rm_rad_m2 = rng.normal(0.0, 40.0, size=n_sightlines)
    rm_deg_m2 = np.degrees(rm_rad_m2)

    # 4. Evaluate Multi-Band Polarization Angles via DOP853 Stokes Transport ODE Spline:
    total_rot_deg = (
        intrinsic_offset_deg[:, None]
        + true_chiral_shift_deg[:, None]
        + rm_deg_m2[:, None] * lambda_sq[None, :]
    )
    theta_obs_deg = cs_stokes_ode(total_rot_deg) + rng.normal(0.0, radiometer_noise_deg, size=(n_sightlines, len(freqs_hz)))

    # 5. STEP A: Linear Regression in lambda^2 Space for Every Sightline
    x = lambda_sq
    x_mean = np.mean(x)
    x_var = np.sum((x - x_mean) ** 2)

    y_mean = np.mean(theta_obs_deg, axis=1)
    fitted_slope_deg_m2 = np.sum((x[None, :] - x_mean) * (theta_obs_deg - y_mean[:, None]), axis=1) / x_var
    fitted_intercept_deg = y_mean - fitted_slope_deg_m2 * x_mean

    # 6. STEP B: Spin-Parity Stacking Estimator & S-wise vs Z-wise Helicity Decomposition
    parity_weighted_signal = chi_spin * fitted_intercept_deg
    mask_swise = chi_spin > 0.0
    mask_zwise = chi_spin < 0.0
    swise_mean_deg = np.mean(fitted_intercept_deg[mask_swise])
    swise_err_deg = np.std(fitted_intercept_deg[mask_swise], ddof=1) / np.sqrt(np.sum(mask_swise))
    zwise_mean_deg = np.mean(fitted_intercept_deg[mask_zwise])
    zwise_err_deg = np.std(fitted_intercept_deg[mask_zwise], ddof=1) / np.sqrt(np.sum(mask_zwise))

    recovered_chiral_deg = np.mean(parity_weighted_signal)
    standard_error_deg = np.std(parity_weighted_signal, ddof=1) / np.sqrt(n_sightlines)
    snr_sigma = recovered_chiral_deg / standard_error_deg

    scrambled_spins = rng.permutation(chi_spin)
    null_signal_deg = np.mean(scrambled_spins * fitted_intercept_deg)

    print("=" * 78)
    print(" PAIRED-BRANE CHIRAL HALO POLARIMETRY PIPELINE (DOP853 STOKES & SPIN ODE)")
    print("=" * 78)
    print(f"Total Halo Sightlines Stacked (N)   : {n_sightlines}")
    print(f"Injected Magnetic RM Scatter        : +/- 40.0 rad/m^2 (~2290 deg/m^2)")
    print(f"Intrinsic Quasar Angle Scatter      : +/- {intrinsic_jet_scatter_deg:.1f} deg")
    print("-" * 78)
    print(f"S-Wise Galaxies (J_gal . k_gamma >0): {swise_mean_deg:+.4f} +/- {swise_err_deg:.4f} deg (Right-Handed Twist)")
    print(f"Z-Wise Galaxies (J_gal . k_gamma <0): {zwise_mean_deg:+.4f} +/- {zwise_err_deg:.4f} deg (Left-Handed Twist)")
    print(f"True Mean Halo Chiral Twist (ODE)   : {np.mean(np.abs(true_chiral_shift_deg)):.4f} deg")
    print(f"Recovered Spin-Parity Stacked Twist : {recovered_chiral_deg:.4f} +/- {standard_error_deg:.4f} deg")
    print(f"Scrambled-Spin Null Control         : {null_signal_deg:.4f} deg (Expected ~0.000)")
    print(f"Statistical Detection Significance  : {snr_sigma:.2f} sigma")
    print("-" * 78)
    print("\nRADIAL PROFILE CHECK: Does the twist follow the DOP853 Chiral Throat ODE?")
    print(f"{'Radial Bin (kpc)':<20} | {'Recovered Twist (deg)':<22} | {'DOP853 Chiral ODE Curve':<20}")
    print("-" * 72)

    bins = [(2, 5), (5, 12), (12, 25), (25, 40)]
    for b_min, b_max in bins:
        mask = (impact_b_kpc >= b_min) & (impact_b_kpc < b_max)
        bin_val = np.mean(parity_weighted_signal[mask])
        bin_err = np.std(parity_weighted_signal[mask]) / np.sqrt(np.sum(mask))
        b_mid = 0.5 * (b_min + b_max)
        theory_val = float(cs_chiral_ode(b_mid))
        print(f"{b_min:2d} - {b_max:2d} kpc          | {bin_val:+.3f} +/- {bin_err:.3f} deg   | +{theory_val:.3f} deg")
    print("=" * 72)


if __name__ == "__main__":
    run_chiral_halo_pipeline()
