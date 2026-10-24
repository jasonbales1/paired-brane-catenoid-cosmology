"""
real_sky_chiral_halo_crossmatch.py (Approaching vs. Receding Halo Dipole DOP853 ODE Audit)
================================================================================
LIVE OBSERVATORY CROSS-MATCH & PAIRED-BRANE HALO DIPOLE ANALYZER (DOP853 ODE)
Workspace: C:/2branesbetterthannobrain/

Why this test is critical:
  - A rotating halo / paired-brane vortex has opposite line-of-sight velocity 
    and shear on the APPROACHING side vs. the RECEDING side of the major axis.
  - Averaging all sightlines uniformly around 360 deg cancels a dipole to zero.
  - This script extracts the 'Appr' (Approaching side cardinal direction) and 
    'S/Z' columns directly from Iye et al. (2019) [J/ApJ/886/133], integrates the
    radial Chiral Catenoid Throat ODE dw/du = 1 - w^2 via `solve_ivp(method="DOP853")`,
    and projects each background VLA NVSS sightline onto the galaxy's approaching-receding axis:
        Dipole Weight = cos(PA_sightline - PA_approaching) * w_ODE(b / R_throat)
================================================================================
"""

import re
import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import CubicSpline
import astropy.units as u
from astropy.coordinates import SkyCoord
from astroquery.vizier import Vizier


def build_dimensionless_chiral_throat_ode(u_max=40.0):
    """Integrates the dimensionless radial Chiral Catenoid Throat ODE dw/du = 1 - w^2 via DOP853."""
    u_grid = np.linspace(0.0, u_max, 500)
    sol = solve_ivp(
        lambda u, y: [1.0 - (y[0] ** 2)],
        (0.0, u_max),
        [0.0],
        method="DOP853",
        t_eval=u_grid,
        rtol=1e-10,
        atol=1e-12,
    )
    return CubicSpline(sol.t, sol.y[0])


def normalize_gal_name(name_str):
    """Normalizes galaxy identifiers ('NGC 0300' -> 'NGC300', 'UGC 02953' -> 'UGC2953')."""
    s = str(name_str).strip().upper().replace(" ", "").replace("-", "")
    m = re.match(r"^([A-Z]+)0*(\d+[A-Z]*)$", s)
    if m:
        return f"{m.group(1)}{m.group(2)}"
    return s


CARDINAL_TO_PA_DEG = {
    "N": 0.0, "NNE": 22.5, "NE": 45.0, "ENE": 67.5,
    "E": 90.0, "ESE": 112.5, "SE": 135.0, "SSE": 157.5,
    "S": 180.0, "SSW": 202.5, "SW": 225.0, "WSW": 247.5,
    "W": 270.0, "WNW": 292.5, "NW": 315.0, "NNW": 337.5,
}


def parse_cardinal_pa(val_str):
    """Converts an Iye (2019) cardinal direction string (e.g. 'NW', 'SE', 'N') to PA in degrees."""
    s = str(val_str).strip().upper().replace(" ", "")
    if s in CARDINAL_TO_PA_DEG:
        return CARDINAL_TO_PA_DEG[s]
    for k in sorted(CARDINAL_TO_PA_DEG.keys(), key=len, reverse=True):
        if s == k:
            return CARDINAL_TO_PA_DEG[k]
    try:
        return float(s) % 360.0
    except Exception:
        return np.nan


def clean_col(table, candidate_names, dtype=float):
    if isinstance(candidate_names, str):
        candidate_names = [candidate_names]
    col_map = {c.lower(): c for c in table.colnames}
    chosen = None
    for cand in candidate_names:
        if cand in table.colnames:
            chosen = cand
            break
        if cand.lower() in col_map:
            chosen = col_map[cand.lower()]
            break
    if chosen is None:
        raise KeyError(f"None of {candidate_names} found in table columns: {table.colnames}")
    col = table[chosen]
    if dtype == float:
        col = col.astype(float)
        if hasattr(col, "filled"):
            col = col.filled(np.nan)
        return np.array(col, dtype=float)
    else:
        if hasattr(col, "filled"):
            col = col.filled("")
        return np.array(col, dtype=dtype)


def extract_skycoords(table):
    if "_RAJ2000" in table.colnames and "_DEJ2000" in table.colnames:
        ra_deg = clean_col(table, "_RAJ2000", float)
        dec_deg = clean_col(table, "_DEJ2000", float)
        return SkyCoord(ra=ra_deg * u.deg, dec=dec_deg * u.deg, frame="icrs"), ra_deg, dec_deg
    for ra_cand, dec_cand in [("RAJ2000", "DEJ2000"), ("_RA", "_DE"), ("RA", "DEC")]:
        if ra_cand in table.colnames and dec_cand in table.colnames:
            ra_col = table[ra_cand]
            if np.issubdtype(ra_col.dtype, np.number):
                ra_deg = clean_col(table, ra_cand, float)
                dec_deg = clean_col(table, dec_cand, float)
                return SkyCoord(ra=ra_deg * u.deg, dec=dec_deg * u.deg, frame="icrs"), ra_deg, dec_deg
            else:
                ra_str = clean_col(table, ra_cand, str)
                dec_str = clean_col(table, dec_cand, str)
                sky = SkyCoord(ra_str, dec_str, unit=(u.hourangle, u.deg), frame="icrs")
                return sky, sky.ra.deg, sky.dec.deg
    raise KeyError(f"No coordinate columns found in {table.colnames}")


def load_iye_kinematics_and_spins(gal_names_norm, gal_sky):
    """Extracts both 'S/Z' spin parity and 'Appr' (Approaching side PA) from Iye et al. (2019)."""
    print("[1b/7] Querying VizieR for Iye et al. (2019) Spin & Approaching-Side Catalogs (J/ApJ/886/133)...")
    v_iye = Vizier(columns=["*", "_RAJ2000", "_DEJ2000"], row_limit=-1)
    v_iye.timeout = 120

    spin_dict = {}
    appr_pa_dict = {}

    try:
        tables = v_iye.get_catalogs("J/ApJ/886/133")
        for tab in tables:
            if "S/Z" not in tab.colnames:
                continue
            has_appr = "Appr" in tab.colnames

            try:
                iye_sky, _, _ = extract_skycoords(tab)
                idx_s, sep2d, _ = gal_sky.match_to_catalog_sky(iye_sky)
                close_mask = sep2d.arcsec <= 45.0
                for g_i in np.where(close_mask)[0]:
                    row = tab[idx_s[g_i]]
                    gnorm = gal_names_norm[g_i]
                    sval = str(row["S/Z"]).strip().upper()
                    if sval in ("S", "Z"):
                        spin_dict[gnorm] = +1.0 if sval == "S" else -1.0
                    if has_appr:
                        pa_a = parse_cardinal_pa(row["Appr"])
                        if np.isfinite(pa_a):
                            appr_pa_dict[gnorm] = pa_a
            except Exception:
                pass

        print(f"       -> Matched {len(spin_dict)} SPARC galaxies with true S/Z spins and {len(appr_pa_dict)} with Approaching-Side PAs.")
    except Exception as exc:
        print(f"       [!] Notice querying J/ApJ/886/133: {exc}")

    return spin_dict, appr_pa_dict


def run_dipole_permutation_test(label, obs_vec, err_vec, g_idx, template_vec, gal_names, seed=42):
    """Runs cluster-balanced weighted estimation and 1000-trial galaxy-level sign permutation."""
    valid = np.isfinite(obs_vec) & np.isfinite(err_vec) & np.isfinite(template_vec) & (np.abs(template_vec) > 0.05)
    if np.sum(valid) < 4:
        print(f"\n>>> {label}: Insufficient sightlines ({np.sum(valid)}).")
        return

    v_g = g_idx[valid]
    v_obs = obs_vec[valid]
    v_err = err_vec[valid]
    v_temp = template_vec[valid]

    gal_counts = np.array([np.sum(v_g == gi) for gi in v_g], dtype=float)
    w = (1.0 / (v_err ** 2)) / gal_counts

    denom = np.sum(w * (v_temp ** 2))
    amp = np.sum(w * v_temp * v_obs) / denom
    amp_err = np.sqrt(np.sum((w ** 2) * (v_temp ** 2) * (v_err ** 2))) / denom
    snr_meas = amp / amp_err

    rng = np.random.default_rng(seed)
    u_gals = np.unique(v_g)
    null_amps = np.zeros(1000)
    for t in range(1000):
        flips = {ug: rng.choice([-1.0, 1.0]) for ug in u_gals}
        f_vec = np.array([flips[gi] for gi in v_g])
        t_temp = f_vec * v_temp
        null_amps[t] = np.sum(w * t_temp * v_obs) / np.sum(w * (t_temp ** 2))

    p_val = np.mean(np.abs(null_amps) >= np.abs(amp))
    z_cluster = (amp - np.mean(null_amps)) / np.std(null_amps, ddof=1)

    print(f"\n>>> {label}")
    print(f"    Sightlines: {len(v_obs)} across {len(u_gals)} unique foreground galaxies")
    print(f"    Cluster-Balanced Amplitude     : {amp:+.4f} +/- {amp_err:.4f} ({snr_meas:+.2f} sigma_meas)")
    print(f"    1000-Trial Scrambled Null Mean : {np.mean(null_amps):+.4f} +/- {np.std(null_amps, ddof=1):.4f}")
    print(f"    Cluster-Robust Z-Score & p-val : Z = {z_cluster:+.2f} sigma_cluster | p = {p_val:.4f}")


def run_real_sky_chiral_crossmatch(max_halo_kpc=60.0, seed=42):
    c_m_s = 2.99792458e8
    H0_s = (67.4 * 1e3) / 3.08567758e22
    R_cosmos = c_m_s / H0_s
    a_brane = (c_m_s ** 2) / (6.0 * R_cosmos)

    print("=" * 92)
    print(" 2BRANESBETTERTHANNOBRAIN — APPROACHING vs. RECEDING HALO DIPOLE AUDIT (DOP853 ODE)")
    print("=" * 92)

    v = Vizier(columns=["*", "_RAJ2000", "_DEJ2000"], row_limit=-1)
    v.timeout = 120

    print("[1/7] Querying VizieR for SPARC Spiral Galaxy Catalog (J/AJ/152/157/table1)...")
    sparc_cat = v.get_catalogs("J/AJ/152/157/table1")[0]
    _, ra_sparc, dec_sparc = extract_skycoords(sparc_cat)
    dist_mpc = clean_col(sparc_cat, ["Dist", "D"], float)
    vflat_kms = clean_col(sparc_cat, ["Vflat", "V_flat"], float)
    inc_raw = clean_col(sparc_cat, ["Inc", "i"], float)
    names_raw = [str(n).strip().replace(" ", "") for n in clean_col(sparc_cat, ["Name", "Galaxy"], str)]

    valid_gals = (
        np.isfinite(ra_sparc) & np.isfinite(dec_sparc) & np.isfinite(dist_mpc)
        & (dist_mpc > 0.5) & np.isfinite(vflat_kms) & (vflat_kms > 25.0)
    )
    ra_sparc, dec_sparc, dist_mpc, vflat_kms = (
        ra_sparc[valid_gals], dec_sparc[valid_gals], dist_mpc[valid_gals], vflat_kms[valid_gals]
    )
    inc_deg = np.nan_to_num(inc_raw[valid_gals], nan=60.0)
    gal_names = np.array(names_raw)[valid_gals]
    gal_names_norm = np.array([normalize_gal_name(n) for n in gal_names])

    gal_sky = SkyCoord(ra=ra_sparc * u.deg, dec=dec_sparc * u.deg, frame="icrs")
    spin_dict, appr_pa_dict = load_iye_kinematics_and_spins(gal_names_norm, gal_sky)

    kpc_to_m = 3.08567758e19
    r_throat_kpc = ((vflat_kms * 1e3) ** 2) / a_brane / kpc_to_m * 0.15

    print("[2/7] Querying VizieR for VLA NVSS Polarized Radio Catalog (J/ApJ/702/1230)...")
    nvss_cat = v.get_catalogs("J/ApJ/702/1230/catalog")[0]
    _, ra_nvss, dec_nvss = extract_skycoords(nvss_cat)
    rm_val = clean_col(nvss_cat, ["RM"], float)
    rm_err = clean_col(nvss_cat, ["e_RM"], float)
    valid_nvss = np.isfinite(ra_nvss) & np.isfinite(dec_nvss) & np.isfinite(rm_val)
    ra_nvss, dec_nvss, rm_val = ra_nvss[valid_nvss], dec_nvss[valid_nvss], rm_val[valid_nvss]
    rm_err = np.where(np.isfinite(rm_err[valid_nvss]) & (rm_err[valid_nvss] > 0), rm_err[valid_nvss], 8.0)
    nvss_sky = SkyCoord(ra=ra_nvss * u.deg, dec=dec_nvss * u.deg, frame="icrs")

    print(f"[3/7] Cross-matching background radio sightlines within {max_halo_kpc:.0f} kpc of halos...")
    idx_gal, idx_nvss, sep2d, _ = nvss_sky.search_around_sky(gal_sky, 3.5 * u.deg)
    b_kpc = dist_mpc[idx_gal] * 1000.0 * np.tan(sep2d.to(u.rad).value)
    halo_mask = (b_kpc >= 1.5) & (b_kpc <= max_halo_kpc)
    g_idx, n_idx, b_hit_kpc = idx_gal[halo_mask], idx_nvss[halo_mask], b_kpc[halo_mask]

    gal_matched_sky = gal_sky[g_idx]
    nvss_matched_sky = nvss_sky[n_idx]
    los_pa_deg = gal_matched_sky.position_angle(nvss_matched_sky).to(u.deg).value

    rm_residual = np.zeros(len(b_hit_kpc))
    spin_signs = np.full(len(b_hit_kpc), np.nan)
    appr_pa = np.full(len(b_hit_kpc), np.nan)
    tanh_weights = np.zeros(len(b_hit_kpc))
    sin_inc = np.sin(np.radians(np.clip(inc_deg[g_idx], 10.0, 88.0)))

    cs_throat_ode = build_dimensionless_chiral_throat_ode(u_max=40.0)

    for k in range(len(b_hit_kpc)):
        gi, ni = g_idx[k], n_idx[k]
        gnorm = gal_names_norm[gi]
        bg_mask = (idx_gal == gi) & (b_kpc > max_halo_kpc * 1.5) & (sep2d.deg <= 3.5)
        mw_foreground_rm = np.median(rm_val[idx_nvss[bg_mask]]) if np.sum(bg_mask) >= 3 else 0.0
        rm_residual[k] = rm_val[ni] - mw_foreground_rm

        if gnorm in spin_dict:
            spin_signs[k] = spin_dict[gnorm]
        if gnorm in appr_pa_dict:
            appr_pa[k] = appr_pa_dict[gnorm]
        u_b = min(b_hit_kpc[k] / max(r_throat_kpc[gi], 2.0), 40.0)
        tanh_weights[k] = float(cs_throat_ode(u_b))

    dipole_cos_phi = np.cos(np.radians(los_pa_deg - appr_pa))

    lambda_nvss_sq = (c_m_s / 1.4e9) ** 2
    v_nvss_master = Vizier(columns=["**", "_RAJ2000", "_DEJ2000"], row_limit=5)
    v_nvss_master.timeout = 120
    jet_pa_deg, jet_pa_err, pol_pa_deg, pol_pa_err = (
        np.full(len(n_idx), np.nan), np.full(len(n_idx), 90.0),
        np.full(len(n_idx), np.nan), np.full(len(n_idx), 5.0)
    )

    try:
        master_res = v_nvss_master.query_region(nvss_matched_sky, radius=15.0 * u.arcsec, catalog="VIII/65/nvss")
        if len(master_res) > 0:
            m_tab = master_res[0]
            m_sky, _, _ = extract_skycoords(m_tab)
            m_pa = clean_col(m_tab, ["PA"], float)
            m_epa = clean_col(m_tab, ["e_PA"], float)
            m_polpa = clean_col(m_tab, ["polPA", "PolPA", "PolAng"], float)
            m_epolpa = clean_col(m_tab, ["e_polPA", "e_PolPA", "e_PolAng"], float)
            idx_m, sep_m, _ = nvss_matched_sky.match_to_catalog_sky(m_sky)
            good_m = sep_m.arcsec <= 15.0
            jet_pa_deg[good_m] = m_pa[idx_m[good_m]]
            jet_pa_err[good_m] = np.where(np.isfinite(m_epa[idx_m[good_m]]) & (m_epa[idx_m[good_m]] > 0), m_epa[idx_m[good_m]], 45.0)
            pol_pa_deg[good_m] = m_polpa[idx_m[good_m]]
            pol_pa_err[good_m] = np.where(np.isfinite(m_epolpa[idx_m[good_m]]) & (m_epolpa[idx_m[good_m]] > 0), m_epolpa[idx_m[good_m]], 5.0)
    except Exception as exc:
        print(f"      [!] VizieR VIII/65 batch query note: {exc}")

    theta_0_deg = pol_pa_deg - np.degrees(rm_val[n_idx] * lambda_nvss_sq)
    raw_diff_rad = np.radians(theta_0_deg - jet_pa_deg)
    biref_offset_deg = np.degrees(0.25 * np.arctan2(np.sin(4.0 * raw_diff_rad), np.cos(4.0 * raw_diff_rad)))
    biref_err_deg = np.sqrt(pol_pa_err ** 2 + (0.25 * jet_pa_err) ** 2 + 6.0 ** 2)

    print("\n" + "=" * 92)
    print(" [4/7] APPROACHING vs. RECEDING MAJOR-AXIS DIPOLE SHEAR TESTS (IYE 2019 GOLD SAMPLE, DOP853 ODE)")
    print("=" * 92)

    clean_rm = np.abs(rm_residual) < 80.0
    strict_jet = clean_rm & (jet_pa_err <= 25.0)

    run_dipole_permutation_test(
        "TEST 1: FARADAY RM KINEMATIC DIPOLE [dRM ~ sin(i) * cos(PA_los - PA_appr) * w_ODE(b/R_c)] (rad/m^2)",
        np.where(clean_rm, rm_residual, np.nan),
        np.sqrt(rm_err[n_idx] ** 2 + 12.0 ** 2),
        g_idx,
        sin_inc * dipole_cos_phi * tanh_weights,
        gal_names,
        seed=seed,
    )

    run_dipole_permutation_test(
        "TEST 2: JET BIREFRINGENCE KINEMATIC DIPOLE [Delta_theta_0 ~ sin(i) * cos(PA_los - PA_appr)] (deg)",
        np.where(strict_jet, biref_offset_deg, np.nan),
        biref_err_deg,
        g_idx,
        sin_inc * dipole_cos_phi * tanh_weights,
        gal_names,
        seed=seed,
    )

    run_dipole_permutation_test(
        "TEST 3: CHIRAL-COUPLED DIPOLE [Delta_theta_0 ~ Spin(S/Z) * sin(i) * cos(PA_los - PA_appr)] (deg)",
        np.where(strict_jet, biref_offset_deg, np.nan),
        biref_err_deg,
        g_idx,
        spin_signs * sin_inc * dipole_cos_phi * tanh_weights,
        gal_names,
        seed=seed,
    )

    print("=" * 92)


if __name__ == "__main__":
    run_real_sky_chiral_crossmatch()
