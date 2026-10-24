"""
run_all_test_scripts.py
================================================================================
MASTER ONE-COMMAND VERIFICATION RUNNER FOR ALL PAIRED-BRANE CATENOID TEST SCRIPTS
Workspace: C:\\Users\\jason\\OneDrive\\Documents\\gravity\\2branesbetterthannobrain\\
================================================================================
"""

import os
import sys
import time
import subprocess

WORKSPACE = os.path.dirname(os.path.abspath(__file__))

CORE_TEST_SUITES = [
    ("0. Willmore-Biharmonic Catenoid Eigenvalue Proof", "catenoid_biharmonic_eigenvalue_proof.py"),
    ("1. Master 21-Sector Unified Brane Engine (76 ODEs)", "master_brane_engine.py"),
    ("2. Local Catenoid Divergence Field Solver ODE", "brane_catenoid_field_solver.py"),
    ("3. SPARC 151-Galaxy Radial Acceleration Relation ODE", "sparc_rar_catenoid_validator.py"),
    ("4. SPARC 123-Galaxy Baryonic Tully-Fisher ODE", "sparc_btfr_catenoid_validator.py"),
    ("5. HST SLACS 70-Galaxy Strong Lensing Null-Geodesic ODE", "paired_brane_lensing_validator.py"),
    ("6. Chandra 15-Galaxy-Cluster Hydrostatic ODE", "cluster_brane_tension_validator.py"),
    ("7. Bullet Cluster (1E 0657-56) 2D Cylindrical Soliton ODE", "bullet_cluster_collision_solver.py"),
    ("8. Solar System Cassini PPN, GP-B & Gaia Wide Binaries ODE", "solar_system_ppn_validator.py"),
    ("9. Atomic, Subatomic & Planck GUP Bridge ODE", "atomic_subatomic_planck_validator.py"),
    ("10. Planck 2018 CMB TT+EE Acoustic Peaks (30-State Radau ODE)", "cmb_acoustic_peak_validator.py"),
    ("11. UDG Twin Paradox (DF2/DF44), GW170817, Pulsars & EHT ODE", "udg_and_gw170817_boss_fights_validator.py"),
    ("12. Stage 3 Covariant Unification ODE", "stage3_covariant_unification_ode.py"),
    ("13. Stage 3 Matter Power Spectrum P(k), S_8 & Soliton Halo ODE", "stage3_pk_s8_and_lane_emden_soliton_ode.py"),
]


def main():
    print("=" * 96)
    print(" PAIRED-BRANE CATENOID COSMOLOGY — COMPLETE TEST SUITE RUNNER")
    print("=" * 96)
    passed = 0
    for label, script_name in CORE_TEST_SUITES:
        script_path = os.path.join(WORKSPACE, script_name)
        if not os.path.isfile(script_path):
            print(f" [SKIP] {label:<62} ({script_name} not found)")
            continue
        t0 = time.time()
        res = subprocess.run(
            [sys.executable, script_path],
            cwd=WORKSPACE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        dt = time.time() - t0
        status = "PASS" if res.returncode == 0 else f"FAIL (code {res.returncode})"
        if res.returncode == 0:
            passed += 1
        print(f" [{status}] {label:<62} | {script_name:<42} ({dt:5.2f}s)")
    print("=" * 96)
    print(f" SUMMARY: {passed}/{len(CORE_TEST_SUITES)} Verification Suites Passed.")
    print("=" * 96)


if __name__ == "__main__":
    main()
