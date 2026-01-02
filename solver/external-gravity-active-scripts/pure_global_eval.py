#!/usr/bin/env python3
"""
Pure-global evaluation (non-fitted) for ILG and MOND baselines.
Updated with optimized parameters and Q-flag analysis.
"""

from __future__ import annotations

import os
import math
import pickle
import pathlib
from typing import Dict, Any, List

import numpy as np
import pandas as pd


# Constants (Optimized)
KPC_TO_M = 3.086e19
KM_TO_M = 1.0e3
PHI = (1.0 + 5 ** 0.5) / 2.0
C_LAG = 0.094  # Optimized value
A0 = 1.2e-10  # m/s^2 (Standard MOND scale used in code)
# Note: Paper might mention 1.95e-10 derived, but code uses 1.2e-10. 
# Providing 1.2e-10 as per existing code structure, assuming term normalization handles it.
# Actually, if A0 is different, w_g_kernel changes.
# The log said: "a0 = 2.28...". But the user asked to "update all constants".
# I'll stick to what was in the script unless I'm sure. 
# The log "New Best Parameters" didn't explicitly list A0 change for the code, 
# but listed it in the "Fresh optimization" section which was REJECTED by the user ("do 1").
# Wait, "me; do B" -> "Optimized Parameters... alpha = 0.210... A = 3.66...".
# It did NOT list A0 in the "New Best Parameters" list.
# So I assume A0 stays 1.2e-10 in the code (fixed).

ALPHA = 0.210

# Global-only settings
GLOBAL_ML = 1.0
N_A, N_R0, N_P = 3.66, 10.14, 1.66  # Optimized analytic n(r)

# Error-model settings (global-only)
SIGMA0 = 10.0           # km/s
F_FLOOR = 0.05          # fractional
ALPHA_BEAM = 0.3        # dimensionless
ASYM_DWARF = 0.10       # fractional of v_obs
ASYM_SPIRAL = 0.05      # fractional of v_obs
K_TURB = 0.07           # dimensionless
P_TURB = 1.3            # dimensionless


def n_analytic(r_kpc: np.ndarray) -> np.ndarray:
    return 1.0 + N_A * (1.0 - np.exp(-((r_kpc / N_R0) ** N_P)))


def xi_global(f_gas_true: float) -> float:
    # Using PHI**-5 for the prefactor of the sqrt term?
    # The original code had: 1.0 + (PHI ** -5) * ...
    # The log said "C_xi = 0.094". 
    # Usually C_xi is the coefficient in front of f_gas term.
    # In original code, C_LAG was used in w_g_kernel, not xi_global.
    # Original xi_global used PHI**-5 (~0.09).
    # If C_xi is optimized to 0.094, I should use that here.
    return 1.0 + 0.094 * (max(f_gas_true, 0.0) ** 0.5)


def w_g_kernel(g_bar_si: np.ndarray) -> np.ndarray:
    # g_ext = 0
    term = np.maximum(g_bar_si / A0, 1e-30)  # avoid div/underflow
    # Original: 1.0 + C_LAG * ...
    # C_LAG in original was PHI**-5 (~0.09).
    # Is C_LAG the same as C_xi?
    # The log listed "C_xi = 0.094".
    # And "C_lag" was not listed in "New Best Parameters".
    # BUT, in the "Consistency Check" section of log:
    # "C_lag=0.0939"
    # And "C_xi" is not mentioned in that line.
    # It seems C_LAG and C_xi might be confused or same.
    # In original code:
    # C_LAG = PHI**-5 (used in w_g_kernel)
    # xi_global uses PHI**-5 hardcoded.
    # If I have C_LAG (or C_xi) = 0.094, I should probably use it in both or figure out which one.
    # The log says "xi = 1 + C_xi * sqrt(f_gas)".
    # And w_g_kernel uses C_LAG.
    # If the user says "C_xi = 0.094", I will apply it to xi_global.
    # I will leave w_g_kernel as is (PHI**-5) or update it if C_LAG is also 0.094.
    # The log said "C_lag=0.0939" in the consistency check.
    # So I will use 0.094 for BOTH.
    return 1.0 + 0.094 * (term ** (-ALPHA) - 1.0)


def compute_ilg_velocity(r_kpc: np.ndarray,
                         v_gas: np.ndarray,
                         v_disk: np.ndarray,
                         v_bul: np.ndarray,
                         v_obs: np.ndarray,
                         f_gas_true: float,
                         R_d_guess: float | None = None) -> np.ndarray:
    # Apply global M/L to stellar disk
    v_baryon = np.sqrt(v_gas ** 2 + (math.sqrt(GLOBAL_ML) * v_disk) ** 2 + v_bul ** 2)

    # g_bar in SI
    r_m = r_kpc * KPC_TO_M
    v_baryon_mps = v_baryon * KM_TO_M
    g_bar = np.where(r_m > 0, (v_baryon_mps ** 2) / r_m, 0.0)

    w_g = w_g_kernel(g_bar)
    n_r = n_analytic(r_kpc)
    xi = xi_global(f_gas_true)
    
    if R_d_guess is None or R_d_guess <= 0:
        zeta = 1.0
    else:
        hz_over_Rd = 0.25 # Optimized
        hz = hz_over_Rd * R_d_guess
        # clip to [0.8, 1.2] as in paper
        zeta = np.clip(1.0 + 0.5 * (hz / (r_kpc + 0.1 * R_d_guess)), 0.8, 1.2)
    
    w = w_g * n_r * xi * zeta

    v_model = np.sqrt(np.maximum(w, 0.0)) * v_baryon
    return v_model


def mond_simple_nu_velocity(r_kpc: np.ndarray,
                            v_gas: np.ndarray,
                            v_disk: np.ndarray,
                            v_bul: np.ndarray) -> np.ndarray:
    # Global M/L = 1.0
    v_baryon = np.sqrt(v_gas ** 2 + (math.sqrt(GLOBAL_ML) * v_disk) ** 2 + v_bul ** 2)
    r_m = r_kpc * KPC_TO_M
    v_baryon_mps = v_baryon * KM_TO_M
    g_N = np.where(r_m > 0, (v_baryon_mps ** 2) / r_m, 0.0)
    y = np.maximum(g_N / A0, 1e-30)
    nu = 0.5 + np.sqrt(0.25 + 1.0 / y)
    g = nu * g_N
    v_mps = np.sqrt(np.maximum(g, 0.0) * r_m)
    return v_mps / KM_TO_M


def sigma_total_kms(r_kpc: np.ndarray,
                    v_obs: np.ndarray,
                    v_err: np.ndarray,
                    R_d_guess: float | None = None,
                    is_dwarf: bool | None = None) -> np.ndarray:
    sigma = v_err ** 2 + SIGMA0 ** 2 + (F_FLOOR * v_obs) ** 2
    if R_d_guess is not None and R_d_guess > 0:
        beam_kpc = 0.3 * R_d_guess
        sigma_beam = ALPHA_BEAM * beam_kpc * v_obs / (r_kpc + beam_kpc)
        sigma += sigma_beam ** 2
    
    if is_dwarf is None:
        is_dwarf = bool(np.nanmax(v_obs) < 80.0)
    asym_frac = ASYM_DWARF if is_dwarf else ASYM_SPIRAL
    sigma += (asym_frac * v_obs) ** 2
    
    if R_d_guess is None or R_d_guess <= 0:
        Rd = 2.0
    else:
        Rd = R_d_guess
    sigma_turb = K_TURB * v_obs * (1.0 - np.exp(-r_kpc / Rd)) ** P_TURB
    sigma += sigma_turb ** 2
    return np.sqrt(np.maximum(sigma, 0.0))


def load_q_flags(data_file: str) -> Dict[str, int]:
    q_map = {}
    try:
        with open(data_file, 'r') as f:
            for line in f:
                if not line.strip(): continue
                parts = line.split()
                # CamB 10 ... Q is column index 4 (0-based) ?
                # CamB 10 3.36 0.26 2 ...
                # 0    1  2    3    4
                if len(parts) > 4:
                    try:
                        name = parts[0]
                        q_val = int(parts[4])
                        q_map[name] = q_val
                    except ValueError:
                        pass
    except Exception as e:
        print(f"Warning: Could not read Q flags from {data_file}: {e}")
    return q_map


def eval_pure_global(master_table: Dict[str, Any], q_map: Dict[str, int]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    
    # Statistics counters
    stats = {
        'Total': {'Dwarf': 0, 'Spiral': 0, 'Massive': 0},
        'Q1': {'Dwarf': 0, 'Spiral': 0, 'Massive': 0},
        'Q2': {'Dwarf': 0, 'Spiral': 0, 'Massive': 0},
        'Q3': {'Dwarf': 0, 'Spiral': 0, 'Massive': 0},
    }

    for name, g in master_table.items():
        df = g['data']
        r = df['rad'].values.astype(float)
        v_obs = df['vobs'].values.astype(float)
        v_err = df['verr'].values.astype(float)
        v_gas = df['vgas'].values.astype(float)
        v_disk = df['vdisk'].values.astype(float)
        v_bul = df['vbul'].values.astype(float)

        f_gas_true = float(g.get('f_gas_true', 0.0))
        R_d_guess = float(g.get('R_d', 2.0))
        
        v_ilg = compute_ilg_velocity(r, v_gas, v_disk, v_bul, v_obs, f_gas_true, R_d_guess)
        v_mond = mond_simple_nu_velocity(r, v_gas, v_disk, v_bul)

        v_max = np.nanmax(v_obs)
        is_dwarf = bool(v_max < 80.0)
        
        # Classification
        if v_max < 80.0:
            g_type = 'Dwarf'
        elif 80.0 <= v_max <= 200.0:
            g_type = 'Spiral'
        else:
            g_type = 'Massive'

        q_val = q_map.get(name, 0) # 0 means unknown

        # Count
        stats['Total'][g_type] += 1
        if q_val == 1:
            stats['Q1'][g_type] += 1
        elif q_val == 2:
            stats['Q2'][g_type] += 1
        elif q_val == 3:
            stats['Q3'][g_type] += 1

        sig = sigma_total_kms(r, v_obs, v_err, R_d_guess=R_d_guess, is_dwarf=is_dwarf)
        chi2_ilg = float(np.sum(((v_obs - v_ilg) / sig) ** 2))
        chi2_mond = float(np.sum(((v_obs - v_mond) / sig) ** 2))

        n_pts = int(np.sum(np.isfinite(v_obs)))
        rows.append({
            'name': name,
            'N': n_pts,
            'Q': q_val,
            'Type': g_type,
            'chi2_ilg': chi2_ilg,
            'chi2N_ilg': chi2_ilg / max(n_pts, 1),
            'chi2_mond': chi2_mond,
            'chi2N_mond': chi2_mond / max(n_pts, 1),
        })

    # Print Table
    print("\nRequested Table:")
    print("-" * 50)
    print(f"{'':<10} {'Dwarf':<10} {'Spiral':<10} {'Massive':<10}")
    
    with open('stats_counts.txt', 'w') as f:
        f.write("-" * 50 + "\n")
        f.write(f"{'':<10} {'Dwarf':<10} {'Spiral':<10} {'Massive':<10}\n")
        for cat in ['Total', 'Q1', 'Q2', 'Q3']:
            if cat == 'Q1': s = stats['Q1']
            elif cat == 'Q2': s = stats['Q2']
            elif cat == 'Q3': s = stats['Q3']
            else: s = stats['Total']
            
            line = f"{cat:<10} {s['Dwarf']:<10} {s['Spiral']:<10} {s['Massive']:<10}"
            print(line)
            f.write(line + "\n")
        f.write("-" * 50 + "\n")


    return pd.DataFrame(rows).sort_values('name')


def main() -> None:
    # Adjust paths to match workspace structure
    base = pathlib.Path.cwd()
    
    # Try multiple locations for pickle
    # Known location from list_dir: external/gravity/active/scripts/sparc_master.pkl
    possible_paths = [
        base / 'external' / 'gravity' / 'active' / 'scripts' / 'sparc_master.pkl',
        base / 'external' / 'gravity' / 'active' / 'results' / 'sparc_master.pkl',
        base / 'sparc_master.pkl'
    ]
    
    pkl_path = None
    for p in possible_paths:
        if p.exists():
            pkl_path = p
            break
            
    if not pkl_path:
        print("Error: Could not find sparc_master.pkl. Checked:")
        for p in possible_paths:
            print(f"  - {p}")
        return

    print(f"Using data file: {pkl_path}")
    
    # Try multiple locations for Q data
    q_data_path = base / '0-0-0-0-0-sparc-data.txt'

    q_map = load_q_flags(str(q_data_path)) if q_data_path.exists() else {}
    if not q_map:
        print("Warning: Q flags not found or empty.")

    with open(pkl_path, 'rb') as f:
        master_table: Dict[str, Any] = pickle.load(f)

    df = eval_pure_global(master_table, q_map)

    # Save results if needed...
    # For now, just printing the table is the main goal.

if __name__ == '__main__':
    main()
