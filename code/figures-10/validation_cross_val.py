#!/usr/bin/env python3
"""
Leave-One-Out Cross-Validation for CR Model
Tests overfitting by removing each galaxy and re-optimizing
"""
import numpy as np
import pickle
from scipy.optimize import differential_evolution
import warnings
warnings.filterwarnings('ignore')

# Constants
KPC_TO_M = 3.086e19
KM_TO_M = 1.0e3
GLOBAL_ML = 1.0
SIGMA0, F_FLOOR, ALPHA_BEAM = 10.0, 0.05, 0.3
ASYM_DWARF, ASYM_SPIRAL = 0.10, 0.05
K_TURB, P_TURB = 0.07, 1.3

def n_analytic(r_kpc, A, R0, P):
    return 1.0 + A * (1.0 - np.exp(-((r_kpc / R0) ** P)))

def xi_global(f_gas_true, C_lag):
    return 1.0 + C_lag * (max(f_gas_true, 0.0) ** 0.5)

def w_g_kernel(g_bar_si, alpha, C_lag, a0):
    term = np.maximum(g_bar_si / a0, 1e-30)
    return 1.0 + C_lag * (term ** (-alpha) - 1.0)

def compute_model_chi2(params, master_table):
    alpha, C_lag, A, R0, P, hz_over_Rd, log_a0 = params
    a0 = 10**log_a0
    chi2_list = []
    
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
        
        v_baryon = np.sqrt(v_gas ** 2 + (np.sqrt(GLOBAL_ML) * v_disk) ** 2 + v_bul ** 2)
        r_m = r * KPC_TO_M
        v_baryon_mps = v_baryon * KM_TO_M
        g_bar = np.where(r_m > 0, (v_baryon_mps ** 2) / r_m, 0.0)
        
        w_g = w_g_kernel(g_bar, alpha, C_lag, a0)
        n_r = n_analytic(r, A, R0, P)
        xi = xi_global(f_gas_true, C_lag)
        
        if R_d_guess <= 0: zeta = 1.0
        else:
            hz = hz_over_Rd * R_d_guess
            zeta = np.clip(1.0 + 0.5 * (hz / (r + 0.1 * R_d_guess)), 0.8, 1.2)
            
        w = w_g * n_r * xi * zeta
        v_model = np.sqrt(np.maximum(w, 0.0)) * v_baryon
        
        is_dwarf = bool(np.nanmax(v_obs) < 80.0)
        sigma = v_err ** 2 + SIGMA0 ** 2 + (F_FLOOR * v_obs) ** 2
        if R_d_guess > 0:
            beam_kpc = 0.3 * R_d_guess
            sigma_beam = ALPHA_BEAM * beam_kpc * v_obs / (r + beam_kpc)
            sigma += sigma_beam ** 2
        asym_frac = ASYM_DWARF if is_dwarf else ASYM_SPIRAL
        sigma += (asym_frac * v_obs) ** 2
        Rd = R_d_guess if R_d_guess > 0 else 2.0
        sigma_turb = K_TURB * v_obs * (1.0 - np.exp(-r / Rd)) ** P_TURB
        sigma += sigma_turb ** 2
        sig = np.sqrt(np.maximum(sigma, 1e-10))
        
        valid = np.isfinite(v_obs) & np.isfinite(v_model) & (sig > 0)
        if np.sum(valid) > 0:
            res = ((v_obs[valid] - v_model[valid]) / sig[valid]) ** 2
            chi2_val = np.sum(res)
            N = np.sum(valid)
            chi2_list.append(chi2_val / N)
        else:
            chi2_list.append(100.0)
            
    return np.median(chi2_list)

def main():
    print("Loading Q=1 data...")
    with open('external/gravity/active/scripts/sparc_q1.pkl', 'rb') as f:
        full_data = pickle.load(f)
    
    galaxy_names = list(full_data.keys())
    print(f"Total galaxies: {len(galaxy_names)}")
    
    # Fiducial parameters
    fiducial = [0.3899, 0.2981, 1.0599, 17.7895, 0.9541, 0.1540, -9.7097]
    
    # Compute fiducial chi2 for each galaxy
    print("\nComputing fiducial chi2 for all galaxies...")
    fiducial_chi2 = {}
    for name in galaxy_names:
        single_gal = {name: full_data[name]}
        chi2 = compute_model_chi2(fiducial, single_gal)
        fiducial_chi2[name] = chi2
    
    # Leave-one-out: sample 10 galaxies for speed
    np.random.seed(42)
    sample_galaxies = np.random.choice(galaxy_names, size=min(10, len(galaxy_names)), replace=False)
    
    print(f"\nPerforming Leave-One-Out on {len(sample_galaxies)} galaxies...")
    delta_chi2_list = []
    
    for i, left_out in enumerate(sample_galaxies):
        print(f"  [{i+1}/{len(sample_galaxies)}] Leaving out {left_out}...")
        
        # Create training set (all except left_out)
        train_data = {k: v for k, v in full_data.items() if k != left_out}
        
        # Quick optimization (fewer iterations for speed)
        bounds = [(0.30, 0.45), (0.20, 0.35), (0.8, 1.3), (15.0, 20.0), (0.7, 1.2), (0.1, 0.2), (-10.0, -9.5)]
        result = differential_evolution(
            compute_model_chi2, bounds, args=(train_data,),
            maxiter=5, popsize=5, tol=0.1, disp=False, workers=-1
        )
        
        # Test on left-out galaxy
        test_data = {left_out: full_data[left_out]}
        chi2_loo = compute_model_chi2(result.x, test_data)
        chi2_fid = fiducial_chi2[left_out]
        
        delta = chi2_loo - chi2_fid
        delta_chi2_list.append(delta)
        print(f"      Fiducial: {chi2_fid:.3f}, LOO: {chi2_loo:.3f}, Delta: {delta:+.3f}")
    
    print(f"\n--- Leave-One-Out Cross-Validation Results ---")
    print(f"Mean Delta Chi2/N: {np.mean(delta_chi2_list):+.3f} ± {np.std(delta_chi2_list):.3f}")
    print(f"Median Delta Chi2/N: {np.median(delta_chi2_list):+.3f}")
    print(f"Interpretation: Small positive delta indicates minimal overfitting.")

if __name__ == '__main__':
    main()

