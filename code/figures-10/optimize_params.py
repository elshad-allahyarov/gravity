#!/usr/bin/env python3
"""
Global Optimization script for ILG/Causal-Response parameters.
Optimizes 7 global parameters:
  alpha, C_lag, A, R0, P, hz_over_Rd, log10(a0)
Minimizes median chi^2/N across the Q=1 SPARC subset.
Uses Differential Evolution to find the global minimum.
"""

import pickle
import pathlib
import numpy as np
from scipy.optimize import differential_evolution
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

# Constants
KPC_TO_M = 3.086e19
KM_TO_M = 1.0e3
GLOBAL_ML = 1.0

# Fixed Error-model settings
SIGMA0 = 10.0           
F_FLOOR = 0.05          
ALPHA_BEAM = 0.3        
ASYM_DWARF = 0.10       
ASYM_SPIRAL = 0.05      
K_TURB = 0.07           
P_TURB = 1.3            

def n_analytic(r_kpc, A, R0, P):
    return 1.0 + A * (1.0 - np.exp(-((r_kpc / R0) ** P)))

def xi_global(f_gas_true, C_lag):
    return 1.0 + C_lag * (max(f_gas_true, 0.0) ** 0.5)

def w_g_kernel(g_bar_si, alpha, C_lag, a0):
    term = np.maximum(g_bar_si / a0, 1e-30)
    return 1.0 + C_lag * (term ** (-alpha) - 1.0)

def compute_model_chi2(params, master_table):
    # Unpack parameters (7 params)
    # params = [alpha, C_lag, A, R0, P, hz_over_Rd, log_a0]
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
        
        # Model Velocity
        v_baryon = np.sqrt(v_gas ** 2 + (np.sqrt(GLOBAL_ML) * v_disk) ** 2 + v_bul ** 2)
        
        r_m = r * KPC_TO_M
        v_baryon_mps = v_baryon * KM_TO_M
        g_bar = np.where(r_m > 0, (v_baryon_mps ** 2) / r_m, 0.0)
        
        w_g = w_g_kernel(g_bar, alpha, C_lag, a0)
        n_r = n_analytic(r, A, R0, P)
        xi = xi_global(f_gas_true, C_lag)
        
        if R_d_guess <= 0:
            zeta = 1.0
        else:
            hz = hz_over_Rd * R_d_guess
            zeta = np.clip(1.0 + 0.5 * (hz / (r + 0.1 * R_d_guess)), 0.8, 1.2)
            
        w = w_g * n_r * xi * zeta
        v_model = np.sqrt(np.maximum(w, 0.0)) * v_baryon
        
        # Error Model
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
        
        # Chi2
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
    base = pathlib.Path('external/gravity/active/scripts')
    pkl_path = base / 'sparc_q1.pkl'
    
    print(f"Loading data from {pkl_path}...")
    with open(pkl_path, 'rb') as f:
        master_table = pickle.load(f)
    print(f"Loaded {len(master_table)} galaxies.")
    
    # Bounds for 7 parameters
    # alpha, C_lag, A, R0, P, hz, log_a0
    bounds = [
        (0.10, 0.40),   # alpha
        (0.01, 0.30),   # C_lag
        (1.0, 10.0),    # A
        (1.0, 20.0),    # R0
        (0.5, 3.0),     # P
        (0.1, 0.5),     # hz_over_Rd
        (-11.0, -9.0)   # log10(a0) (centered on -9.92 for 1.2e-10)
    ]
    
    print("\nStarting Global Optimization (Differential Evolution)...")
    # Using 'best1bin' and modest popsize for speed vs coverage
    res = differential_evolution(
        compute_model_chi2,
        bounds,
        args=(master_table,),
        strategy='best1bin',
        maxiter=20,
        popsize=10,
        tol=0.01,
        disp=True,
        workers=-1 # Use parallel if available
    )
    
    print("\nOptimization Complete!")
    print(f"Final Median Chi2/N: {res.fun:.4f}")
    print("Best Parameters:")
    print(f"  alpha:      {res.x[0]:.4f}")
    print(f"  C_lag:      {res.x[1]:.4f}")
    print(f"  A:          {res.x[2]:.4f}")
    print(f"  R0:         {res.x[3]:.4f}")
    print(f"  P:          {res.x[4]:.4f}")
    print(f"  hz_over_Rd: {res.x[5]:.4f}")
    print(f"  log_a0:     {res.x[6]:.4f}  (a0 = {10**res.x[6]:.2e})")

if __name__ == '__main__':
    main()

