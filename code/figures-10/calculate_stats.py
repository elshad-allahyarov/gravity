import numpy as np
import pickle
from scipy.stats import wilcoxon
import math

# Optimized Parameters (from optimize_params.py results)
ALPHA = 0.3899
C_LAG = 0.2981
N_A, N_R0, N_P = 1.0599, 17.7895, 0.9541
HZ_OVER_RD = 0.1540
A0_OPT = 1.95e-10
A0_MOND = 1.2e-10

# Constants
KPC_TO_M = 3.086e19
KM_TO_M = 1.0e3
GLOBAL_ML = 1.0
GYR_TO_S = 3.154e16

# Fixed Error-model settings (MUST MATCH optimize_params.py)
SIGMA0 = 10.0           
F_FLOOR = 0.05          
ALPHA_BEAM = 0.3        
ASYM_DWARF = 0.10       
ASYM_SPIRAL = 0.05      
K_TURB = 0.07           
P_TURB = 1.3            

def n_analytic(r_kpc): return 1.0 + N_A * (1.0 - np.exp(-((r_kpc / N_R0) ** N_P)))
def xi_global(f_gas_true): return 1.0 + C_LAG * (max(f_gas_true, 0.0) ** 0.5)
def w_g_kernel(g_bar_si):
    term = np.maximum(g_bar_si / A0_OPT, 1e-30)
    return 1.0 + C_LAG * (term ** (-ALPHA) - 1.0)

def get_sigma_total(r_kpc, v_obs, v_err, R_d_guess, is_dwarf):
    """Full error model matching optimize_params.py"""
    sigma = v_err ** 2 + SIGMA0 ** 2 + (F_FLOOR * v_obs) ** 2
    if R_d_guess > 0:
        beam_kpc = 0.3 * R_d_guess
        sigma_beam = ALPHA_BEAM * beam_kpc * v_obs / (r_kpc + beam_kpc)
        sigma += sigma_beam ** 2
    asym_frac = ASYM_DWARF if is_dwarf else ASYM_SPIRAL
    sigma += (asym_frac * v_obs) ** 2
    Rd = R_d_guess if R_d_guess > 0 else 2.0
    sigma_turb = K_TURB * v_obs * (1.0 - np.exp(-r_kpc / Rd)) ** P_TURB
    sigma += sigma_turb ** 2
    return np.sqrt(np.maximum(sigma, 1e-10))

def compute_chi2(g):
    df = g['data']
    r = df['rad'].values
    v_obs = df['vobs'].values
    v_err = df['verr'].values
    v_gas = df['vgas'].values
    v_disk = df['vdisk'].values
    v_bul = df['vbul'].values
    
    # Baryonic Velocity
    v_baryon = np.sqrt(v_gas**2 + (np.sqrt(GLOBAL_ML) * v_disk)**2 + v_bul**2)
    
    # 1. Causal Response Model
    r_m = r * KPC_TO_M
    v_baryon_mps = v_baryon * KM_TO_M
    g_bar = np.where(r_m > 0, (v_baryon_mps ** 2) / r_m, 0.0)
    
    w = w_g_kernel(g_bar) * n_analytic(r) * xi_global(g.get('f_gas_true', 0.0))
    if g.get('R_d', 0) > 0:
        hz = HZ_OVER_RD * g['R_d']
        zeta = np.clip(1.0 + 0.5 * (hz / (r + 0.1 * g['R_d'])), 0.8, 1.2)
        w *= zeta
    v_cr = np.sqrt(np.maximum(w, 0)) * v_baryon
    
    # 2. MOND Model (Standard nu)
    y = np.maximum(g_bar / A0_MOND, 1e-30)
    nu = 0.5 + np.sqrt(0.25 + 1.0/y)
    v_mond = np.sqrt(nu * g_bar * r_m) / KM_TO_M
    
    # Full Error Model
    is_dwarf = bool(np.nanmax(v_obs) < 80.0)
    R_d_guess = float(g.get('R_d', 2.0))
    sigma_eff = get_sigma_total(r, v_obs, v_err, R_d_guess, is_dwarf)
    
    valid = np.isfinite(v_obs) & np.isfinite(v_cr) & np.isfinite(v_mond) & (sigma_eff > 0)
    if np.sum(valid) < 3: return None, None, None, None
    
    chi2_cr = np.sum(((v_obs[valid] - v_cr[valid])/sigma_eff[valid])**2) / np.sum(valid)
    chi2_mond = np.sum(((v_obs[valid] - v_mond[valid])/sigma_eff[valid])**2) / np.sum(valid)
    
    # Also compute RMS residuals in km/s
    rms_cr = np.sqrt(np.mean((v_obs[valid] - v_cr[valid])**2))
    rms_mond = np.sqrt(np.mean((v_obs[valid] - v_mond[valid])**2))
    
    return chi2_cr, chi2_mond, rms_cr, rms_mond

def main():
    print("Loading Q=1 Data...")
    with open('external/gravity/active/scripts/sparc_q1.pkl', 'rb') as f:
        data = pickle.load(f)
        
    chi2_cr_list = []
    chi2_mond_list = []
    rms_cr_list = []
    rms_mond_list = []
    
    for name, g in data.items():
        c, m, rms_c, rms_m = compute_chi2(g)
        if c is not None:
            chi2_cr_list.append(c)
            chi2_mond_list.append(m)
            rms_cr_list.append(rms_c)
            rms_mond_list.append(rms_m)
            
    # 1. Wilcoxon Signed-Rank Test
    stat, p_val = wilcoxon(chi2_cr_list, chi2_mond_list, alternative='less') # Testing if CR < MOND
    
    print(f"\n--- Statistical Significance ---")
    print(f"N = {len(chi2_cr_list)}")
    print(f"Median Chi2/N (CR):   {np.median(chi2_cr_list):.3f}")
    print(f"Mean Chi2/N (CR):     {np.mean(chi2_cr_list):.2f}")
    print(f"Median Chi2/N (MOND): {np.median(chi2_mond_list):.3f}")
    print(f"Mean Chi2/N (MOND):   {np.mean(chi2_mond_list):.2f}")
    print(f"Wilcoxon Statistic:   {stat}")
    print(f"P-value (One-sided):  {p_val:.2e}")
    if p_val < 0.05:
        print(f"*** CR is SIGNIFICANTLY better than MOND (p < 0.05) ***")
    else:
        print(f"*** CR is NOT significantly better than MOND (p >= 0.05) ***")
    
    # RMS Residuals
    print(f"\n--- RMS Residuals (km/s) ---")
    print(f"Median RMS (CR):   {np.median(rms_cr_list):.2f}")
    print(f"Median RMS (MOND): {np.median(rms_mond_list):.2f}")
    print(f"Improvement:       {100*(1 - np.median(rms_cr_list)/np.median(rms_mond_list)):.1f}%")
    
    # Count outliers
    outliers_cr = np.sum(np.array(chi2_cr_list) > 5.0)
    outliers_mond = np.sum(np.array(chi2_mond_list) > 5.0)
    print(f"\n--- Outliers (Chi2/N > 5) ---")
    print(f"CR:   {outliers_cr} galaxies")
    print(f"MOND: {outliers_mond} galaxies")
    
    # 2. Tau-star Calculation
    # Formula: tau = sqrt(2 * pi * r0 / a0)
    # Units: r0 in meters, a0 in m/s^2
    
    r0_m = N_R0 * KPC_TO_M
    tau_sq = 2 * np.pi * r0_m / A0_OPT
    tau_s = np.sqrt(tau_sq)
    tau_gyr = tau_s / GYR_TO_S
    
    # Hubble Time
    # H0 ~ 70 km/s/Mpc
    # 1/H0 ~ 13.96 Gyr
    H0_inv_gyr = 13.96 
    
    print(f"\n--- Timescale Analysis ---")
    print(f"Fitted r0: {N_R0:.2f} kpc")
    print(f"Fitted a0: {A0_OPT:.2e} m/s^2")
    print(f"Derived tau_star: {tau_gyr:.2f} Gyr = {tau_gyr*1000:.0f} Myr")
    print(f"Hubble Time (1/H0): {H0_inv_gyr:.2f} Gyr")
    print(f"Ratio (tau / H_inv): {tau_gyr/H0_inv_gyr:.3f}")
    print(f"Interpretation: tau_star ~ galactic dynamical timescale, NOT cosmological")

if __name__ == "__main__":
    main()

