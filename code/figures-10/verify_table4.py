import numpy as np
import pickle
import math

# Parameters (Matches Code/Text)
ALPHA = 0.3899
C_LAG = 0.2981
N_A, N_R0, N_P = 1.0599, 17.7895, 0.9541
HZ_OVER_RD = 0.1540
A0_OPT = 1.95e-10
KPC_TO_M = 3.086e19
KM_TO_M = 1.0e3
GLOBAL_ML = 1.0
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

def compute_chi2_val(g):
    df = g['data']
    r = df['rad'].values
    v_obs = df['vobs'].values
    v_err = df['verr'].values
    v_gas = df['vgas'].values
    v_disk = df['vdisk'].values
    v_bul = df['vbul'].values
    
    v_baryon = np.sqrt(v_gas**2 + v_disk**2 + v_bul**2)
    r_m = r * KPC_TO_M
    v_baryon_mps = v_baryon * KM_TO_M
    g_bar = np.where(r_m > 0, (v_baryon_mps ** 2) / r_m, 0.0)
    
    w = w_g_kernel(g_bar) * n_analytic(r) * xi_global(g['f_gas_true'])
    if g.get('R_d', 0) > 0:
        hz = HZ_OVER_RD * g['R_d']
        zeta = np.clip(1.0 + 0.5 * (hz / (r + 0.1 * g['R_d'])), 0.8, 1.2)
    else:
        zeta = 1.0
    v_model = np.sqrt(np.maximum(w, 0)) * v_baryon
    
    # Error Model (Full)
    is_dwarf = bool(np.nanmax(v_obs) < 80.0)
    sigma = v_err ** 2 + SIGMA0 ** 2 + (F_FLOOR * v_obs) ** 2
    if g.get('R_d', 0) > 0:
        beam_kpc = 0.3 * g['R_d']
        sigma_beam = ALPHA_BEAM * beam_kpc * v_obs / (r + beam_kpc)
        sigma += sigma_beam ** 2
    asym_frac = ASYM_DWARF if is_dwarf else ASYM_SPIRAL
    sigma += (asym_frac * v_obs) ** 2
    sigma += (K_TURB * v_obs * (1.0 - np.exp(-r / g.get('R_d', 2.0)))**P_TURB)**2
    sig = np.sqrt(np.maximum(sigma, 1e-10))
    
    valid = (v_obs > 0) & (sig > 0)
    if np.sum(valid) > 0:
        res = ((v_obs[valid] - v_model[valid]) / sig[valid]) ** 2
        return np.sum(res) / np.sum(valid)
    return None

targets = ['NGC3198', 'DDO161', 'NGC2403', 'NGC7814', 'UGC2885']

with open('external/gravity/active/scripts/sparc_q1.pkl', 'rb') as f:
    data = pickle.load(f)

print("--- Verification of Table 4 ---")
for name in targets:
    if name in data:
        chi2 = compute_chi2_val(data[name])
        print(f"{name}: {chi2:.2f}")

