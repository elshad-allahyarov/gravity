import matplotlib.pyplot as plt
import numpy as np
import pickle
import math
from scipy.stats import linregress

# Parameters
ALPHA = 0.3899
C_LAG = 0.2981
N_A, N_R0, N_P = 1.0599, 17.7895, 0.9541
HZ_OVER_RD = 0.1540
A0_OPT = 1.95e-10
A0_MOND = 1.2e-10
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

def compute_models(g):
    df = g['data']
    r = df['rad'].values
    v_obs = df['vobs'].values
    v_err = df['verr'].values
    
    v_baryon = np.sqrt(df['vgas']**2 + df['vdisk']**2 + df['vbul']**2)
    r_m = r * KPC_TO_M
    v_baryon_mps = v_baryon * KM_TO_M
    g_bar = np.where(r_m > 0, (v_baryon_mps ** 2) / r_m, 0.0)
    
    # CR
    w = w_g_kernel(g_bar) * n_analytic(r) * xi_global(g['f_gas_true'])
    if g.get('R_d', 0) > 0:
        w *= np.clip(1.0 + 0.5 * (HZ_OVER_RD * g['R_d'] / (r + 0.1 * g['R_d'])), 0.8, 1.2)
    v_cr = np.sqrt(np.maximum(w, 0)) * v_baryon
    
    # MOND
    y = np.maximum(g_bar / A0_MOND, 1e-30)
    nu = 0.5 + np.sqrt(0.25 + 1.0/y)
    v_mond = np.sqrt(nu * g_bar * r_m) / KM_TO_M
    
    # Sigma
    sigma = v_err**2 + SIGMA0**2 + (F_FLOOR * v_obs)**2
    if g.get('R_d', 0) > 0: sigma += (ALPHA_BEAM * 0.3 * g['R_d'] * v_obs / (r + 0.3*g['R_d']))**2
    is_dwarf = bool(np.nanmax(v_obs) < 80.0)
    asym_frac = ASYM_DWARF if is_dwarf else ASYM_SPIRAL
    sigma += (asym_frac * v_obs)**2
    sigma += (K_TURB * v_obs * (1.0 - np.exp(-r / (g.get('R_d', 2.0))))**P_TURB)**2
    sig = np.sqrt(np.maximum(sigma, 1e-10))
    
    valid = (v_obs > 0) & (sig > 0)
    if np.sum(valid) < 3: return None
    
    # RMS Residual (unweighted or weighted? Chi2/N is weighted squared. Let's use RMS of weighted residuals)
    res_cr = (v_obs[valid] - v_cr[valid]) / sig[valid]
    res_mond = (v_obs[valid] - v_mond[valid]) / sig[valid]
    
    rms_cr = np.sqrt(np.mean(res_cr**2))
    rms_mond = np.sqrt(np.mean(res_mond**2))
    
    return rms_cr, rms_mond

# Load Q=1
with open('external/gravity/active/scripts/sparc_q1.pkl', 'rb') as f:
    data = pickle.load(f)

fgas_list = []
rms_cr_list = []
rms_mond_list = []

for name, g in data.items():
    res = compute_models(g)
    if res:
        fgas_list.append(g['f_gas_true'])
        rms_cr_list.append(res[0])
        rms_mond_list.append(res[1])

fgas = np.array(fgas_list)
rms_cr = np.array(rms_cr_list)
rms_mond = np.array(rms_mond_list)

# Plot
plt.style.use('default')
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 10
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'

fig, ax = plt.subplots(figsize=(7, 5))

ax.scatter(fgas, rms_mond, c='red', alpha=0.5, label='MOND', s=20)
ax.scatter(fgas, rms_cr, c='blue', alpha=0.5, label='Causal-response', s=20)

# Fits
slope_cr, intercept_cr, r_cr, p_cr, _ = linregress(fgas, rms_cr)
slope_mond, intercept_mond, r_mond, p_mond, _ = linregress(fgas, rms_mond)

x_fit = np.linspace(0, 1, 10)
ax.plot(x_fit, slope_cr*x_fit + intercept_cr, 'b--', linewidth=2, label=f'CR Trend (r={r_cr:.2f})')
ax.plot(x_fit, slope_mond*x_fit + intercept_mond, 'r--', linewidth=2, label=f'MOND Trend (r={r_mond:.2f})')

ax.set_xlabel(r'Gas Fraction $f_{\rm gas}$')
ax.set_ylabel(r'RMS Weighted Residual ($\sqrt{\chi^2/N}$)')
ax.set_title('Residuals vs. Gas Fraction')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('frg_res_vs_fgas.pdf')
plt.savefig('frg_res_vs_fgas.eps')
print("Generated frg_res_vs_fgas.pdf")

