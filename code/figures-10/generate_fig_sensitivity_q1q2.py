import matplotlib.pyplot as plt
import numpy as np
import pickle
import math

# Optimized Parameters (Q1 Baseline)
ALPHA_OPT = 0.3899
C_LAG = 0.2981
N_A, N_R0, N_P = 1.0599, 17.7895, 0.9541
HZ_OVER_RD = 0.1540
A0_OPT = 1.95e-10
KPC_TO_M = 3.086e19
KM_TO_M = 1.0e3
GLOBAL_ML = 1.0

# Helper Functions
def n_analytic(r_kpc):
    return 1.0 + N_A * (1.0 - np.exp(-((r_kpc / N_R0) ** N_P)))

def xi_global(f_gas_true):
    return 1.0 + C_LAG * (max(f_gas_true, 0.0) ** 0.5)

def w_g_kernel(g_bar_si, alpha_val, a0_val):
    term = np.maximum(g_bar_si / a0_val, 1e-30)
    return 1.0 + C_LAG * (term ** (-alpha_val) - 1.0)

def compute_chi2_median(data, alpha_val, ml_val):
    chi2_list = []
    
    for name, g in data.items():
        df = g['data']
        r = df['rad'].values
        v_obs = df['vobs'].values
        v_err = df['verr'].values
        
        # Baryon with specific ML
        v_gas = df['vgas'].values
        v_disk = df['vdisk'].values
        v_bul = df['vbul'].values
        
        v_baryon = np.sqrt(v_gas**2 + (math.sqrt(ml_val)*v_disk)**2 + v_bul**2)
        
        r_m = r * KPC_TO_M
        v_baryon_mps = v_baryon * KM_TO_M
        g_bar = np.where(r_m > 0, (v_baryon_mps ** 2) / r_m, 0.0)
        
        w_g = w_g_kernel(g_bar, alpha_val, A0_OPT)
        n_r = n_analytic(r)
        xi = xi_global(g['f_gas_true'])
        
        if g.get('R_d', 0) > 0:
            hz = HZ_OVER_RD * g['R_d']
            zeta = np.clip(1.0 + 0.5 * (hz / (r + 0.1 * g['R_d'])), 0.8, 1.2)
        else:
            zeta = 1.0
            
        w = w_g * n_r * xi * zeta
        v_model = np.sqrt(np.maximum(w, 0.0)) * v_baryon
        
        # Error Model
        sigma = v_err**2 + 10.0**2 + (0.05*v_obs)**2
        if g.get('R_d', 0) > 0:
            sigma += (0.3 * 0.3 * g['R_d'] * v_obs / (r + 0.3*g['R_d']))**2
        is_dwarf = bool(np.nanmax(v_obs) < 80.0)
        asym_frac = 0.10 if is_dwarf else 0.05
        sigma += (asym_frac * v_obs)**2
        sigma += (0.07 * v_obs * (1-np.exp(-r/g.get('R_d', 2.0)))**1.3)**2
        sig = np.sqrt(np.maximum(sigma, 1e-10))
        
        valid = (v_obs > 0) & (sig > 0)
        if np.sum(valid) > 0:
            chi2 = np.sum(((v_obs[valid]-v_model[valid])/sig[valid])**2) / np.sum(valid)
            chi2_list.append(chi2)
            
    return np.median(chi2_list)

# Load Data
print("Loading sparc_master.pkl...")
with open('external/gravity/active/scripts/sparc_master.pkl', 'rb') as f:
    raw_data = pickle.load(f)

# Filter Q=1+2
q_dict = {}
try:
    with open('0-0-0-0-0-sparc-data.txt', 'r') as f:
        for line in f:
            p = line.strip().split()
            if len(p)>1: q_dict[p[0]] = int(p[-2]) if p[-2].isdigit() else 0
except: pass

data = {k: v for k, v in raw_data.items() if q_dict.get(k, 0) in [1, 2]}
print(f"Analying {len(data)} Q=1+2 galaxies.")

# Baseline Chi2
base_chi2 = compute_chi2_median(data, ALPHA_OPT, GLOBAL_ML)
print(f"Baseline Median Chi2/N (alpha={ALPHA_OPT}, ML={GLOBAL_ML}): {base_chi2:.4f}")

# Scan Alpha
alphas = np.linspace(0.30, 0.50, 20)
chi2_alpha = []
for a in alphas:
    c = compute_chi2_median(data, a, GLOBAL_ML)
    chi2_alpha.append(c - base_chi2)

# Scan ML
mls = np.linspace(0.5, 1.5, 20)
chi2_ml = []
for m in mls:
    c = compute_chi2_median(data, ALPHA_OPT, m)
    chi2_ml.append(c - base_chi2)

# Plot
plt.style.use('default')
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 10
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

# Alpha Plot
ax1.plot(alphas, chi2_alpha, 'b-o', markersize=4)
ax1.axvline(ALPHA_OPT, color='k', linestyle='--', label='Fiducial')
ax1.set_xlabel(r'Dynamical exponent $\alpha$')
ax1.set_ylabel(r'$\Delta$ Median $\chi^2/N$')
ax1.set_title('Sensitivity to Alpha (Q=1+2)')
ax1.legend()
ax1.grid(True, alpha=0.3)

# ML Plot
ax2.plot(mls, chi2_ml, 'g-s', markersize=4)
ax2.axvline(GLOBAL_ML, color='k', linestyle='--', label='Fiducial')
ax2.set_xlabel(r'Global $M/L$')
ax2.set_ylabel(r'$\Delta$ Median $\chi^2/N$')
ax2.set_title('Sensitivity to M/L (Q=1+2)')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('frg_sensitivity_q1q2.pdf')
plt.savefig('frg_sensitivity_q1q2.eps')
print("Generated frg_sensitivity_q1q2.pdf and .eps")

