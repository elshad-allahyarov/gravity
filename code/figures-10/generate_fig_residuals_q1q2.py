import matplotlib.pyplot as plt
import numpy as np
import pickle
import math
from scipy.stats import norm

# Parameters
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

def compute_model_velocity(r_kpc, v_gas, v_disk, v_bul, f_gas_true, R_d_guess):
    v_baryon = np.sqrt(v_gas ** 2 + (math.sqrt(GLOBAL_ML) * v_disk) ** 2 + v_bul ** 2)
    r_m = r_kpc * KPC_TO_M
    v_baryon_mps = v_baryon * KM_TO_M
    g_bar = np.where(r_m > 0, (v_baryon_mps ** 2) / r_m, 0.0)
    w_g = w_g_kernel(g_bar)
    n_r = n_analytic(r_kpc)
    xi = xi_global(f_gas_true)
    if R_d_guess <= 0: zeta = 1.0
    else:
        hz = HZ_OVER_RD * R_d_guess
        zeta = np.clip(1.0 + 0.5 * (hz / (r_kpc + 0.1 * R_d_guess)), 0.8, 1.2)
    w = w_g * n_r * xi * zeta
    v_model = np.sqrt(np.maximum(w, 0.0)) * v_baryon
    return v_model

def get_sigma_total(r_kpc, v_obs, v_err, R_d_guess):
    is_dwarf = bool(np.nanmax(v_obs) < 80.0)
    sigma = v_err ** 2 + SIGMA0 ** 2 + (F_FLOOR * v_obs) ** 2
    if R_d_guess > 0:
        beam_kpc = 0.3 * R_d_guess
        sigma += (ALPHA_BEAM * beam_kpc * v_obs / (r_kpc + beam_kpc)) ** 2
    asym_frac = ASYM_DWARF if is_dwarf else ASYM_SPIRAL
    sigma += (asym_frac * v_obs) ** 2
    sigma_turb = K_TURB * v_obs * (1.0 - np.exp(-r_kpc / (R_d_guess if R_d_guess > 0 else 2.0))) ** P_TURB
    sigma += sigma_turb ** 2
    return np.sqrt(np.maximum(sigma, 1e-10))

# Load Data
print("Loading sparc_master.pkl...")
with open('external/gravity/active/scripts/sparc_master.pkl', 'rb') as f:
    data = pickle.load(f)

# Filter Q=1+2
q_dict = {}
try:
    with open('0-0-0-0-0-sparc-data.txt', 'r') as f:
        for line in f:
            p = line.strip().split()
            if len(p)>1: q_dict[p[0]] = int(p[-2]) if p[-2].isdigit() else 0
except: pass

all_res = []
for name, g in data.items():
    if q_dict.get(name, 0) not in [1, 2]: continue
    
    df = g['data']
    r = df['rad'].values
    v_obs = df['vobs'].values
    v_err = df['verr'].values
    
    v_model = compute_model_velocity(r, df['vgas'].values, df['vdisk'].values, df['vbul'].values, g['f_gas_true'], g.get('R_d', 2.0))
    sig = get_sigma_total(r, v_obs, v_err, g.get('R_d', 2.0))
    
    valid = (v_obs > 0) & (sig > 0)
    if np.sum(valid) > 0:
        res = (v_obs[valid] - v_model[valid]) / sig[valid]
        all_res.extend(res)

all_res = np.array(all_res)
print(f"Computed {len(all_res)} residuals from Q=1+2.")

# Plot A: Histogram
plt.style.use('default')
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 10
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'

fig, ax = plt.subplots(figsize=(7, 6))
counts, bins, _ = ax.hist(all_res, bins=50, density=True, alpha=0.6, color='blue', label='Normalized Residuals')

# Fit
_, std_fit = norm.fit(all_res)
peak_idx = np.argmax(counts)
mu_fit = (bins[peak_idx] + bins[peak_idx+1]) / 2.0

xmin, xmax = ax.get_xlim()
x = np.linspace(xmin, xmax, 100)
p = norm.pdf(x, mu_fit, std_fit)
ax.plot(x, p, 'r-', linewidth=2, label=f'Best Fit ($\mu={mu_fit:.2f}, \sigma={std_fit:.2f}$)')

ax.set_xlabel(r'Residual $(v_{\rm obs} - v_{\rm model}) / \sigma$')
ax.set_ylabel('Density')
ax.set_title('Residual Distribution (Q=1+2)')
ax.legend()
plt.tight_layout()
plt.savefig('frg_residuals_a_q1q2.pdf')
plt.savefig('frg_residuals_a_q1q2.eps')
print("Generated frg_residuals_a_q1q2")

# Plot B: Q-Q
fig, ax = plt.subplots(figsize=(6, 6))
sorted_res = np.sort(all_res)
probs = (np.arange(len(sorted_res)) + 0.5) / len(sorted_res)

# Use actual stats for Q-Q reference to check shape
mu_qq, std_qq = norm.fit(all_res)
expected_quantiles = norm.ppf(probs, loc=mu_qq, scale=std_qq)

ax.scatter(expected_quantiles, sorted_res, alpha=0.5, s=10, color='blue', label='Residuals')

min_val = min(np.min(expected_quantiles), np.min(sorted_res))
max_val = max(np.max(expected_quantiles), np.max(sorted_res))
ax.plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=1.5, label='Perfect Gaussian')

ax.set_xlabel(f'Theoretical Quantiles ($N(\mu={mu_qq:.2f}, \sigma={std_qq:.2f})$)')
ax.set_ylabel('Sample Quantiles')
ax.set_title('Q-Q Plot (Q=1+2)')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('frg_residuals_b_q1q2.pdf')
plt.savefig('frg_residuals_b_q1q2.eps')
print("Generated frg_residuals_b_q1q2")

