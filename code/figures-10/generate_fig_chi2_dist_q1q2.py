import matplotlib.pyplot as plt
import numpy as np
import pickle
import math

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

def compute_mond_velocity(r_kpc, v_gas, v_disk, v_bul):
    v_baryon = np.sqrt(v_gas ** 2 + (math.sqrt(GLOBAL_ML) * v_disk) ** 2 + v_bul ** 2)
    g_N = (v_baryon * 1000)**2 / (r_kpc * KPC_TO_M)
    y = np.maximum(g_N / A0_MOND, 1e-30)
    nu = 0.5 + np.sqrt(0.25 + 1.0/y)
    v_mond = np.sqrt(nu * g_N * r_kpc * KPC_TO_M) / 1000
    return v_mond

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

chi2_rs_list = []
chi2_mond_list = []

for name, g in data.items():
    if q_dict.get(name, 0) not in [1, 2]: continue
    
    df = g['data']
    r = df['rad'].values
    v_obs = df['vobs'].values
    v_err = df['verr'].values
    
    v_cr = compute_model_velocity(r, df['vgas'].values, df['vdisk'].values, df['vbul'].values, g['f_gas_true'], g.get('R_d', 2.0))
    v_mond = compute_mond_velocity(r, df['vgas'].values, df['vdisk'].values, df['vbul'].values)
    sig = get_sigma_total(r, v_obs, v_err, g.get('R_d', 2.0))
    
    valid = (v_obs > 0) & (sig > 0)
    if np.sum(valid) > 0:
        chi2_rs_list.append(np.sum(((v_obs[valid]-v_cr[valid])/sig[valid])**2) / np.sum(valid))
        chi2_mond_list.append(np.sum(((v_obs[valid]-v_mond[valid])/sig[valid])**2) / np.sum(valid))

chi2_rs = np.array(chi2_rs_list)
chi2_mond = np.array(chi2_mond_list)

print(f"Computed Chi2 for {len(chi2_rs)} galaxies (Q=1+2).")
print(f"RS Median: {np.median(chi2_rs):.2f}")
print(f"MOND Median: {np.median(chi2_mond):.2f}")

# Plot
plt.style.use('default')
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 10
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'

fig, ax = plt.subplots(figsize=(8, 5))
bins = np.logspace(np.log10(0.1), np.log10(50), 20)

n_cr, bins, _ = ax.hist(chi2_rs, bins=bins, alpha=0.4, color='blue', label='Causal-response')
n_mond, _, _ = ax.hist(chi2_mond, bins=bins, alpha=0.4, color='red', label='MOND')

overlap = np.minimum(n_cr, n_mond)
ax.bar(bins[:-1], overlap, width=np.diff(bins), align='edge', color='green', alpha=0.6, label='Overlap')

ax.axvline(np.median(chi2_rs), color='blue', linestyle='--', linewidth=2, label=f'CR Med ({np.median(chi2_rs):.2f})')
ax.axvline(np.median(chi2_mond), color='red', linestyle='--', linewidth=2, label=f'MOND Med ({np.median(chi2_mond):.2f})')

ax.set_xscale('log')
ax.set_xlabel(r'$\chi^2/N$')
ax.set_ylabel('Count')
ax.set_title('Goodness of Fit (Q=1+2)')
ax.legend()

plt.tight_layout()
plt.savefig('frg_chi2_dist_q1q2.pdf')
plt.savefig('frg_chi2_dist_q1q2.eps')
print("Generated frg_chi2_dist_q1q2")

