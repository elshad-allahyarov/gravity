import matplotlib.pyplot as plt
import numpy as np
import pickle
import math

# Parameters (Same as Q=1 Optimized)
ALPHA = 0.3899
C_LAG = 0.2981
N_A, N_R0, N_P = 1.0599, 17.7895, 0.9541
HZ_OVER_RD = 0.1540
A0_OPT = 1.95e-10
A0_MOND = 1.2e-10
KPC_TO_M = 3.086e19
KM_TO_M = 1.0e3
GLOBAL_ML = 1.0

def n_analytic(r_kpc):
    return 1.0 + N_A * (1.0 - np.exp(-((r_kpc / N_R0) ** N_P)))

def xi_global(f_gas_true):
    return 1.0 + C_LAG * (max(f_gas_true, 0.0) ** 0.5)

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

# Load MASTER data (Q1 + Q2 + Q3)
print("Loading sparc_master.pkl...")
with open('external/gravity/active/scripts/sparc_master.pkl', 'rb') as f:
    data = pickle.load(f)

targets = ['DDO161', 'NGC2403', 'NGC3198', 'NGC7814']
titles = ['DDO 161 (Dwarf)', 'NGC 2403 (LSB)', 'NGC 3198 (Spiral)', 'NGC 7814 (Massive)']

fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.flatten()

plt.style.use('default')
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 10
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'

for i, name in enumerate(targets):
    ax = axes[i]
    if name not in data:
        ax.text(0.5, 0.5, f'{name} not found', ha='center')
        continue
        
    g = data[name]
    df = g['data']
    r = df['rad'].values
    v_obs = df['vobs'].values
    v_err = df['verr'].values
    
    v_model = compute_model_velocity(r, df['vgas'].values, df['vdisk'].values, df['vbul'].values, g['f_gas_true'], g.get('R_d', 2.0))
    v_mond = compute_mond_velocity(r, df['vgas'].values, df['vdisk'].values, df['vbul'].values)
    v_baryon = np.sqrt(df['vgas']**2 + df['vdisk']**2 + df['vbul']**2)
    
    ax.errorbar(r, v_obs, yerr=v_err, fmt='ko', label='Observed', markersize=4, alpha=0.7)
    ax.plot(r, v_model, 'b-', label='Causal-response', linewidth=2)
    ax.plot(r, v_mond, 'r--', label='MOND', linewidth=1.5)
    ax.plot(r, v_baryon, 'g:', label='Newtonian', linewidth=2)
    
    ax.set_title(titles[i] + ' (Master Data)', fontsize=12)
    ax.set_xlabel('Radius (kpc)')
    ax.set_ylabel('Velocity (km/s)')
    ax.grid(True, alpha=0.3)
    
    if i == 0:
        ax.legend(loc='lower right', fontsize=9)

plt.tight_layout()
outfile_pdf = 'frg_rotation_curves_q1q2.pdf'
outfile_eps = 'frg_rotation_curves_q1q2.eps'
plt.savefig(outfile_pdf)
plt.savefig(outfile_eps)
print(f"Generated {outfile_pdf} and {outfile_eps}")

