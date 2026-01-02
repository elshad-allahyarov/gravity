import matplotlib.pyplot as plt
import numpy as np
import pickle
from scipy.stats import linregress
import math

# Parameters
ALPHA = 0.210
C_LAG = 0.094
N_A, N_R0, N_P = 3.66, 10.14, 1.66
HZ_OVER_RD = 0.1540
A0_OPT = 1.2e-10
KPC_TO_M = 3.086e19
KM_TO_M = 1.0e3
GLOBAL_ML = 1.0

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
    return v_model, v_baryon, w_g

# Load Q=1
with open('external/gravity/active/scripts/sparc_q1.pkl', 'rb') as f:
    data = pickle.load(f)

v_flat_list = []
r_trans_data = []
r_trans_model = []
types = []

for name, g in data.items():
    df = g['data']
    r = df['rad'].values
    v_obs = df['vobs'].values
    if len(v_obs) < 5: continue
    
    v_model, v_bar, w_g = compute_model_velocity(r, df['vgas'].values, df['vdisk'].values, df['vbul'].values, g['f_gas_true'], g.get('R_d', 2.0))
    
    # Data Transition
    ratio_data = v_obs / (v_bar + 1e-5)
    idx_d = -1
    for i in range(len(r)):
        if ratio_data[i] > 1.2:
            if i+3 < len(r) and np.mean(ratio_data[i:i+3]) > 1.1:
                idx_d = i
                break
            elif i == len(r)-1: # End point
                idx_d = i
                break
    
    # Model Transition (Kernel Driven)
    # We define transition where the Memory Kernel (w_g) provides > 20% enhancement
    # i.e., sqrt(w_g) > 1.2. This isolates the acceleration scale from gas fraction effects.
    ratio_model = np.sqrt(np.maximum(w_g, 0.1))
    idx_m = -1
    for i in range(len(r)):
        if ratio_model[i] > 1.2:
            if i+3 < len(r) and np.mean(ratio_model[i:i+3]) > 1.1:
                idx_m = i
                break
            elif i == len(r)-1:
                idx_m = i
                break
                
    if idx_d != -1:
        v_flat = np.mean(v_obs[-3:])
        v_flat_list.append(v_flat)
        r_trans_data.append(r[idx_d])
        
        if idx_m != -1:
            r_trans_model.append(r[idx_m])
        else:
            r_trans_model.append(np.nan)
            
        v_max = np.nanmax(v_obs)
        if v_max < 80: t = 'Dwarf'
        elif v_max > 200: t = 'Massive'
        else: t = 'Spiral'
        types.append(t)

v_flat = np.array(v_flat_list)
r_trans_data = np.array(r_trans_data)
r_trans_model = np.array(r_trans_model)
types = np.array(types)

# Filter valid model points
mask = ~np.isnan(r_trans_model)
v_flat = v_flat[mask]
r_trans_data = r_trans_data[mask]
r_trans_model = r_trans_model[mask]
types = types[mask]

# Plot
plt.style.use('default')
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 10
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'

fig, ax = plt.subplots(figsize=(7, 5))

# Data Points
colors = np.where(types == 'Dwarf', 'blue', np.where(types == 'Massive', 'red', 'green'))
ax.scatter(v_flat, r_trans_data, c=colors, alpha=0.4, s=30, label='Observed Data')

# Model Trend (Binning or Fit)
# Fit Log-Log for Data
slope_d, int_d, _, _, _ = linregress(np.log10(v_flat), np.log10(r_trans_data))
x_fit = np.logspace(1.3, 2.6, 10)
y_fit_d = 10**int_d * x_fit**slope_d
ax.plot(x_fit, y_fit_d, 'k--', linewidth=2, label=f'Data Fit ($R \\propto V^{{{slope_d:.2f}}}$)')

# Fit Log-Log for Model
slope_m, int_m, _, _, _ = linregress(np.log10(v_flat), np.log10(r_trans_model))
y_fit_m = 10**int_m * x_fit**slope_m
ax.plot(x_fit, y_fit_m, 'b-', linewidth=2, label=f'CR Model ($R \\propto V^{{{slope_m:.2f}}}$)')

# MOND Prediction
a0 = 1.2e-10
kpc = 3.086e19
y_mond = (x_fit * 1000)**2 / a0 / kpc
ax.plot(x_fit, y_mond, 'r:', alpha=0.7, label=r'MOND Prediction ($R \propto V^2$)')

ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel(r'Flat Rotation Velocity (km/s)')
ax.set_ylabel(r'Transition Radius $R_{\rm trans}$ (kpc)')
ax.set_title('Transition Radius Scaling: Model vs Data')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('frg_transition_scaling.pdf')
plt.savefig('frg_transition_scaling.eps')
print(f"Generated frg_transition_scaling.pdf. Data Slope={slope_d:.2f}, CR Model Slope={slope_m:.2f}")
