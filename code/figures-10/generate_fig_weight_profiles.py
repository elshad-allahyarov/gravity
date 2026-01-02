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
KPC_TO_M = 3.086e19
KM_TO_M = 1.0e3
GLOBAL_ML = 1.0

def n_analytic(r_kpc): return 1.0 + N_A * (1.0 - np.exp(-((r_kpc / N_R0) ** N_P)))
def xi_global(f_gas_true): return 1.0 + C_LAG * (max(f_gas_true, 0.0) ** 0.5)
def w_g_kernel(g_bar_si):
    term = np.maximum(g_bar_si / A0_OPT, 1e-30)
    return 1.0 + C_LAG * (term ** (-ALPHA) - 1.0)

def compute_weight(g):
    df = g['data']
    r = df['rad'].values
    v_gas = df['vgas'].values
    v_disk = df['vdisk'].values
    v_bul = df['vbul'].values
    
    v_baryon = np.sqrt(v_gas**2 + v_disk**2 + v_bul**2)
    r_m = r * KPC_TO_M
    v_baryon_mps = v_baryon * KM_TO_M
    g_bar = np.where(r_m > 0, (v_baryon_mps ** 2) / r_m, 0.0)
    
    w_g = w_g_kernel(g_bar)
    n_r = n_analytic(r)
    xi = xi_global(g['f_gas_true'])
    
    if g.get('R_d', 0) > 0:
        hz = HZ_OVER_RD * g['R_d']
        zeta = np.clip(1.0 + 0.5 * (hz / (r + 0.1 * g['R_d'])), 0.8, 1.2)
    else:
        zeta = 1.0
        
    w = w_g * n_r * xi * zeta
    return r, w

def main():
    with open('external/gravity/active/scripts/sparc_q1.pkl', 'rb') as f:
        data = pickle.load(f)
        
    targets = ['DDO161', 'NGC2403', 'NGC3198', 'NGC7814']
    labels = ['DDO 161 (Dwarf)', 'NGC 2403 (LSB)', 'NGC 3198 (Spiral)', 'NGC 7814 (Massive)']
    colors = ['blue', 'cyan', 'green', 'red']
    
    plt.style.use('default')
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.size'] = 10
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    
    fig, ax = plt.subplots(figsize=(7, 5))
    
    for name, label, color in zip(targets, labels, colors):
        if name in data:
            r, w = compute_weight(data[name])
            ax.plot(r, w, color=color, linewidth=2, label=label)
            
    ax.set_xlabel('Radius (kpc)')
    ax.set_ylabel('Gravitational Enhancement $w(r)$')
    ax.set_title('Radial Profile of the Response Weight')
    ax.axhline(1.0, color='k', linestyle=':', label='Newtonian')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('frg_weight_profiles.pdf')
    plt.savefig('frg_weight_profiles.eps')
    print("Generated frg_weight_profiles.pdf")

if __name__ == "__main__":
    main()

