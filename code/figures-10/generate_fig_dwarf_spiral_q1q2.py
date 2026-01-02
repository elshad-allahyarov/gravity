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

def get_galaxy_type(v_obs):
    if len(v_obs) == 0: return 'Unknown'
    v_max = np.nanmax(v_obs)
    if v_max < 80.0: return 'Dwarf'
    if v_max > 200.0: return 'Massive'
    return 'Spiral'

def compute_w_at_rd(g):
    df = g['data']
    r = df['rad'].values
    R_d = g.get('R_d', 2.0)
    if len(r) == 0: return None
    
    # Find index closest to R_d
    idx = (np.abs(r - R_d)).argmin()
    
    v_baryon = np.sqrt(df['vgas']**2 + df['vdisk']**2 + df['vbul']**2)
    r_m = r * KPC_TO_M
    v_baryon_mps = v_baryon * KM_TO_M
    g_bar = np.where(r_m > 0, (v_baryon_mps ** 2) / r_m, 0.0)
    
    w_g = w_g_kernel(g_bar)
    n_r = n_analytic(r)
    xi = xi_global(g['f_gas_true'])
    
    if R_d > 0:
        hz = HZ_OVER_RD * R_d
        zeta = np.clip(1.0 + 0.5 * (hz / (r + 0.1 * R_d)), 0.8, 1.2)
    else:
        zeta = 1.0
        
    w = w_g * n_r * xi * zeta
    return w[idx]

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

w_dwarfs = []
w_spirals = [] # Includes Spirals + Massive

for name, g in data.items():
    if q_dict.get(name, 0) not in [1, 2]: continue
    
    w = compute_w_at_rd(g)
    if w is None: continue
    
    gtype = get_galaxy_type(g['data']['vobs'].values)
    if gtype == 'Dwarf':
        w_dwarfs.append(w)
    else:
        w_spirals.append(w) # Spirals + Massive

print(f"Dwarfs: {len(w_dwarfs)}, Spirals (incl Massive): {len(w_spirals)}")

# Plot
plt.style.use('default')
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 10
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'

fig, ax = plt.subplots(figsize=(7, 5))
y1 = w_dwarfs
x1 = np.random.normal(1, 0.04, len(y1))
y2 = w_spirals
x2 = np.random.normal(2, 0.04, len(y2))

ax.scatter(x1, y1, alpha=0.3, s=20, color='blue', edgecolors='none')
ax.scatter(x2, y2, alpha=0.3, s=20, color='green', edgecolors='none')

med_d = np.median(y1)
med_s = np.median(y2)
ax.scatter([1, 2], [med_d, med_s], color='red', s=80, zorder=3)

ax.set_xticks([1, 2])
ax.set_xticklabels([f'Dwarfs\n(N={len(y1)})', f'Spirals\n(N={len(y2)})'])
ax.set_ylabel(r'Gravitational enhancement $w(r \approx R_d)$')
ax.set_title('Dwarf vs. Spiral Enhancement (Q=1+2)')

plt.tight_layout()
plt.savefig('frg_dwarf_spiral_q1q2.pdf')
plt.savefig('frg_dwarf_spiral_q1q2.eps')
print("Generated frg_dwarf_spiral_q1q2.pdf and .eps")

