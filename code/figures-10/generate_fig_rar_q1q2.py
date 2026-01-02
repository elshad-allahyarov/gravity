import matplotlib.pyplot as plt
import numpy as np
import pickle
import math

# Parameters
ALPHA = 0.3899
C_LAG = 0.2981
A0_OPT = 1.95e-10
A0_MOND = 1.2e-10
KPC_TO_M = 3.086e19
KM_TO_M = 1.0e3
GLOBAL_ML = 1.0

# Q-Flag Loader
def load_q_flags():
    q_dict = {}
    try:
        with open('0-0-0-0-0-sparc-data.txt', 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) > 1:
                    try:
                        q_dict[parts[0]] = int(parts[-2])
                    except ValueError:
                        pass
    except FileNotFoundError:
        print("Warning: 0-0-0-0-0-sparc-data.txt not found. Assuming all Q=1.")
    return q_dict

def get_galaxy_type(v_obs):
    if len(v_obs) == 0: return 'Unknown'
    v_max = np.nanmax(v_obs)
    if v_max < 80.0: return 'Dwarf'
    if v_max > 200.0: return 'Massive'
    return 'Spiral'

# Load Data
print("Loading sparc_master.pkl...")
with open('external/gravity/active/scripts/sparc_master.pkl', 'rb') as f:
    data = pickle.load(f)

q_flags = load_q_flags()

all_g_bar = []
all_g_obs = []
all_types = []

for name, g in data.items():
    # Filter Q=1 or Q=2
    q = q_flags.get(name, 0)
    if q not in [1, 2]: continue
    
    df = g['data']
    r = df['rad'].values
    v_obs = df['vobs'].values
    if len(v_obs) == 0: continue
    
    valid = (v_obs > 0) & (r > 0)
    if not np.any(valid): continue
    
    v_gas = df['vgas'].values[valid]
    v_disk = df['vdisk'].values[valid]
    v_bul = df['vbul'].values[valid]
    r_kpc = r[valid]
    v_o = v_obs[valid]
    
    v_baryon = np.sqrt(v_gas**2 + (math.sqrt(GLOBAL_ML)*v_disk)**2 + v_bul**2)
    g_b = (v_baryon * 1000)**2 / (r_kpc * KPC_TO_M)
    g_o = (v_o * 1000)**2 / (r_kpc * KPC_TO_M)
    
    all_g_bar.extend(g_b)
    all_g_obs.extend(g_o)
    gtype = get_galaxy_type(df['vobs'].values)
    all_types.extend([gtype] * len(g_b))

all_g_bar = np.array(all_g_bar)
all_g_obs = np.array(all_g_obs)
all_types = np.array(all_types)

print(f"Plotting RAR for {len(all_g_bar)} points from Q=1+2 galaxies.")

# Plot
plt.style.use('default')
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 10
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'

fig, ax = plt.subplots(figsize=(7, 6))

colors = np.where(all_types == 'Dwarf', 'blue', np.where(all_types == 'Massive', 'red', 'green'))
ax.scatter(all_g_bar, all_g_obs, c=colors, s=10, alpha=0.4, edgecolors='none')

# Model Line
gb_line = np.logspace(-13, -8, 100)
term = np.maximum(gb_line / A0_OPT, 1e-30)
w_line = 1.0 + C_LAG * (term ** (-ALPHA) - 1.0)
go_line = w_line * gb_line

# MOND Line
y_mond = np.maximum(gb_line / A0_MOND, 1e-30)
nu_mond = 0.5 + np.sqrt(0.25 + 1.0/y_mond)
go_mond = nu_mond * gb_line

ax.plot(gb_line, go_mond, 'm--', linewidth=2.5, label='MOND ($a_0=1.2\\times 10^{-10}$)')
ax.plot(gb_line, go_line, 'k-', linewidth=2.5, label=f'Model ($\\alpha={ALPHA:.2f}$)')
ax.plot([1e-13, 1e-8], [1e-13, 1e-8], 'k--', alpha=0.5, label='1:1')

ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel(r'Baryonic Acceleration $g_{\rm bar}$ (m/s$^2$)')
ax.set_ylabel(r'Observed Acceleration $g_{\rm obs}$ (m/s$^2$)')
ax.set_title(f'Radial Acceleration Relation (Q=1+2)')
ax.set_xlim(1e-13, 1e-8)
ax.set_ylim(1e-13, 1e-8)
ax.legend(loc='upper left')

plt.tight_layout()
outfile_pdf = 'frg_rar_q1q2.pdf'
outfile_eps = 'frg_rar_q1q2.eps'
plt.savefig(outfile_pdf)
plt.savefig(outfile_eps)
print(f"Generated {outfile_pdf} and {outfile_eps}")

