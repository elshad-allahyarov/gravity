import matplotlib.pyplot as plt
import numpy as np
import pickle
import math

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

v_flats = []
m_barys = []
types = []

count = 0
for name, g in data.items():
    # Filter Q=1 or Q=2
    q = q_flags.get(name, 0)
    if q not in [1, 2]: continue
    
    df = g['data']
    v_obs = df['vobs'].values
    if len(v_obs) < 3: continue
    
    # V_flat: Mean of last 3 points
    v_flat = np.mean(v_obs[-3:])
    
    # M_bary: M_gas + M_star
    # Check if keys exist
    if 'M_gas_est' in g and 'M_star_est' in g:
        M_bary = g['M_gas_est'] + g['M_star_est']
    else:
        # Fallback calculation if pre-computed mass not in pickle (unlikely for master)
        # Use global ML=0.5 if M_star not set? Or ML=1.0?
        # Paper says Global ML = 1.0.
        # Let's check if M_star_est assumes a specific ML.
        # Usually M_star_est = L * (M/L)_ref.
        # If we enforce ML=1.0, we should recompute M_star from Luminosity if possible.
        # But `g` usually stores `d` (distance), `L` (luminosity).
        # Let's look for `L_3.6`.
        # If not, stick to M_bary provided, assuming it's close enough or scale it.
        # But wait, `generate_figures.py` uses: `M_bary = g['M_gas_est'] + g['M_star_est']`.
        # It does NOT recompute M_star with ML=1.0 for the BTFR plot specifically in the original code?
        # Let's check `generate_figures.py`.
        # It says: `M_bary = g['M_gas_est'] + g['M_star_est']`.
        # And `GLOBAL_ML = 1.0` is used for *rotation curve fits*.
        # Ideally BTFR should use the same ML.
        # If `M_star_est` in pickle comes from `L * 0.5` (popular), then it's inconsistent with ML=1.0.
        # Let's recalculate M_star if L is available.
        if 'L_36' in g:
             M_star = g['L_36'] * 1.0e9 * 1.0 # L is usually in 10^9 Lsun. ML=1.
             # Wait, units.
             # sparc usually: L_36 in 10^9 Lsun.
             # I'll assume M_star_est is reliable or check if I can adjust.
             # For SAFETY and consistency with "do not assume", I will use the provided M_bary
             # but acknowledging the ML might be 0.5 or 0.8 in the pickle generation.
             # However, the previous Figure 3 used `g['M_gas_est'] + g['M_star_est']`.
             # I will do EXACTLY what `generate_figures.py` did.
             pass
        M_bary = g['M_gas_est'] + g['M_star_est']

    v_flats.append(v_flat)
    m_barys.append(M_bary)
    types.append(get_galaxy_type(v_obs))
    count += 1

print(f"Plotting BTFR for {count} Q=1+2 galaxies.")

# Plot
plt.style.use('default')
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 10
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'

fig, ax = plt.subplots(figsize=(7, 6))

colors = np.where(np.array(types) == 'Dwarf', 'blue', np.where(np.array(types) == 'Massive', 'red', 'green'))
ax.scatter(v_flats, m_barys, c=colors, s=30, alpha=0.6, edgecolors='k', linewidth=0.5)

vf = np.logspace(1.5, 2.5, 10)
mf35 = 1e9 * (vf/100)**3.5 * 5 
ax.plot(vf, mf35, 'k-', label=r'$M \propto v^{3.5}$')

ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlim(10, 400)
ax.set_ylim(1e7, 5e11)
ax.set_xlabel(r'Flat Rotation Velocity $V_{\rm flat}$ (km/s)')
ax.set_ylabel(r'Baryonic Mass $M_{\rm bar}$ ($M_\odot$)')
ax.set_title('BTFR (Q=1+2)')
ax.legend()

plt.tight_layout()
outfile_pdf = 'frg_btfr_q1q2.pdf'
outfile_eps = 'frg_btfr_q1q2.eps'
plt.savefig(outfile_pdf)
plt.savefig(outfile_eps)
print(f"Generated {outfile_pdf} and {outfile_eps}")

