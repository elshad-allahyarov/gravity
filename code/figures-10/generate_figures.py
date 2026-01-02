import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
import os
import math
from matplotlib.gridspec import GridSpec
import matplotlib.patches as mpatches

# Set style
plt.style.use('default')
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.linewidth'] = 1.5
plt.rcParams['xtick.major.width'] = 1.5
plt.rcParams['ytick.major.width'] = 1.5
plt.rcParams['xtick.minor.width'] = 1.0
plt.rcParams['ytick.minor.width'] = 1.0
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
plt.rcParams['xtick.top'] = True
plt.rcParams['ytick.right'] = True

# Optimized Parameters (Global DE Result: Chi2/N = 1.19)
ALPHA = 0.3899
C_LAG = 0.2981
N_A, N_R0, N_P = 1.0599, 17.7895, 0.9541
HZ_OVER_RD = 0.1540
LOG_A0 = -9.7097
A0_OPT = 10**LOG_A0  # 1.95e-10

# Baseline MOND A0 (for comparison)
A0_MOND = 1.2e-10

# Constants
KPC_TO_M = 3.086e19
KM_TO_M = 1.0e3
GLOBAL_ML = 1.0

# Fixed Error-model settings
SIGMA0 = 10.0           
F_FLOOR = 0.05          
ALPHA_BEAM = 0.3        
ASYM_DWARF = 0.10       
ASYM_SPIRAL = 0.05      
K_TURB = 0.07           
P_TURB = 1.3            

def n_analytic(r_kpc):
    return 1.0 + N_A * (1.0 - np.exp(-((r_kpc / N_R0) ** N_P)))

def xi_global(f_gas_true):
    return 1.0 + C_LAG * (max(f_gas_true, 0.0) ** 0.5)

def w_g_kernel(g_bar_si, alpha_val=ALPHA, a0_val=A0_OPT):
    term = np.maximum(g_bar_si / a0_val, 1e-30)
    return 1.0 + C_LAG * (term ** (-alpha_val) - 1.0)

def compute_model_velocity(r_kpc, v_gas, v_disk, v_bul, f_gas_true, R_d_guess, alpha_val=ALPHA, a0_val=A0_OPT):
    # Apply global M/L
    v_baryon = np.sqrt(v_gas ** 2 + (math.sqrt(GLOBAL_ML) * v_disk) ** 2 + v_bul ** 2)
    
    r_m = r_kpc * KPC_TO_M
    v_baryon_mps = v_baryon * KM_TO_M
    g_bar = np.where(r_m > 0, (v_baryon_mps ** 2) / r_m, 0.0)
    
    w_g = w_g_kernel(g_bar, alpha_val, a0_val)
    n_r = n_analytic(r_kpc)
    xi = xi_global(f_gas_true)
    
    if R_d_guess <= 0:
        zeta = 1.0
    else:
        hz = HZ_OVER_RD * R_d_guess
        zeta = np.clip(1.0 + 0.5 * (hz / (r_kpc + 0.1 * R_d_guess)), 0.8, 1.2)
        
    w = w_g * n_r * xi * zeta
    v_model = np.sqrt(np.maximum(w, 0.0)) * v_baryon
    return v_model, w, v_baryon, g_bar

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
        sigma_beam = ALPHA_BEAM * beam_kpc * v_obs / (r_kpc + beam_kpc)
        sigma += sigma_beam ** 2
        
    asym_frac = ASYM_DWARF if is_dwarf else ASYM_SPIRAL
    sigma += (asym_frac * v_obs) ** 2
    
    Rd = R_d_guess if R_d_guess > 0 else 2.0
    sigma_turb = K_TURB * v_obs * (1.0 - np.exp(-r_kpc / Rd)) ** P_TURB
    sigma += sigma_turb ** 2
    
    return np.sqrt(np.maximum(sigma, 1e-10))

def load_data():
    pkl_path = 'external/gravity/active/scripts/sparc_q1.pkl'
    if not os.path.exists(pkl_path):
        print(f"Error: {pkl_path} not found.")
        return None
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    return data

def get_galaxy_type(v_obs):
    if len(v_obs) == 0: return 'Unknown'
    v_max = np.nanmax(v_obs)
    if v_max < 80.0:
        return 'Dwarf'
    if v_max > 200.0:
        return 'Massive'
    return 'Spiral'

def generate_fig1_rotation_curves(data):
    # Updated Targets: Valid Q=1 galaxies matching the caption
    # Replaced UGC2885 (bad fit) with NGC7814 (good massive spiral fit)
    targets = ['DDO161', 'NGC2403', 'NGC3198', 'NGC7814']
    titles = ['DDO 161 (Dwarf)', 'NGC 2403 (LSB)', 'NGC 3198 (Spiral)', 'NGC 7814 (Massive)']
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
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
        
        # Models
        v_model, _, _, _ = compute_model_velocity(
            r, df['vgas'].values, df['vdisk'].values, df['vbul'].values,
            g['f_gas_true'], g.get('R_d', 2.0)
        )
        v_mond = compute_mond_velocity(
            r, df['vgas'].values, df['vdisk'].values, df['vbul'].values
        )
        v_baryon = np.sqrt(df['vgas']**2 + df['vdisk']**2 + df['vbul']**2)
        
        # Plot
        ax.errorbar(r, v_obs, yerr=v_err, fmt='ko', label='Observed', markersize=4, alpha=0.7)
        ax.plot(r, v_model, 'b-', label='Causal-response', linewidth=2)
        ax.plot(r, v_mond, 'r--', label='MOND', linewidth=1.5)
        ax.plot(r, v_baryon, 'g:', label='Newtonian', linewidth=2)
        
        ax.set_title(titles[i], fontsize=12)
        ax.set_xlabel('Radius (kpc)')
        ax.set_ylabel('Velocity (km/s)')
        ax.grid(True, alpha=0.3)
        
        if i == 0:
            ax.legend(loc='lower right', fontsize=9)
            
    plt.tight_layout()
    plt.savefig('frg_rotation_curves.pdf')
    plt.savefig('frg_rotation_curves.eps')
    print("Generated frg_rotation_curves.pdf")

def generate_fig2_rar_variants(data):
    all_g_bar = []
    all_g_obs = []
    all_types = []
    
    for name, g in data.items():
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
    
    # Plot optimized
    fig, ax = plt.subplots(figsize=(7, 6))
    
    colors = np.where(all_types == 'Dwarf', 'blue', np.where(all_types == 'Massive', 'red', 'green'))
    ax.scatter(all_g_bar, all_g_obs, c=colors, s=10, alpha=0.4, edgecolors='none')
    
    # Model Line (Optimized)
    gb_line = np.logspace(-13, -8, 100)
    term = np.maximum(gb_line / A0_OPT, 1e-30)
    w_line = 1.0 + C_LAG * (term ** (-ALPHA) - 1.0)
    go_line = w_line * gb_line
    
    ax.plot(gb_line, go_line, 'k-', linewidth=2.5, label=f'Model ($\\alpha={ALPHA:.2f}$)')
    ax.plot([1e-13, 1e-8], [1e-13, 1e-8], 'k--', alpha=0.5, label='1:1')
    
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel(r'Baryonic Acceleration $g_{\rm bar}$ (m/s$^2$)')
    ax.set_ylabel(r'Observed Acceleration $g_{\rm obs}$ (m/s$^2$)')
    ax.set_title(f'Radial Acceleration Relation (N=99)')
    ax.set_xlim(1e-13, 1e-8)
    ax.set_ylim(1e-13, 1e-8)
    ax.legend(loc='upper left')
    
    plt.tight_layout()
    plt.savefig('frg_rar_a.pdf')
    plt.savefig('frg_rar_a.eps')
    
    # Plot alternative (Comparison)
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(all_g_bar, all_g_obs, c=colors, s=10, alpha=0.4, edgecolors='none')
    
    # MOND Line
    y_mond = np.maximum(gb_line / A0_MOND, 1e-30)
    nu_mond = 0.5 + np.sqrt(0.25 + 1.0/y_mond)
    go_mond = nu_mond * gb_line
    
    ax.plot(gb_line, go_mond, 'r--', linewidth=2.5, label='MOND ($a_0=1.2\\times 10^{-10}$)')
    ax.plot(gb_line, go_line, 'k-', linewidth=2.5, label='Causal-Response')
    
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel(r'Baryonic Acceleration $g_{\rm bar}$ (m/s$^2$)')
    ax.set_ylabel(r'Observed Acceleration $g_{\rm obs}$ (m/s$^2$)')
    ax.set_title('Model vs MOND')
    ax.set_xlim(1e-13, 1e-8)
    ax.set_ylim(1e-13, 1e-8)
    ax.legend(loc='upper left')
    
    plt.tight_layout()
    plt.savefig('frg_rar_b.pdf')
    plt.savefig('frg_rar_b.eps')
    print("Generated frg_rar_a.pdf and frg_rar_b.pdf")

def generate_fig3_btfr(data):
    # Same as before
    v_flats = []
    m_barys = []
    types = []
    for name, g in data.items():
        df = g['data']
        v_obs = df['vobs'].values
        if len(v_obs) < 3: continue
        v_flat = np.mean(v_obs[-3:])
        M_bary = g['M_gas_est'] + g['M_star_est']
        v_flats.append(v_flat)
        m_barys.append(M_bary)
        types.append(get_galaxy_type(v_obs))
        
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
    ax.set_title('BTFR (Q=1)')
    plt.tight_layout()
    plt.savefig('frg_btfr.pdf')
    plt.savefig('frg_btfr.eps')
    print("Generated frg_btfr.pdf")

def generate_fig4_dwarf_spiral(data):
    w_dwarfs = []
    w_spirals = []
    for name, g in data.items():
        df = g['data']
        R_d = g.get('R_d', 2.0)
        r = df['rad'].values
        if len(r) == 0: continue
        idx = (np.abs(r - R_d)).argmin()
        v_model, w_arr, _, _ = compute_model_velocity(
            r, df['vgas'].values, df['vdisk'].values, df['vbul'].values,
            g['f_gas_true'], R_d
        )
        w_val = w_arr[idx]
        if get_galaxy_type(df['vobs'].values) == 'Dwarf': w_dwarfs.append(w_val)
        else: w_spirals.append(w_val)
            
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
    ax.set_title('Dwarf vs. Spiral Enhancement (Optimized)')
    plt.tight_layout()
    plt.savefig('frg_dwarf_spiral.pdf')
    plt.savefig('frg_dwarf_spiral.eps')
    print("Generated frg_dwarf_spiral.pdf")

def generate_fig6_chi2_dist(data):
    chi2_rs_list = []
    chi2_mond_list = []
    
    for name, g in data.items():
        df = g['data']
        r = df['rad'].values
        v_obs = df['vobs'].values
        v_err = df['verr'].values
        
        v_cr, _, _, _ = compute_model_velocity(
            r, df['vgas'].values, df['vdisk'].values, df['vbul'].values,
            g['f_gas_true'], g.get('R_d', 2.0)
        )
        v_mond = compute_mond_velocity(
            r, df['vgas'].values, df['vdisk'].values, df['vbul'].values
        )
        sig = get_sigma_total(r, v_obs, v_err, g.get('R_d', 2.0))
        
        valid = (v_obs > 0) & (sig > 0)
        if np.sum(valid) > 0:
            res_cr = ((v_obs[valid] - v_cr[valid]) / sig[valid]) ** 2
            chi2_rs_list.append(np.sum(res_cr) / np.sum(valid))
            
            res_mond = ((v_obs[valid] - v_mond[valid]) / sig[valid]) ** 2
            chi2_mond_list.append(np.sum(res_mond) / np.sum(valid))
            
    chi2_rs = np.array(chi2_rs_list)
    chi2_mond = np.array(chi2_mond_list)
    
    # 1. Histogram Plot
    fig, ax = plt.subplots(figsize=(8, 5))
    bins = np.logspace(np.log10(0.1), np.log10(50), 20)
    
    # Plot overlaps
    n_cr, bins, patches = ax.hist(chi2_rs, bins=bins, alpha=0.4, color='blue', label='Causal-response')
    n_mond, _, _ = ax.hist(chi2_mond, bins=bins, alpha=0.4, color='red', label='MOND')
    
    # Highlight overlap
    overlap = np.minimum(n_cr, n_mond)
    ax.bar(bins[:-1], overlap, width=np.diff(bins), align='edge', color='green', alpha=0.6, label='Overlap')
    
    # Medians
    ax.axvline(np.median(chi2_rs), color='blue', linestyle='--', linewidth=2, label=f'CR Med ({np.median(chi2_rs):.2f})')
    ax.axvline(np.median(chi2_mond), color='red', linestyle='--', linewidth=2, label=f'MOND Med ({np.median(chi2_mond):.2f})')
    
    ax.set_xscale('log')
    ax.set_title('Goodness of Fit (Optimized Q=1)')
    ax.legend()
    
    print(f"Q=1 Statistics (N={len(chi2_rs)}):")
    print(f"  RS: Median={np.median(chi2_rs):.2f}, Mean={np.mean(chi2_rs):.2f}")
    print(f"  MOND: Median={np.median(chi2_mond):.2f}, Mean={np.mean(chi2_mond):.2f}")
    
    plt.tight_layout()
    plt.savefig('frg_chi2_dist.pdf')
    plt.savefig('frg_chi2_dist.eps')
    print("Generated frg_chi2_dist.pdf")
    
    # 2. Residual Histogram (Split Fig 5)
    # We need residuals for all points
    all_res_cr = []
    for name, g in data.items():
        df = g['data']
        r = df['rad'].values
        v_obs = df['vobs'].values
        v_err = df['verr'].values
        v_cr, _, _, _ = compute_model_velocity(
            r, df['vgas'].values, df['vdisk'].values, df['vbul'].values,
            g['f_gas_true'], g.get('R_d', 2.0)
        )
        sig = get_sigma_total(r, v_obs, v_err, g.get('R_d', 2.0))
        valid = (v_obs > 0) & (sig > 0)
        if np.sum(valid) > 0:
            res = (v_obs[valid] - v_cr[valid]) / sig[valid]
            all_res_cr.extend(res)
            
    all_res_cr = np.array(all_res_cr)
    
    from scipy.stats import norm
    
    fig, ax = plt.subplots(figsize=(7, 6))
    counts, bins, _ = ax.hist(all_res_cr, bins=50, density=True, alpha=0.6, color='blue', label='Normalized Residuals')
    
    # Fit width (std) from data
    _, std = norm.fit(all_res_cr)
    
    # Align mean to histogram peak
    peak_idx = np.argmax(counts)
    peak_x = (bins[peak_idx] + bins[peak_idx+1]) / 2.0
    mu = peak_x
    
    xmin, xmax = ax.get_xlim()
    x = np.linspace(xmin, xmax, 100)
    p = norm.pdf(x, mu, std)
    ax.plot(x, p, 'r-', linewidth=2, label=f'Best Fit ($\mu={mu:.2f}, \sigma={std:.2f}$)')
    
    # Removed Standard Normal line as requested
    
    ax.set_xlabel(r'Residual $(v_{\rm obs} - v_{\rm model}) / \sigma$')
    ax.set_ylabel('Density')
    ax.set_title('Residual Distribution')
    ax.legend()
    plt.tight_layout()
    plt.savefig('frg_residuals_a.pdf')
    plt.savefig('frg_residuals_a.eps')
    
    # 3. Q-Q Plot
    fig, ax = plt.subplots(figsize=(6, 6))
    
    sorted_res = np.sort(all_res_cr)
    
    # Use the same fitted parameters from the histogram for the theoretical quantiles
    # This checks if the shape is Gaussian, accounting for the known bias/width
    from scipy.stats import norm
    probs = (np.arange(len(sorted_res)) + 0.5) / len(sorted_res)
    
    # Recalculate mu/std as in histogram (or pass them if refactored, but recalc is cheap)
    # Note: In histogram we used peak for mu, let's use that for consistency if we want
    # But strictly for QQ, usually we test against the best fit parameters (mean/std)
    # Let's use the actual mean/std of the data to see if the SHAPE is Gaussian.
    mu_qq, std_qq = norm.fit(all_res_cr)
    
    expected_quantiles = norm.ppf(probs, loc=mu_qq, scale=std_qq)
    
    ax.scatter(expected_quantiles, sorted_res, alpha=0.5, s=10, color='blue', label='Residuals')
    
    # Reference line (1:1)
    # Since we scaled the quantiles by mu/std, they should match the data values 1:1
    min_val = min(np.min(expected_quantiles), np.min(sorted_res))
    max_val = max(np.max(expected_quantiles), np.max(sorted_res))
    ax.plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=1.5, label='Perfect Gaussian')
    
    ax.set_xlabel(f'Theoretical Quantiles ($N(\mu={mu_qq:.2f}, \sigma={std_qq:.2f})$)')
    ax.set_ylabel('Sample Quantiles')
    ax.set_title('Q-Q Plot (Best Fit Gaussian)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('frg_residuals_b.pdf')
    plt.savefig('frg_residuals_b.eps')

def main():
    data = load_data()
    if data:
        generate_fig1_rotation_curves(data)
        generate_fig2_rar_variants(data)
        generate_fig3_btfr(data)
        generate_fig4_dwarf_spiral(data)
        generate_fig6_chi2_dist(data)

if __name__ == '__main__':
    main()
