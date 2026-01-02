
import pickle
import numpy as np
from scipy.optimize import minimize

def load_data():
    with open('external/gravity/active/scripts/sparc_master.pkl', 'rb') as f:
        data = pickle.load(f)
    return data

def compute_rar_alpha(alpha_arr, galaxies):
    alpha = alpha_arr[0]
    C_LAG = 0.094
    A0 = 1.2e-10
    
    residuals = []
    
    for name, g in galaxies.items():
        # Q=1 check is good practice but let's use what we have in master table
        if g.get('Q') != 1: continue
        
        # Extract data
        inc = g['inc']
        if inc < 30: continue # Skip face-on
        
        v_obs = g['Vobs']
        v_gas = g['Vgas']
        v_disk = g['Vdisk']
        v_bul = g['Vbul']
        r_kpc = g['Rad']
        
        # Basic masking
        mask = (v_obs > 0) & (r_kpc > 0)
        if np.sum(mask) < 3: continue
        
        r_m = r_kpc[mask] * 3.086e19
        v_obs_ms = v_obs[mask] * 1000.0
        
        # Baryonic acceleration (Newtonian)
        v_baryon_sq = np.abs(v_gas[mask]) * v_gas[mask] + \
                      np.abs(v_disk[mask]) * v_disk[mask] + \
                      np.abs(v_bul[mask]) * v_bul[mask] 
        # Ensure non-negative for sqrt if needed, but we work with a_baryon directly
        # a_baryon = v_baryon_sq / r
        
        gb = v_baryon_sq / r_m * 1000**2 # Convert (km/s)^2/kpc to m/s^2? No.
        # Units: v in km/s, r in kpc.
        # a = v^2/r. (km/s)^2 / kpc = (10^3 m/s)^2 / (3.086e19 m) = 10^6 / 3e19 ~ 3e-14 m/s^2
        # Wait, let's stick to SI if possible or consistent units.
        
        # Using consistent SI calc from generate_figures:
        # gb = (v_bar_ms**2) / r_m
        v_baryon_ms_sq = v_baryon_sq * (1000**2)
        gb = v_baryon_ms_sq / r_m
        go = (v_obs_ms**2) / r_m
        
        # Filter very low accel where noise dominates
        valid = (gb > 1e-13) & (go > 1e-13)
        if np.sum(valid) == 0: continue
        
        gb = gb[valid]
        go = go[valid]
        
        # Compute prediction for this alpha
        # w = 1 + C_LAG * ( (gb/A0)^-alpha - 1 )
        # But we need to be careful with the formula.
        # w = 1 + C_LAG * (sqrt(u_b) is in C_xi, but here we use average C_LAG which includes C_xi effect?)
        # The paper says: w = xi * n(r) * (T/tau)^alpha * zeta
        # generate_figures uses: w_pred = 1.0 + 0.094 * ( (a_pred / a0)**(-alpha_val) - 1.0 )
        # This is a simplified effective RAR relation. Let's stick to this form for the RAR plot optimization.
        
        term = (gb / A0)
        # w = 1.0 + C_LAG * (term**(-alpha) - 1.0)
        # prediction: a_eff = w * a_baryon
        
        # The model in the RAR plot (line 453 in generate_figures) is:
        w = 1.0 + C_LAG * (term**(-alpha) - 1.0)
        pred_go = w * gb
        
        # Log space residuals
        diff = np.log10(go) - np.log10(pred_go)
        residuals.extend(diff)

    if not residuals:
        return 1e9

    # Robust objective: Median Absolute Deviation (MAD) of residuals
    # We want the line to go through the "middle" of the scatter.
    return np.median(np.abs(residuals))

def main():
    data = load_data()
    
    # Initial guess
    x0 = [0.19]
    bounds = [(0.10, 0.25)]
    
    res = minimize(
        compute_rar_alpha, 
        x0, 
        args=(data,), 
        method='Nelder-Mead',
        bounds=bounds
    )
    
    print(f"Optimal visual alpha for RAR scatter (MAD): {res.x[0]:.4f}")

if __name__ == "__main__":
    main()
