
import pickle
import numpy as np

def load_data():
    try:
        with open('external/gravity/active/scripts/sparc_master.pkl', 'rb') as f:
            data = pickle.load(f)
        return data
    except FileNotFoundError:
        print("Error: File not found")
        return {}

def compute_mad(alpha, galaxies):
    C_LAG = 0.094
    A0 = 1.2e-10
    residuals = []
    count = 0
    
    for name, g in galaxies.items():
        if not isinstance(g, dict) or 'Vobs' not in g:
            continue
            
        # Convert to numpy arrays explicitly
        v_obs = np.array(g['Vobs'])
        rad = np.array(g['Rad'])
        v_gas = np.array(g['Vgas'])
        v_disk = np.array(g['Vdisk'])
        v_bul = np.array(g['Vbul'])
        
        mask = (v_obs > 0) & (rad > 0)
        if np.sum(mask) < 3: continue
        
        r_m = rad[mask] * 3.086e19
        v_obs_ms = v_obs[mask] * 1000.0
        
        v_baryon_sq = (np.abs(v_gas[mask])*v_gas[mask] + 
                       np.abs(v_disk[mask])*v_disk[mask] + 
                       np.abs(v_bul[mask])*v_bul[mask]) * 1000**2
                       
        gb = v_baryon_sq / r_m
        go = (v_obs_ms**2) / r_m
        
        # Use a very lenient filter to see if we get ANY data
        valid = (gb > 0) & (go > 0)
        gb = gb[valid]
        go = go[valid]
        
        if len(gb) == 0: continue
        
        count += 1
        term = (gb / A0)
        
        # Handle term <= 0 if any (though gb > 0 check should prevent this)
        term = np.maximum(term, 1e-10)
        
        w = 1.0 + C_LAG * (term**(-alpha) - 1.0)
        pred_go = w * gb
        
        diff = np.log10(go) - np.log10(pred_go)
        residuals.extend(diff)
        
    if len(residuals) == 0:
        # print("No residuals collected")
        return 1e9
        
    return np.median(np.abs(residuals))

def main():
    data = load_data()
    # print(f"Loaded {len(data)} entries")
    
    alphas = np.linspace(0.15, 0.25, 21)
    best_mad = 1e9
    best_alpha = 0.0
    
    print(f"{'Alpha':<10} | {'MAD':<10}")
    print("-" * 25)
    
    for alpha in alphas:
        mad = compute_mad(alpha, data)
        print(f"{alpha:.4f}     | {mad:.6f}")
        if mad < best_mad:
            best_mad = mad
            best_alpha = alpha
            
    print("-" * 25)
    print(f"Best Alpha: {best_alpha:.4f} with MAD: {best_mad:.6f}")

if __name__ == "__main__":
    main()
