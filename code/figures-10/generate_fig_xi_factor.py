#!/usr/bin/env python3
"""
Generate figure showing xi(f_gas) for all galaxies
"""
import matplotlib.pyplot as plt
import numpy as np
import pickle

plt.style.use('default')
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 11

# Parameters
C_LAG = 0.2981

def xi_global(f_gas): return 1.0 + C_LAG * np.sqrt(np.maximum(f_gas, 0.0))

def get_galaxy_type(v_max):
    if v_max < 80: return 'Dwarf'
    elif v_max <= 200: return 'Spiral'
    else: return 'Massive'

def main():
    with open('external/gravity/active/scripts/sparc_q1.pkl', 'rb') as f:
        data = pickle.load(f)
    
    f_gas_list = []
    xi_list = []
    types = []
    
    for name, g in data.items():
        f_gas = g.get('f_gas_true', 0.0)
        xi = xi_global(f_gas)
        v_max = np.nanmax(g['data']['vobs'].values)
        gtype = get_galaxy_type(v_max)
        
        f_gas_list.append(f_gas)
        xi_list.append(xi)
        types.append(gtype)
    
    # Plot
    fig, ax = plt.subplots(figsize=(7, 5))
    
    colors = {'Dwarf': 'blue', 'Spiral': 'green', 'Massive': 'red'}
    for gtype in ['Dwarf', 'Spiral', 'Massive']:
        mask = np.array(types) == gtype
        ax.scatter(np.array(f_gas_list)[mask], np.array(xi_list)[mask],
                  c=colors[gtype], label=gtype, alpha=0.6, s=40, edgecolors='k', linewidths=0.5)
    
    # Theoretical curve
    f_gas_theory = np.linspace(0, 1, 100)
    xi_theory = xi_global(f_gas_theory)
    ax.plot(f_gas_theory, xi_theory, 'k--', lw=2, label=r'$\xi = 1 + C_\xi \sqrt{f_{\rm gas}}$')
    
    ax.set_xlabel(r'Gas Fraction $f_{\rm gas}$', fontsize=12)
    ax.set_ylabel(r'Complexity Factor $\xi$', fontsize=12)
    ax.legend(loc='lower right', frameon=True, fontsize=10)
    ax.grid(alpha=0.3, ls=':')
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(0.95, 1.35)
    
    plt.tight_layout()
    plt.savefig('frg_xi_factor.pdf', dpi=300, bbox_inches='tight')
    plt.savefig('frg_xi_factor.eps', dpi=300, bbox_inches='tight')
    print("Saved frg_xi_factor.pdf and .eps")
    plt.close()

if __name__ == '__main__':
    main()

