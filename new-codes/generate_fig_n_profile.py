#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

# Fitted parameters from the manuscript
A = 1.06
r0 = 17.79  # kpc
p  = 0.95

def n_of_r(r):
    return 1.0 + A * (1.0 - np.exp(- (r / r0)**p))

def main():
    r = np.linspace(0.0, 60.0, 500)  # kpc
    n = n_of_r(r)

    plt.figure(figsize=(6.0, 4.0))
    plt.plot(r, n, 'k', lw=2, label='n(r)')
    plt.axhline(1.0 + A, color='gray', ls='--', lw=1.0, label='1 + A (asymptote)')
    plt.xlabel('r [kpc]')
    plt.ylabel('n(r)')
    plt.title('Spatial profile n(r) = 1 + A [1 - exp(-(r/r0)^p)]')
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig('frg_n_profile.pdf')
    plt.savefig('frg_n_profile.eps')
    plt.close()

if __name__ == '__main__':
    main()





