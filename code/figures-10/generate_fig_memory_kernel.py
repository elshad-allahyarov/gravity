#!/usr/bin/env python3
"""
Generate schematic figure of memory kernel Gamma(tau)
"""
import matplotlib.pyplot as plt
import numpy as np

plt.style.use('default')
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 11

def main():
    # Parameters
    w_typical = 1.5  # Typical enhancement
    tau_star_myr = 133  # From paper
    
    # Time axis (in Myr)
    tau = np.linspace(0, 500, 1000)
    
    # Memory kernel: Gamma(tau) = (w-1)/tau_star * exp(-tau/tau_star)
    Gamma = (w_typical - 1) / tau_star_myr * np.exp(-tau / tau_star_myr)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Panel 1: Memory Kernel
    ax1.plot(tau, Gamma, 'b-', lw=2.5)
    ax1.axvline(tau_star_myr, color='r', ls='--', lw=1.5, label=r'$\tau_\star = 133$ Myr')
    ax1.fill_between(tau, 0, Gamma, alpha=0.2, color='blue')
    ax1.set_xlabel(r'Time Lag $\tau$ (Myr)', fontsize=12)
    ax1.set_ylabel(r'Memory Kernel $\Gamma(\tau)$ (Myr$^{-1}$)', fontsize=12)
    ax1.set_title('Exponential Memory Kernel', fontsize=13, fontweight='bold')
    ax1.legend(loc='upper right', fontsize=10)
    ax1.grid(alpha=0.3, ls=':')
    ax1.set_xlim(0, 500)
    ax1.text(250, 0.003, r'$\Gamma(\tau) = \frac{w-1}{\tau_\star} e^{-\tau/\tau_\star}$',
             fontsize=12, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Panel 2: Transfer Function H(i omega)
    # omega in units of 1/tau_star
    omega_norm = np.logspace(-2, 2, 500)
    
    # H(i omega) = 1 + (w-1)/(1 + i omega tau_star)
    # Real part: C(omega) = 1 + (w-1)/(1 + omega^2 tau^2)
    C_omega = 1 + (w_typical - 1) / (1 + omega_norm**2)
    
    # Imaginary part: S(omega) = -(w-1) omega tau / (1 + omega^2 tau^2)
    S_omega = -(w_typical - 1) * omega_norm / (1 + omega_norm**2)
    
    ax2.semilogx(omega_norm, C_omega, 'b-', lw=2.5, label=r'$C(\omega)$ (Real, Conservative)')
    ax2.semilogx(omega_norm, S_omega, 'r--', lw=2.5, label=r'$S(\omega)$ (Imaginary, Dissipative)')
    ax2.axhline(1.0, color='k', ls=':', lw=1, alpha=0.5)
    ax2.axhline(w_typical, color='b', ls=':', lw=1, alpha=0.5, label=f'$w = {w_typical}$ (low-freq limit)')
    ax2.axvline(1.0, color='gray', ls='--', lw=1, alpha=0.5)
    ax2.set_xlabel(r'Frequency $\omega \tau_\star$ (dimensionless)', fontsize=12)
    ax2.set_ylabel(r'Transfer Function Components', fontsize=12)
    ax2.set_title('Frequency-Domain Response', fontsize=13, fontweight='bold')
    ax2.legend(loc='right', fontsize=9)
    ax2.grid(alpha=0.3, ls=':', which='both')
    ax2.set_xlim(0.01, 100)
    ax2.set_ylim(-0.3, 1.6)
    ax2.text(0.02, 1.45, r'Fast limit: $H \to 1$ (Newtonian)', fontsize=10,
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
    ax2.text(0.02, 0.2, r'Slow limit: $C \to w$ (Enhanced)', fontsize=10,
             bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))
    
    plt.tight_layout()
    plt.savefig('frg_memory_kernel.pdf', dpi=300, bbox_inches='tight')
    plt.savefig('frg_memory_kernel.eps', dpi=300, bbox_inches='tight')
    print("Saved frg_memory_kernel.pdf and .eps")
    plt.close()

if __name__ == '__main__':
    main()

