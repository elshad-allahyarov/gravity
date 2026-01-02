#!/bin/bash
# Reproducibility Script for Causal-Response Gravity Paper

set -e  # Exit on error

echo "Step 1: Filtering Q=1 Data..."
python filter_q1.py

echo "Step 2: Optimizing Global Parameters..."
# Note: optimize_params.py runs Differential Evolution. It might take time.
# For exact reproducibility, one should set a random seed, but here we run it to verify convergence.
python optimize_params.py

echo "Step 3: Generating Main Figures (Q=1)..."
python generate_figures.py

echo "Step 4: Generating Q=1+Q=2 Validation Figures..."
python generate_fig1_q1q2.py
python generate_fig_rar_q1q2.py
python generate_fig_btfr_q1q2.py
python generate_fig_dwarf_spiral_q1q2.py
python generate_fig_residuals_q1q2.py
python generate_fig_chi2_dist_q1q2.py
python generate_fig_sensitivity_q1q2.py

echo "Step 5: Generating Diagnostic and Falsification Figures..."
python generate_fig_res_vs_fgas.py
python generate_fig_transition_radius.py
python generate_fig_weight_profiles.py

echo "Step 6: Calculating Statistics and Timescales..."
python calculate_stats.py

echo "All results reproduced successfully."

