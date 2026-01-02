0-0-0-0-0-0-paper/
====================

This directory contains the complete reproduction package for the manuscript.

Structure:
----------

1. paper/
   - Contains the main LaTeX source file (.tex), the compiled PDF, and the bibliography file (.bbl).
   - Main file: 0-0-0-gravity-submission-aaa-v07-shorted-v04.tex

2. figures/
   - Contains all figure files (EPS and PDF) referenced in the manuscript.
   - These are the exact files used by the \includegraphics commands in the TeX file.

3. code/
   - figures-10/: Contains the Python scripts used to generate the plots and analysis for the paper.
     - Key script: generate_figures.py (or reproduce_results.sh)
     - See FIGURE_GENERATION_GUIDE.txt inside for details.
   - root-scripts/: Helper utility scripts used during the drafting process.

4. solver/
   - external-gravity-active-scripts/: Contains the core solver/fitting codes used for the theoretical modeling.
   - Includes Rotmod_LTG (rotation curve modeling tool) if applicable.

5. data/
   - Contains the data artifacts (PKL, CSV) required by the scripts to run and reproduce the plots.
   - external-gravity/active/scripts/: Input/Output pickle files for the solver.
   - external-gravity/active/results/: Result files from the optimization runs.
   - external-gravity/results/: Benchmark summary CSVs (SPARC/MOND comparisons).

Instructions:
-------------
- To compile the paper: Run pdflatex (or latexmk) on the .tex file in the paper/ directory. Ensure the figures/ directory is accessible (the TeX file assumes figures are in the search path or local).
- To reproduce plots: Navigate to code/figures-10/ and run the relevant scripts (ensure Python environment is set up with requirements.txt).

