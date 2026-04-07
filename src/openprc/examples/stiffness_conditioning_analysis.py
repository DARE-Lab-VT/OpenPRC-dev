"""
Stiffness Ratio vs. Conditioning Number Analysis
=================================================
This script performs an analysis on a 2D spring-mass system to understand
how the stiffness ratio of its components affects the conditioning of the
feature matrix (`X_full`) used in reservoir computing tasks.

It systematically runs a physics simulation of a 4x4 spring-mass grid,
varying the stiffness of one set of springs ("L") while keeping another
set ("H") constant. For each generated simulation, it computes the
condition number of the resulting node position feature matrix.

The output is a plot of the stiffness ratios vs. their corresponding
condition numbers, saved as an SVG file. This plot helps find the "knee point"
where the conditioning of the feature matrix drops significantly.
"""

import sys
import csv
from pathlib import Path
import numpy as np

from sklearn.preprocessing import StandardScaler

# --- Path Setup ---
current_dir = Path(__file__).parent
src_dir = current_dir.parent
sys.path.insert(0, str(src_dir))

# --- Core Library Imports ---
# Renaming for clarity in this script's context
from openprc.examples.spring_mass_2D import run_pipeline as run_spring_mass_simulation
from openprc.reservoir.io.state_loader import StateLoader
from openprc.reservoir.features.node_features import NodePositions, NodeDisplacements
from openprc.reservoir.features.bar_features import BarLengths

def create_k_matrix(h_stiffness, l_stiffness):
    """Creates the 4x4 stiffness matrix with specified H and L values."""
    H = h_stiffness
    L = l_stiffness
    # This matrix defines the connectivity and stiffness for a 4x4 grid.
    # 'H' represents high-stiffness springs, 'L' represents low-stiffness ones.
    return np.array([
        # 0   1   2   3   4   5   6   7   8   9  10  11  12  13  14  15
        [ 0,  L,  0,  0,  H,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0], # Node 0
        [ L,  0,  H,  0,  0,  H,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0], # Node 1
        [ 0,  H,  0,  H,  0,  0,  L,  0,  0,  0,  0,  0,  0,  0,  0,  0], # Node 2
        [ 0,  0,  H,  0,  0,  0,  0,  L,  0,  0,  0,  0,  0,  0,  0,  0], # Node 3
        [ H,  0,  0,  0,  0,  L,  0,  0,  L,  0,  0,  0,  0,  0,  0,  0], # Node 4
        [ 0,  H,  0,  0,  L,  0,  H,  0,  0,  L,  0,  0,  0,  0,  0,  0], # Node 5
        [ 0,  0,  L,  0,  0,  H,  0,  H,  0,  0,  H,  0,  0,  0,  0,  0], # Node 6
        [ 0,  0,  0,  L,  0,  0,  H,  0,  0,  0,  0,  H,  0,  0,  0,  0], # Node 7
        [ 0,  0,  0,  0,  L,  0,  0,  0,  0,  H,  0,  0,  L,  0,  0,  0], # Node 8
        [ 0,  0,  0,  0,  0,  L,  0,  0,  H,  0,  L,  0,  0,  L,  0,  0], # Node 9
        [ 0,  0,  0,  0,  0,  0,  H,  0,  0,  L,  0,  L,  0,  0,  H,  0], # Node 10
        [ 0,  0,  0,  0,  0,  0,  0,  H,  0,  0,  L,  0,  0,  0,  0,  H], # Node 11
        [ 0,  0,  0,  0,  0,  0,  0,  0,  L,  0,  0,  0,  0,  H,  0,  0], # Node 12
        [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  L,  0,  0,  H,  0,  L,  0], # Node 13
        [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  H,  0,  0,  L,  0,  L], # Node 14
        [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  H,  0,  0,  L,  0]  # Node 15
    ])

def main():
    """
    Main function to run the stiffness vs. conditioning analysis.
    """
    # --- 1. Analysis Parameters ---
    H_STIFFNESS = 222.15  # Keep H constant, as requested.
    # Sweep L from H down to 10 on a log scale for better sampling.
    L_STIFFNESS_VALUES = np.logspace(np.log10(H_STIFFNESS), np.log10(10), num=15) 

    stiffness_ratios = []
    condition_numbers = []
    
    # --- 2. Run the Parameter Sweep ---
    print("Starting stiffness sweep analysis...")
    print(f"Sweeping L-stiffness from {L_STIFFNESS_VALUES[0]:.2f} down to {L_STIFFNESS_VALUES[-1]:.2f}.")
    
    for i, l_stiff in enumerate(L_STIFFNESS_VALUES):
        ratio = H_STIFFNESS / l_stiff
        print(f"\n--- Iteration {i+1}/{len(L_STIFFNESS_VALUES)} ---")
        print(f"L-Stiffness: {l_stiff:.2f}, H/L Ratio: {ratio:.2f}")

        # --- a. Run Physics Simulation ---
        k_mat = create_k_matrix(H_STIFFNESS, l_stiff)
        # Use the iteration index as the 'generation' to keep outputs separate.
        _, output_dir = run_spring_mass_simulation(rows=4, cols=4, k_mat=k_mat, ga_generation=i)
        
        sim_path = output_dir / "output" / "simulation.h5"
        if not sim_path.exists():
            print(f"ERROR: Simulation failed to generate output for L={l_stiff:.2f}. Skipping.")
            continue

        # --- b. Load Data and Calculate Condition Number ---
        loader = StateLoader(sim_path)
        
        # We use NodePositions as the feature set. You can swap this with other
        # feature extractors like NodeDisplacements to see how it affects results.
        # features = NodeDisplacements(reference_node=0, dims=[0])
        features = NodeDisplacements(reference_node=0, dims=[0])
        X_full = features.transform(loader)
        print(X_full.shape)
        scaler_X = StandardScaler()
        X_std = scaler_X.fit_transform(X_full)
        
        rank = np.linalg.matrix_rank(X_std)
        cond_num = np.linalg.cond(X_std)

        print(f"Rank of X_std: {rank:.4e}")    
        print(f"Condition number of X_std: {cond_num:.4e}")
        
        stiffness_ratios.append(ratio)
        condition_numbers.append(cond_num)

    # --- 3. Plot and Save Results ---
    print("\nSweep complete. Generating plot...")

    try:
        import matplotlib.pyplot as plt
        
        output_svg_path = current_dir / "stiffness_conditioning_plot.svg"

        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot data
        ax.plot(stiffness_ratios, condition_numbers, marker='o', linestyle='-')
        
        # Set titles and labels
        ax.set_title("Stiffness Ratio vs. Matrix Condition Number")
        ax.set_xlabel("Stiffness Ratio (H/L)")
        ax.set_ylabel("Condition Number of X_full")
        
        # Use a logarithmic scale for the y-axis
        ax.set_yscale('log')
        
        # Improve layout
        ax.grid(True, which="both", ls="--", c='0.7')
        fig.tight_layout()
        
        # Save the figure
        fig.savefig(output_svg_path)
        print(f"Plot saved successfully to: {output_svg_path}")

    except ImportError:
        print("\n--- Plotting Skipped ---")
        print("Matplotlib is not installed. Cannot generate plot.")
        print("Please install it to get a plot: pip install matplotlib")
        
        # Fallback to saving a CSV file
        output_csv_path = current_dir / "stiffness_conditioning_results.csv"
        print(f"\nSaving raw data to CSV as a fallback: {output_csv_path}")

        with open(output_csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['stiffness_ratio_H_div_L', 'condition_number'])
            for i in range(len(stiffness_ratios)):
                writer.writerow([stiffness_ratios[i], condition_numbers[i]])
        print("CSV file saved.")

    print("\nAnalysis finished.")

if __name__ == "__main__":
    main()
