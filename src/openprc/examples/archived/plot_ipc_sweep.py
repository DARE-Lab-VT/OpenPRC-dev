import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from pathlib import Path

# --- Path Setup ---
current_dir = Path(__file__).parent
src_dir = current_dir.parent
# Make sure TRIAL_NAME matches what you used in the optimization script
TRIAL_NAME = "Taichi_Stiffness_Sweep_Damping" 
EXPERIMENT_DIR = src_dir / "experiments" / TRIAL_NAME

def plot_ipc_sweep(experiment_dir):
    if not experiment_dir.exists():
        print(f"[Error] Main sweep directory not found: {experiment_dir}")
        print("Ensure the sweep optimization has completed and TRIAL_NAME is correct.")
        return
        
    # 1. Find all stiffness subfolders
    sweep_folders = [d for d in experiment_dir.iterdir() if d.is_dir() and d.name.startswith("stiffness_")]
    
    if not sweep_folders:
        print(f"[Error] No 'stiffness_*' subfolders found in {experiment_dir.name}.")
        return

    # 2. Extract Data from CSVs
    sweep_data = []
    print(f"-> Scanning {len(sweep_folders)} sweep folders for loss_log.csv...")
    for folder in sweep_folders:
        try:
            # Extract stiffness value from the folder name
            stiffness = float(folder.name.split('_')[1])
            csv_path = folder / "loss_log.csv"
            
            if csv_path.exists():
                df = pd.read_csv(csv_path)
                if 'IPC_Capacity' in df.columns and 'Epoch' in df.columns:
                    sweep_data.append((stiffness, df))
                else:
                    print(f"  [Warning] Missing required columns in {folder.name}/loss_log.csv")
            else:
                print(f"  [Warning] No loss_log.csv found in {folder.name}")
        except Exception as e:
            print(f"  [Error] Failed to process {folder.name}: {e}")

    if not sweep_data:
        print("[Error] No valid data found to plot.")
        return

    # 3. Sort data by stiffness from lowest to highest
    sweep_data.sort(key=lambda x: x[0])
    stiffness_values = [data[0] for data in sweep_data]

    # 4. Setup Plot and Colormap
    fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
    
    # Use a colormap (e.g., 'viridis', 'plasma', or 'coolwarm')
    cmap = cm.get_cmap('viridis')
    # Normalize the colormap to the min and max stiffness values
    norm = mcolors.Normalize(vmin=min(stiffness_values), vmax=max(stiffness_values))

    # 5. Plot each line
    for stiffness, df in sweep_data:
        color = cmap(norm(stiffness))
        ax.plot(
            df['Epoch'], 
            df['IPC_Capacity'], 
            linewidth=2.5, 
            color=color, 
            label=f'$k = {stiffness}$ N/m',
            alpha=0.9
        )

    # 6. Formatting
    ax.set_title('IPC or sum of R2 across all basis', fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel('Optimization Epoch', fontsize=12)
    #ax.set_ylabel('Total IPC Capacity (Sum of valid $R^2$)', fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.6)
    
    # Add Legend outside the plot area
    ax.legend(title='Base Stiffness', bbox_to_anchor=(1.02, 1), loc='upper left', frameon=True)
    
    # Add a colorbar to visualize the continuous gradient
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([]) # Required for matplotlib
    #cbar = fig.colorbar(sm, ax=ax, pad=0.15)
    #cbar.set_label('Stiffness (N/m)', rotation=270, labelpad=15)

    plt.tight_layout()
    
    # 7. Save Plot
    save_path = experiment_dir / "ipc_stiffness_sweep.png"
    plt.savefig(save_path, bbox_inches='tight')
    plt.show()
    
    print(f"\n[Success] Sweep plot saved to: {save_path}")

if __name__ == "__main__":
    plot_ipc_sweep(EXPERIMENT_DIR)