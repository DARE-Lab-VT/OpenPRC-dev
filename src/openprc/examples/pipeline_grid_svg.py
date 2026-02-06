"""
Grid SVG Pipeline
=================
1. Generates an SVG grid with fixed corners.
2. Imports the SVG to create a Demlat experiment.
3. Runs the simulation with gravity.
"""
import os
import sys
from pathlib import Path
import numpy as np

# Ensure we can import demlat modules
current_dir = Path(__file__).parent
src_dir = current_dir.parent
sys.path.insert(0, str(src_dir))

import demlat
from demlat.models.barhinge import BarHingeModel
from demlat.utils.svg_importer import parse_svg_experiment, save_experiment

def create_grid_svg_manual(filename, rows=10, cols=10, spacing=20):
    """
    Generates a simple SVG grid.
    - Corners are Black (Fixed).
    - Others are Blue (Free).
    - Lines are Gray (Springs).
    """
    lines = []
    lines.append('<svg xmlns="http://www.w3.org/2000/svg" version="1.1" width="500" height="500">')
    
    # Horizontal Lines
    for r in range(rows):
        for c in range(cols - 1):
            x1, y1 = c * spacing + 50, r * spacing + 50
            x2, y2 = (c + 1) * spacing + 50, r * spacing + 50
            lines.append(f'  <line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="#808080" stroke-width="2" />')

    # Vertical Lines
    for r in range(rows - 1):
        for c in range(cols):
            x1, y1 = c * spacing + 50, r * spacing + 50
            x2, y2 = c * spacing + 50, (r + 1) * spacing + 50
            lines.append(f'  <line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="#808080" stroke-width="2" />')

    # Nodes (Draw on top of lines)
    fixed_count = 0
    for r in range(rows):
        for c in range(cols):
            x = c * spacing + 50
            y = r * spacing + 50
            
            # Corners fixed (Black)
            if (r == 0 or r == rows-1) and (c == 0 or c == cols-1):
                color = "#000000"
                fixed_count += 1
            # Center actuated (Red)
            elif r == rows//2 and c == cols//2:
                color = "#ff0000"
            else:
                color = "#0000ff" # Blue (Free)
            
            lines.append(f'  <circle cx="{x}" cy="{y}" r="4" fill="{color}" />')

    lines.append('</svg>')
    
    with open(filename, 'w') as f:
        f.write('\n'.join(lines))
    print(f"SVG saved to {filename}. Fixed nodes: {fixed_count}")

def run_pipeline():
    # Setup paths
    svg_file = current_dir / "grid_example.svg"
    output_dir = src_dir / "experiments" / "grid_svg_test"
    
    # 1. Generate SVG
    print("\n[Step 1] Generating SVG Grid...")
    create_grid_svg_manual(svg_file)
    
    # 2. Import
    print(f"\n[Step 2] Importing SVG to {output_dir}...")
    # Scale down to reasonable physics units (e.g., 1 pixel = 1mm -> 0.001m)
    data = parse_svg_experiment(str(svg_file), str(output_dir), scale=0.001)
    save_experiment(data, str(output_dir))
    
    # 3. Run Simulation
    print("\n[Step 3] Running Simulation...")
    exp = demlat.Experiment(output_dir)
    
    # Use CUDA if available, else CPU
    try:
        import pycuda.driver
        backend = 'cuda'
        print("Using CUDA backend.")
    except ImportError:
        backend = 'cpu'
        print("Using CPU backend.")
        
    eng = demlat.Engine(BarHingeModel, backend=backend)
    eng.run(exp)
    
    print("\nDone. You can now run the simulation player:")
    print(f"python src/demlat/utils/viz_player.py {output_dir}")

if __name__ == "__main__":
    run_pipeline()
