import h5py
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

class PlotResult:
    """
    Holds the figure and saves it to the 'metrics/' directory alongside metrics.h5.
    """
    def __init__(self, figure, source_path, title):
        self.figure = figure
        # source_path is the path to readout.h5 (e.g., .../readout/readout.h5)
        self.source_path = Path(source_path)
        self.title = title

    def save(self):
        # 1. Navigate from 'readout/' folder to experiment root
        experiment_root = self.source_path.parent.parent
        
        # 2. Target the 'metrics/' directory
        metrics_dir = experiment_root / "metrics"
        metrics_dir.mkdir(parents=True, exist_ok=True)
        
        # 3. Define filename
        filename = self.title.lower().replace(" ", "_") + ".svg"
        output_path = metrics_dir / filename
        
        print(f"Saving Plot to: {output_path}")
        self.figure.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close(self.figure)
        
        return output_path

class TimeSeriesComparison:
    """
    Generates Target vs. Prediction plots.
    """
    def plot(self, readout_path, start_frame=0, end_frame=200):
        readout_path = Path(readout_path)
        
        with h5py.File(readout_path, 'r') as f:
            # Prefer validation set
            group_name = 'validation' if 'validation' in f else 'training'
            target = f[f'{group_name}/target'][start_frame:end_frame]
            prediction = f[f'{group_name}/prediction'][start_frame:end_frame]
            task_name = f.attrs.get('task_name', 'Unknown Task')

        # Create Figure
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(target, label='Target', color='green', linestyle='--')
        ax.plot(prediction, label='Prediction', color='red')
        
        ax.set_title(f"Task Performance: {task_name} ({group_name} set)")
        ax.set_xlabel("Time Steps")
        ax.set_ylabel("Signal Value")
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        
        return PlotResult(fig, readout_path, title=f"{task_name}_comparison")