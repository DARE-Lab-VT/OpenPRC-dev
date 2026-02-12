# demlat/utils/data_parser.py
"""
DEMLAT Time Series Plotter
==========================

A flexible utility for plotting any time series data from simulation HDF5 files.

Usage:
------
    # Command line
    python -m demlat.utils.plot_timeseries experiments/test/output/simulation.h5 --plot nodes/positions
    python -m demlat.utils.plot_timeseries simulation.h5 --plot system/kinetic_energy elements/bars/strain
    python -m demlat.utils.plot_timeseries simulation.h5 --list
    python -m demlat.utils.plot_timeseries simulation.h5 --plot nodes/positions --indices 0,1,2 --components z

    # Python API
    from demlat.utils.plot_timeseries import SimulationData

    plotter = SimulationData("simulation.h5")
    plotter.list_datasets()
    plotter.plot("nodes/positions", indices=[0, 1, 2], components=['z'])
    plotter.plot_energy_summary()

Author: Yogesh Phalak
"""

import numpy as np
import h5py
from pathlib import Path
from typing import Optional, List, Union, Tuple
import argparse


class SimulationData:
    """
    Flexible plotter for DEMLAT simulation HDF5 files.
    """

    def __init__(self, filepath: str):
        """
        Initialize plotter with simulation file.

        Parameters
        ----------
        filepath : str
            Path to simulation.h5 file
        """
        self.filepath = Path(filepath)
        if not self.filepath.exists():
            raise FileNotFoundError(f"Simulation file not found: {self.filepath}")

        self.datasets = {}
        self.metadata = {}
        self.time = None

        self._scan_file()

    def _scan_file(self):
        """Scan HDF5 file and catalog all datasets."""
        print(f"\n{'=' * 60}")
        print(f"Scanning: {self.filepath.name}")
        print('=' * 60)

        with h5py.File(self.filepath, 'r') as f:
            # Load metadata
            for key, val in f.attrs.items():
                self.metadata[key] = val
                print(f"  [Attr] {key}: {val}")

            # Load time array
            if 'time_series/time' in f:
                self.time = f['time_series/time'][:]
                print(f"\n  [Time] {len(self.time)} frames, "
                      f"{self.time[0]:.4f}s to {self.time[-1]:.4f}s")

            # Recursively scan datasets
            def scan_group(group, prefix=""):
                for key in group.keys():
                    path = f"{prefix}/{key}" if prefix else key
                    item = group[key]

                    if isinstance(item, h5py.Dataset):
                        shape = item.shape
                        dtype = item.dtype
                        self.datasets[path] = {
                            'shape': shape,
                            'dtype': str(dtype),
                            'ndim': len(shape)
                        }
                    elif isinstance(item, h5py.Group):
                        scan_group(item, path)

            if 'time_series' in f:
                scan_group(f['time_series'], 'time_series')

        print(f"\n  Found {len(self.datasets)} datasets")

    def list_datasets(self, filter_pattern: Optional[str] = None):
        """
        List all available datasets.

        Parameters
        ----------
        filter_pattern : str, optional
            Filter datasets by substring match
        """
        print(f"\n{'=' * 60}")
        print("Available Datasets")
        print('=' * 60)

        # Group by category
        categories = {}
        for path, info in self.datasets.items():
            if filter_pattern and filter_pattern.lower() not in path.lower():
                continue

            parts = path.split('/')
            if len(parts) >= 2:
                category = '/'.join(parts[:2])
            else:
                category = 'root'

            if category not in categories:
                categories[category] = []
            categories[category].append((path, info))

        for category, items in sorted(categories.items()):
            print(f"\n[{category}]")
            for path, info in items:
                short_path = path.replace('time_series/', '')
                shape_str = str(info['shape'])
                print(f"  {short_path:40s} {shape_str:20s} {info['dtype']}")

        print(f"\n{'=' * 60}")
        print("Shape Legend:")
        print("  [T]       = time series (1D)")
        print("  [T, N]    = per-node/element time series")
        print("  [T, N, 3] = per-node 3D vectors over time")
        print('=' * 60)

    def get_dataset(self, path: str) -> Tuple[np.ndarray, dict]:
        """
        Load a dataset from file.

        Parameters
        ----------
        path : str
            Dataset path (can omit 'time_series/' prefix)

        Returns
        -------
        data : np.ndarray
        info : dict
        """
        # Normalize path
        if not path.startswith('time_series/'):
            full_path = f'time_series/{path}'
        else:
            full_path = path

        if full_path not in self.datasets:
            # Try without prefix
            available = [p.replace('time_series/', '') for p in self.datasets.keys()]
            raise ValueError(f"Dataset '{path}' not found. Available: {available}")

        with h5py.File(self.filepath, 'r') as f:
            data = f[full_path][:]

        return data, self.datasets[full_path]

    def plot(self,
             path: str,
             indices: Optional[List[int]] = None,
             components: Optional[List[str]] = None,
             time_range: Optional[Tuple[float, float]] = None,
             ax=None,
             show: bool = True,
             title: Optional[str] = None,
             ylabel: Optional[str] = None,
             legend: bool = True,
             alpha: float = 0.8,
             **plot_kwargs):
        """
        Plot a time series dataset.

        Parameters
        ----------
        path : str
            Dataset path (e.g., 'nodes/positions', 'system/kinetic_energy')
        indices : list of int, optional
            Which indices to plot for [T, N] or [T, N, 3] data.
            Default: first 5 or all if fewer
        components : list of str, optional
            For 3D data: ['x'], ['y'], ['z'], ['x', 'y'], ['magnitude'], etc.
            Default: all components
        time_range : tuple (t_start, t_end), optional
            Time range to plot
        ax : matplotlib axis, optional
            Axis to plot on
        show : bool
            Whether to call plt.show()
        title : str, optional
            Plot title (auto-generated if None)
        ylabel : str, optional
            Y-axis label (auto-generated if None)
        legend : bool
            Whether to show legend
        alpha : float
            Line transparency
        **plot_kwargs
            Additional arguments passed to plt.plot()

        Returns
        -------
        fig, ax : matplotlib figure and axis
        """
        import matplotlib.pyplot as plt

        data, info = self.get_dataset(path)
        ndim = info['ndim']
        shape = info['shape']

        # Time masking
        t = self.time
        t_mask = np.ones(len(t), dtype=bool)
        if time_range is not None:
            t_mask = (t >= time_range[0]) & (t <= time_range[1])
        t_plot = t[t_mask]

        # Create figure if needed
        if ax is None:
            fig, ax = plt.subplots(figsize=(12, 6))
        else:
            fig = ax.figure

        # Auto-generate title
        if title is None:
            title = path.replace('time_series/', '').replace('/', ' / ').title()

        # Component mapping for 3D data
        comp_map = {'x': 0, 'y': 1, 'z': 2}

        # Plot based on dimensionality
        if ndim == 1:
            # [T] - Simple time series
            ax.plot(t_plot, data[t_mask], alpha=alpha, **plot_kwargs)
            if ylabel is None:
                ylabel = path.split('/')[-1].replace('_', ' ').title()

        elif ndim == 2:
            # [T, N] - Per-element time series
            n_elements = shape[1]

            if indices is None:
                indices = list(range(min(5, n_elements)))

            for idx in indices:
                if idx >= n_elements:
                    print(f"Warning: Index {idx} out of range (max {n_elements - 1})")
                    continue
                ax.plot(t_plot, data[t_mask, idx],
                        label=f'#{idx}', alpha=alpha, **plot_kwargs)

            if ylabel is None:
                ylabel = path.split('/')[-1].replace('_', ' ').title()

        elif ndim == 3:
            # [T, N, 3] - Per-node 3D vectors
            n_elements = shape[1]

            if indices is None:
                indices = list(range(min(3, n_elements)))

            if components is None:
                components = ['x', 'y', 'z']

            for idx in indices:
                if idx >= n_elements:
                    print(f"Warning: Index {idx} out of range (max {n_elements - 1})")
                    continue

                for comp in components:
                    if comp.lower() == 'magnitude' or comp.lower() == 'mag':
                        values = np.linalg.norm(data[t_mask, idx, :], axis=1)
                        label = f'#{idx} |mag|'
                    elif comp.lower() in comp_map:
                        c_idx = comp_map[comp.lower()]
                        values = data[t_mask, idx, c_idx]
                        label = f'#{idx} {comp.upper()}'
                    else:
                        print(f"Unknown component: {comp}")
                        continue

                    ax.plot(t_plot, values, label=label, alpha=alpha, **plot_kwargs)

            if ylabel is None:
                ylabel = path.split('/')[-1].replace('_', ' ').title()

        else:
            raise ValueError(f"Unsupported data dimensionality: {ndim}")

        ax.set_xlabel('Time (s)')
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, alpha=0.3)

        if legend and ax.get_legend_handles_labels()[0]:
            ax.legend(loc='best', fontsize=8, ncol=min(4, len(indices) if indices else 1))

        fig.tight_layout()

        if show:
            plt.show()

        return fig, ax

    def plot_multi(self,
                   paths: List[str],
                   indices: Optional[List[int]] = None,
                   components: Optional[List[str]] = None,
                   time_range: Optional[Tuple[float, float]] = None,
                   share_x: bool = True,
                   figsize: Optional[Tuple[float, float]] = None,
                   show: bool = True):
        """
        Plot multiple datasets in subplots.

        Parameters
        ----------
        paths : list of str
            Dataset paths to plot
        indices, components, time_range
            Same as plot()
        share_x : bool
            Whether to share x-axis
        figsize : tuple, optional
            Figure size
        show : bool
            Whether to call plt.show()

        Returns
        -------
        fig, axes : matplotlib figure and axes
        """
        import matplotlib.pyplot as plt

        n_plots = len(paths)
        if figsize is None:
            figsize = (12, 3 * n_plots)

        fig, axes = plt.subplots(n_plots, 1, figsize=figsize, sharex=share_x)
        if n_plots == 1:
            axes = [axes]

        for ax, path in zip(axes, paths):
            try:
                self.plot(path, indices=indices, components=components,
                          time_range=time_range, ax=ax, show=False, legend=True)
            except Exception as e:
                ax.text(0.5, 0.5, f"Error loading {path}:\n{e}",
                        ha='center', va='center', transform=ax.transAxes)
                ax.set_title(path)

        fig.tight_layout()

        if show:
            plt.show()

        return fig, axes

    def plot_energy_summary(self,
                            time_range: Optional[Tuple[float, float]] = None,
                            show: bool = True):
        """
        Plot energy summary (KE, PE, Total, Damping Loss).

        Parameters
        ----------
        time_range : tuple, optional
            Time range to plot
        show : bool
            Whether to call plt.show()

        Returns
        -------
        fig, axes
        """
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 2, figsize=(14, 8))

        energy_datasets = [
            ('system/kinetic_energy', 'Kinetic Energy', axes[0, 0]),
            ('system/potential_energy', 'Potential Energy', axes[0, 1]),
            ('system/damping_loss', 'Cumulative Damping Loss', axes[1, 0]),
        ]

        t = self.time
        t_mask = np.ones(len(t), dtype=bool)
        if time_range is not None:
            t_mask = (t >= time_range[0]) & (t <= time_range[1])
        t_plot = t[t_mask]

        total_energy = np.zeros(np.sum(t_mask))

        for path, title, ax in energy_datasets:
            try:
                data, _ = self.get_dataset(path)
                data_plot = data[t_mask]
                ax.plot(t_plot, data_plot, linewidth=1.5)
                ax.set_title(title)
                ax.set_xlabel('Time (s)')
                ax.set_ylabel('Energy (J)')
                ax.grid(True, alpha=0.3)

                if 'kinetic' in path or 'potential' in path:
                    total_energy += data_plot

            except Exception as e:
                ax.text(0.5, 0.5, f"Not available:\n{e}",
                        ha='center', va='center', transform=ax.transAxes)
                ax.set_title(title)

        # Total energy plot
        ax_total = axes[1, 1]
        ax_total.plot(t_plot, total_energy, linewidth=1.5, color='purple')
        ax_total.set_title('Total Mechanical Energy (KE + PE)')
        ax_total.set_xlabel('Time (s)')
        ax_total.set_ylabel('Energy (J)')
        ax_total.grid(True, alpha=0.3)

        fig.suptitle(f'Energy Summary: {self.filepath.name}', fontsize=12, fontweight='bold')
        fig.tight_layout()

        if show:
            plt.show()

        return fig, axes

    def plot_node_trajectory(self,
                             node_indices: List[int],
                             projection: str = '3d',
                             time_range: Optional[Tuple[float, float]] = None,
                             colorby: str = 'time',
                             show: bool = True):
        """
        Plot node trajectories in 2D or 3D space.

        Parameters
        ----------
        node_indices : list of int
            Which nodes to plot
        projection : str
            '3d', 'xy', 'xz', 'yz'
        time_range : tuple, optional
            Time range
        colorby : str
            'time', 'velocity', 'index'
        show : bool
            Whether to call plt.show()

        Returns
        -------
        fig, ax
        """
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D

        positions, _ = self.get_dataset('nodes/positions')

        t = self.time
        t_mask = np.ones(len(t), dtype=bool)
        if time_range is not None:
            t_mask = (t >= time_range[0]) & (t <= time_range[1])

        positions = positions[t_mask]
        t_plot = t[t_mask]

        if projection == '3d':
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection='3d')

            for idx in node_indices:
                x = positions[:, idx, 0]
                y = positions[:, idx, 1]
                z = positions[:, idx, 2]

                if colorby == 'time':
                    colors = t_plot
                    scatter = ax.scatter(x, y, z, c=colors, cmap='viridis',
                                         s=2, alpha=0.6, label=f'Node {idx}')
                else:
                    ax.plot(x, y, z, alpha=0.7, label=f'Node {idx}')

                # Mark start and end
                ax.scatter(*positions[0, idx], c='green', s=100, marker='o',
                           label='Start' if idx == node_indices[0] else '')
                ax.scatter(*positions[-1, idx], c='red', s=100, marker='x',
                           label='End' if idx == node_indices[0] else '')

            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')

            if colorby == 'time':
                plt.colorbar(scatter, ax=ax, label='Time (s)', shrink=0.6)

        else:
            fig, ax = plt.subplots(figsize=(10, 8))

            proj_map = {
                'xy': (0, 1, 'X', 'Y'),
                'xz': (0, 2, 'X', 'Z'),
                'yz': (1, 2, 'Y', 'Z'),
            }

            if projection not in proj_map:
                raise ValueError(f"Unknown projection: {projection}. Use '3d', 'xy', 'xz', 'yz'")

            i1, i2, label1, label2 = proj_map[projection]

            for idx in node_indices:
                p1 = positions[:, idx, i1]
                p2 = positions[:, idx, i2]

                if colorby == 'time':
                    scatter = ax.scatter(p1, p2, c=t_plot, cmap='viridis',
                                         s=2, alpha=0.6, label=f'Node {idx}')
                else:
                    ax.plot(p1, p2, alpha=0.7, label=f'Node {idx}')

                ax.scatter(p1[0], p2[0], c='green', s=100, marker='o')
                ax.scatter(p1[-1], p2[-1], c='red', s=100, marker='x')

            ax.set_xlabel(label1)
            ax.set_ylabel(label2)
            ax.set_aspect('equal', adjustable='box')

            if colorby == 'time':
                plt.colorbar(scatter, ax=ax, label='Time (s)')

        ax.set_title(f'Node Trajectories ({projection.upper()})')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)

        fig.tight_layout()

        if show:
            plt.show()

        return fig, ax

    def plot_element_stats(self,
                           path: str,
                           stats: List[str] = ['mean', 'std', 'min', 'max'],
                           time_range: Optional[Tuple[float, float]] = None,
                           show: bool = True):
        """
        Plot statistics over all elements (useful for strain, stress, etc.)

        Parameters
        ----------
        path : str
            Dataset path (e.g., 'elements/bars/strain')
        stats : list of str
            Statistics to compute: 'mean', 'std', 'min', 'max', 'median'
        time_range : tuple, optional
        show : bool

        Returns
        -------
        fig, ax
        """
        import matplotlib.pyplot as plt

        data, info = self.get_dataset(path)

        if info['ndim'] != 2:
            raise ValueError(f"Element stats requires [T, N] shaped data, got {info['shape']}")

        t = self.time
        t_mask = np.ones(len(t), dtype=bool)
        if time_range is not None:
            t_mask = (t >= time_range[0]) & (t <= time_range[1])
        t_plot = t[t_mask]
        data = data[t_mask]

        fig, ax = plt.subplots(figsize=(12, 6))

        stat_funcs = {
            'mean': lambda d: np.mean(d, axis=1),
            'std': lambda d: np.std(d, axis=1),
            'min': lambda d: np.min(d, axis=1),
            'max': lambda d: np.max(d, axis=1),
            'median': lambda d: np.median(d, axis=1),
        }

        colors = plt.cm.tab10(np.linspace(0, 1, len(stats)))

        for stat, color in zip(stats, colors):
            if stat not in stat_funcs:
                print(f"Unknown stat: {stat}")
                continue

            values = stat_funcs[stat](data)
            ax.plot(t_plot, values, label=stat.capitalize(), color=color, alpha=0.8)

        # Fill between min and max
        if 'min' in stats and 'max' in stats:
            min_vals = stat_funcs['min'](data)
            max_vals = stat_funcs['max'](data)
            ax.fill_between(t_plot, min_vals, max_vals, alpha=0.2, color='gray', label='Range')

        name = path.split('/')[-1].replace('_', ' ').title()
        ax.set_title(f'{name} Statistics Over All Elements')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel(name)
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)

        fig.tight_layout()

        if show:
            plt.show()

        return fig, ax

    def interactive_plot(self):
        """
        Launch interactive plotting interface.
        """
        import matplotlib.pyplot as plt
        from matplotlib.widgets import Button, CheckButtons, TextBox

        fig = plt.figure(figsize=(14, 8))

        # Main plot area
        ax_main = fig.add_axes([0.1, 0.25, 0.85, 0.7])

        # Dataset list area
        ax_list = fig.add_axes([0.02, 0.25, 0.06, 0.7])
        ax_list.axis('off')

        # Controls area
        ax_controls = fig.add_axes([0.1, 0.02, 0.85, 0.18])
        ax_controls.axis('off')

        # Get short dataset names
        short_names = [p.replace('time_series/', '') for p in self.datasets.keys()]

        # Current selection
        state = {'current': None, 'indices': [0], 'components': ['x', 'y', 'z']}

        # Dataset buttons
        for i, name in enumerate(short_names[:15]):  # Limit to 15
            ax_btn = fig.add_axes([0.02, 0.9 - i * 0.045, 0.06, 0.04])
            btn = Button(ax_btn, name[:12], hovercolor='lightblue')
            btn.on_clicked(lambda event, n=name: self._update_interactive(
                ax_main, n, state, fig))

        # Index input
        ax_idx = fig.add_axes([0.15, 0.12, 0.15, 0.05])
        txt_idx = TextBox(ax_idx, 'Indices:', initial='0,1,2')
        txt_idx.on_submit(lambda text: self._parse_indices(text, state, ax_main, fig))

        # Component checkboxes
        ax_comp = fig.add_axes([0.35, 0.05, 0.1, 0.15])
        check = CheckButtons(ax_comp, ['X', 'Y', 'Z', 'Mag'], [True, True, True, False])
        check.on_clicked(lambda label: self._toggle_component(label, state, ax_main, fig))

        plt.show()

    def _update_interactive(self, ax, path, state, fig):
        """Update interactive plot."""
        state['current'] = path
        ax.clear()
        try:
            self.plot(path, indices=state['indices'],
                      components=state['components'], ax=ax, show=False)
        except Exception as e:
            ax.text(0.5, 0.5, str(e), ha='center', va='center', transform=ax.transAxes)
        fig.canvas.draw_idle()

    def _parse_indices(self, text, state, ax, fig):
        """Parse index input."""
        try:
            state['indices'] = [int(x.strip()) for x in text.split(',')]
            if state['current']:
                self._update_interactive(ax, state['current'], state, fig)
        except:
            pass

    def _toggle_component(self, label, state, ax, fig):
        """Toggle component visibility."""
        comp_map = {'X': 'x', 'Y': 'y', 'Z': 'z', 'Mag': 'magnitude'}
        comp = comp_map.get(label)
        if comp in state['components']:
            state['components'].remove(comp)
        else:
            state['components'].append(comp)
        if state['current']:
            self._update_interactive(ax, state['current'], state, fig)


def main():
    """Command line interface."""
    parser = argparse.ArgumentParser(
        description='Plot time series from DEMLAT simulation HDF5 files',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # List available datasets
  python -m demlat.utils.plot_timeseries simulation.h5 --list
  
  # Plot system energy
  python -m demlat.utils.plot_timeseries simulation.h5 --plot system/kinetic_energy
  
  # Plot multiple datasets
  python -m demlat.utils.plot_timeseries simulation.h5 --plot system/kinetic_energy system/potential_energy
  
  # Plot node positions with specific indices and components
  python -m demlat.utils.plot_timeseries simulation.h5 --plot nodes/positions --indices 0,1,2 --components z
  
  # Plot energy summary
  python -m demlat.utils.plot_timeseries simulation.h5 --energy
  
  # Plot bar strain statistics
  python -m demlat.utils.plot_timeseries simulation.h5 --stats elements/bars/strain
  
  # Plot node trajectories
  python -m demlat.utils.plot_timeseries simulation.h5 --trajectory 0,1,2 --projection 3d
  
  # Interactive mode
  python -m demlat.utils.plot_timeseries simulation.h5 --interactive
        """
    )

    parser.add_argument('filepath', type=str, help='Path to simulation.h5 file')
    parser.add_argument('--list', '-l', action='store_true', help='List available datasets')
    parser.add_argument('--plot', '-p', nargs='+', help='Dataset path(s) to plot')
    parser.add_argument('--indices', '-i', type=str, default=None,
                        help='Comma-separated indices (e.g., "0,1,2")')
    parser.add_argument('--components', '-c', type=str, default=None,
                        help='Components for 3D data (e.g., "x,y,z" or "magnitude")')
    parser.add_argument('--time-range', '-t', type=str, default=None,
                        help='Time range as "start,end" (e.g., "0.0,5.0")')
    parser.add_argument('--energy', '-e', action='store_true', help='Plot energy summary')
    parser.add_argument('--stats', '-s', type=str, default=None,
                        help='Plot element statistics for given path')
    parser.add_argument('--trajectory', type=str, default=None,
                        help='Plot node trajectories (comma-separated indices)')
    parser.add_argument('--projection', type=str, default='3d',
                        help='Projection for trajectory plot (3d, xy, xz, yz)')
    parser.add_argument('--interactive', action='store_true', help='Launch interactive mode')
    parser.add_argument('--save', type=str, default=None, help='Save figure to file')
    parser.add_argument('--filter', '-f', type=str, default=None,
                        help='Filter datasets by pattern (with --list)')

    args = parser.parse_args()

    # Create plotter
    plotter = SimulationData(args.filepath)

    # Parse common arguments
    indices = None
    if args.indices:
        indices = [int(x.strip()) for x in args.indices.split(',')]

    components = None
    if args.components:
        components = [x.strip() for x in args.components.split(',')]

    time_range = None
    if args.time_range:
        parts = args.time_range.split(',')
        time_range = (float(parts[0]), float(parts[1]))

    # Execute requested action
    fig = None

    if args.list:
        plotter.list_datasets(filter_pattern=args.filter)

    elif args.interactive:
        plotter.interactive_plot()

    elif args.energy:
        fig, _ = plotter.plot_energy_summary(time_range=time_range, show=not args.save)

    elif args.stats:
        fig, _ = plotter.plot_element_stats(args.stats, time_range=time_range,
                                            show=not args.save)

    elif args.trajectory:
        traj_indices = [int(x.strip()) for x in args.trajectory.split(',')]
        fig, _ = plotter.plot_node_trajectory(traj_indices, projection=args.projection,
                                              time_range=time_range, show=not args.save)

    elif args.plot:
        if len(args.plot) == 1:
            fig, _ = plotter.plot(args.plot[0], indices=indices, components=components,
                                  time_range=time_range, show=not args.save)
        else:
            fig, _ = plotter.plot_multi(args.plot, indices=indices, components=components,
                                        time_range=time_range, show=not args.save)

    else:
        parser.print_help()

    # Save if requested
    if args.save and fig:
        fig.savefig(args.save, dpi=150, bbox_inches='tight')
        print(f"Saved figure to: {args.save}")


if __name__ == '__main__':
    main()
    """
    
    ## Usage Examples
    
    ### Command Line
    
    ```bash
    # List all datasets
    python -m demlat.utils.plot_timeseries experiments/test/output/simulation.h5 --list
    
    # Plot system energies
    python -m demlat.utils.plot_timeseries simulation.h5 --energy
    
    # Plot specific dataset
    python -m demlat.utils.plot_timeseries simulation.h5 --plot nodes/positions --indices 0,1,2 --components z
    
    # Plot bar strain statistics (mean, min, max over all bars)
    python -m demlat.utils.plot_timeseries simulation.h5 --stats elements/bars/strain
    
    # Plot node trajectories in 3D
    python -m demlat.utils.plot_timeseries simulation.h5 --trajectory 0,4,8 --projection 3d
    
    # Plot multiple datasets as subplots
    python -m demlat.utils.plot_timeseries simulation.h5 --plot system/kinetic_energy system/potential_energy elements/bars/strain
    
    # Time range and save
    python -m demlat.utils.plot_timeseries simulation.h5 --plot nodes/positions -i 0 -c z -t 0,5 --save trajectory.png
    ```
    
    ### Python API
    
    ```python
    from demlat.utils.plot_timeseries import SimulationData
    
    plotter = SimulationData("experiments/yoshimura/output/simulation.h5")
    
    # List what's available
    plotter.list_datasets()
    
    # Quick energy check
    plotter.plot_energy_summary()
    
    # Plot specific nodes
    plotter.plot("nodes/positions", indices=[0, 4, 8], components=['z'])
    
    # Plot strain statistics
    plotter.plot_element_stats("elements/bars/strain")
    
    # Node trajectory
    plotter.plot_node_trajectory([0, 1, 2], projection='xz')
    
    # Multiple plots
    plotter.plot_multi([
        "system/kinetic_energy",
        "system/potential_energy",
        "elements/bars/strain"
    ], indices=[0, 1, 2])
    """
