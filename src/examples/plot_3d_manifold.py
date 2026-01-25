import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Load data
DATA_FILE = Path("experiments/yoshimura_beta_d_sweep/hysteresis_data.npz")  # Update this
OUTPUT_DIR = DATA_FILE.parent

data = np.load(DATA_FILE)
betas = data['betas']
bin_centers = data['bin_centers']
loading_mean = data['loading_mean']
loading_min = data['loading_min']
loading_max = data['loading_max']
unloading_mean = data['unloading_mean']
unloading_min = data['unloading_min']
unloading_max = data['unloading_max']

print(f"Loaded: {len(betas)} betas, {len(bin_centers)} displacement bins")

# Create meshgrid for surface plots
DISP, BETA = np.meshgrid(bin_centers, betas)


def plot_3d_envelope(beta_grid, disp_grid, mean_data, min_data, max_data,
                     title, color, filename):
    """Plot 3D surface with envelope."""
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')  # This auto-imports Axes3D in modern matplotlib

    # Main surface (mean)
    surf = ax.plot_surface(
        BETA, DISP, mean_data,
        cmap=color, alpha=0.8, edgecolor='none'
    )

    # Upper envelope (transparent)
    ax.plot_surface(
        BETA, DISP, max_data,
        color='gray', alpha=0.2, edgecolor='none'
    )

    # Lower envelope (transparent)
    ax.plot_surface(
        BETA, DISP, min_data,
        color='gray', alpha=0.2, edgecolor='none'
    )

    ax.set_xlabel('Beta (degrees)', fontsize=12, labelpad=10)
    ax.set_ylabel('Displacement (m)', fontsize=12, labelpad=10)
    ax.set_zlabel('Potential Energy (J)', fontsize=12, labelpad=10)
    ax.set_title(title, fontsize=14, pad=20)

    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10, label='Energy (J)')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / filename, dpi=150)
    plt.show()


# Plot Loading (Forward)
plot_3d_envelope(
    BETA, DISP, loading_mean, loading_min, loading_max,
    title='Loading (Forward): Beta vs Displacement vs Potential Energy',
    color='Blues',
    filename='3d_envelope_loading.png'
)

# Plot Unloading (Backward)
plot_3d_envelope(
    BETA, DISP, unloading_mean, unloading_min, unloading_max,
    title='Unloading (Backward): Beta vs Displacement vs Potential Energy',
    color='Oranges',
    filename='3d_envelope_unloading.png'
)

# Combined plot
fig = plt.figure(figsize=(14, 10))
ax = fig.add_subplot(111, projection='3d')

ax.plot_surface(BETA, DISP, loading_mean, cmap='Blues', alpha=0.7, edgecolor='none')
ax.plot_surface(BETA, DISP, unloading_mean, cmap='Oranges', alpha=0.7, edgecolor='none')

ax.set_xlabel('Beta (degrees)', fontsize=12, labelpad=10)
ax.set_ylabel('Displacement (m)', fontsize=12, labelpad=10)
ax.set_zlabel('Potential Energy (J)', fontsize=12, labelpad=10)
ax.set_title('Hysteresis: Loading vs Unloading Surfaces', fontsize=14, pad=20)

from matplotlib.patches import Patch

legend_elements = [
    Patch(facecolor='steelblue', alpha=0.7, label='Loading'),
    Patch(facecolor='orangered', alpha=0.7, label='Unloading')
]
ax.legend(handles=legend_elements, loc='upper right')

plt.tight_layout()
plt.savefig(OUTPUT_DIR / '3d_hysteresis_combined.png', dpi=150)
plt.show()

# Hysteresis area vs beta
hysteresis_areas = []
for i in range(len(betas)):
    load_valid = ~np.isnan(loading_mean[i])
    unload_valid = ~np.isnan(unloading_mean[i])
    valid = load_valid & unload_valid

    if np.sum(valid) > 2:
        area = np.abs(np.trapezoid(
            loading_mean[i, valid] - unloading_mean[i, valid],
            bin_centers[valid]
        ))
    else:
        area = np.nan
    hysteresis_areas.append(area)

hysteresis_areas = np.array(hysteresis_areas)

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(betas, hysteresis_areas, 'purple', lw=2)
ax.fill_between(betas, 0, hysteresis_areas, alpha=0.3, color='purple')
ax.set_xlabel('Beta (degrees)', fontsize=12)
ax.set_ylabel('Hysteresis Area (J)', fontsize=12)
ax.set_title('Energy Dissipation vs Fold Angle', fontsize=14)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'hysteresis_area_vs_beta.png', dpi=150)
plt.show()
