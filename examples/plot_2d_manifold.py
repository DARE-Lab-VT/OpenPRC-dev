import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Load data
DATA_FILE = Path("experiments/yoshimura_beta_d_sweep/hysteresis_data.npz")
OUTPUT_DIR = DATA_FILE.parent

data = np.load(DATA_FILE)
betas = data['betas']
bin_centers = data['bin_centers']
loading_mean = data['loading_mean']
unloading_mean = data['unloading_mean']

# Filter: only beta < 42 degrees
beta_mask = betas < 45
betas = betas[beta_mask]
loading_mean = loading_mean[beta_mask]
unloading_mean = unloading_mean[beta_mask]

print(f"Loaded: {len(betas)} betas (filtered < 42°), {len(bin_centers)} displacement bins")

# === FIX: Invert displacement ===
bin_centers_fixed = bin_centers.max() - bin_centers
bin_centers_fixed -= bin_centers_fixed.min()

# Re-sort ascending
sort_idx = np.argsort(bin_centers_fixed)
bin_centers_fixed = bin_centers_fixed[sort_idx]
loading_mean = loading_mean[:, sort_idx]
unloading_mean = unloading_mean[:, sort_idx]

# Create meshgrid
DISP, BETA = np.meshgrid(bin_centers_fixed, betas)

# Shared colorbar limits
vmin = min(np.nanmin(loading_mean), np.nanmin(unloading_mean))
vmax = max(np.nanmax(loading_mean), np.nanmax(unloading_mean))

n_levels = 15
n = 3  # Yoshimura parameter

# Analytical curves
beta_analytical = np.linspace(betas.min(), betas.max(), 500)
beta_rad = np.deg2rad(beta_analytical)

# Three key curves
d_hyperfold = np.zeros_like(beta_analytical)  # d = 0
d_unfolded = np.tan(beta_rad)  # d = tan(β)
d_folded = np.sqrt(np.maximum(np.tan(beta_rad) ** 2 - np.tan(np.pi / (2 * n)) ** 2, 0))  # d = √(tan²β - tan²(π/2n))

# Plot style settings
plt.rcParams['mathtext.fontset'] = 'cm'
plt.rcParams['font.size'] = 12

# Dark red color
DARK_RED = [1, 0, 0, 1]


def plot_heatmap_with_curves(beta_grid, disp_grid, energy_data, title, filename):
    """Plot square heatmap with analytical curves."""
    fig, ax = plt.subplots(figsize=(8, 8))

    pcm = ax.pcolormesh(beta_grid, disp_grid, energy_data, cmap='jet', shading='auto', vmin=vmin, vmax=vmax)
    contours = ax.contour(beta_grid, disp_grid, energy_data, levels=n_levels, colors='white', linewidths=0.8, alpha=0.7)
    ax.clabel(contours, inline=True, fontsize=8, fmt='%.3f')

    # Analytical curves in dark red
    ax.plot(beta_analytical, d_hyperfold, color=DARK_RED, ls='-', lw=2, label=r'Hyperfold: $d/l = 0$')
    ax.plot(beta_analytical, d_unfolded, color=DARK_RED, ls='--', lw=2, label=r'Unfolded: $d/l = \tan\beta$')
    ax.plot(beta_analytical, d_folded, color=DARK_RED, ls=':', lw=2.5,
            label=r'Folded: $d/l = \sqrt{\tan^2\beta - \tan^2(\pi/2n)}$')

    cbar = fig.colorbar(pcm, ax=ax, label='Potential Energy (J)', shrink=0.8)
    ax.set_xlabel(r'$\beta$ (degrees)', fontsize=14)
    ax.set_ylabel(r'$d/l$', fontsize=14)
    ax.set_title(title, fontsize=14)
    ax.legend(loc='upper left', fontsize=10)
    ax.set_aspect('auto')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / filename, dpi=150)
    plt.show()


# Plot Loading
plot_heatmap_with_curves(
    BETA, DISP, loading_mean,
    title='Loading (Forward)',
    filename='2d_heatmap_loading_contours.png'
)

# Plot Unloading
plot_heatmap_with_curves(
    BETA, DISP, unloading_mean,
    title='Unloading (Backward)',
    filename='2d_heatmap_unloading_contours.png'
)

# Side-by-side comparison
fig, axes = plt.subplots(1, 2, figsize=(16, 7))

for ax, energy_data, title in zip(axes, [loading_mean, unloading_mean], ['Loading (Forward)', 'Unloading (Backward)']):
    pcm = ax.pcolormesh(BETA, DISP, energy_data, cmap='jet', shading='auto', vmin=vmin, vmax=vmax)
    cnt = ax.contour(BETA, DISP, energy_data, levels=n_levels, colors='white', linewidths=0.6, alpha=0.7)

    ax.plot(beta_analytical, d_hyperfold, color=DARK_RED, ls='-', lw=2, label=r'Hyperfold: $d/l = 0$')
    ax.plot(beta_analytical, d_unfolded, color=DARK_RED, ls='--', lw=2, label=r'Unfolded: $d/l = \tan\beta$')
    ax.plot(beta_analytical, d_folded, color=DARK_RED, ls=':', lw=2.5,
            label=r'Folded: $d/l = \sqrt{\tan^2\beta - \tan^2(\pi/2n)}$')

    ax.set_xlabel(r'$\beta$ (degrees)', fontsize=14)
    ax.set_ylabel(r'$d/l$', fontsize=14)
    ax.set_title(title, fontsize=14)
    ax.legend(loc='upper left', fontsize=9)

fig.colorbar(pcm, ax=axes, label='Potential Energy (J)', shrink=0.8)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / '2d_heatmap_comparison_contours.png', dpi=150)
plt.show()

# Clean 2D line plot of analytical curves only
fig, ax = plt.subplots(figsize=(8, 8))

ax.plot(beta_analytical, d_hyperfold, 'k-', lw=2, label=r'Hyperfold: $d/l = 0$')
ax.plot(beta_analytical, d_unfolded, 'b--', lw=2, label=r'Unfolded: $d/l = \tan\beta$')
ax.plot(beta_analytical, d_folded, 'r-', lw=2, label=r'Folded: $d/l = \sqrt{\tan^2\beta - \tan^2(\pi/2n)}$')

ax.set_xlabel(r'$\beta$ (degrees)', fontsize=14)
ax.set_ylabel(r'$d/l$', fontsize=14)
ax.set_title('Yoshimura Origami: Configuration Space', fontsize=14)
ax.legend(loc='upper left', fontsize=11)
ax.grid(True, alpha=0.3)
ax.set_xlim(betas.min(), betas.max())
ax.set_ylim(0, bin_centers_fixed.max() * 1.1)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'analytical_curves.png', dpi=150)
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
            bin_centers_fixed[valid]
        ))
    else:
        area = np.nan
    hysteresis_areas.append(area)

hysteresis_areas = np.array(hysteresis_areas)

fig, ax = plt.subplots(figsize=(8, 8))
ax.plot(betas, hysteresis_areas, 'purple', lw=2)
ax.fill_between(betas, 0, hysteresis_areas, alpha=0.3, color='purple')
ax.set_xlabel(r'$\beta$ (degrees)', fontsize=14)
ax.set_ylabel('Hysteresis Area (J)', fontsize=14)
ax.set_title(r'Energy Dissipation vs Fold Angle ($\beta < 42°$)', fontsize=14)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'hysteresis_area_vs_beta.png', dpi=150)
plt.show()
