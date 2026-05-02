# Training Figure Reproduction Guide

This directory contains the training and plotting scripts used for the
feature-observability and data-scaling figures in the physical reservoir
automodeling draft.

The commands below assume:

- You run from the repo `src` directory:

  ```bash
  cd /home/wensin/Documents/OpenPRC-dev/src
  ```

- The automod bundle already exists at:

  ```text
  openprc/automod/robot_bundle
  ```

- Go1 has trajectory files and reservoir simulations:

  ```text
  openprc/automod/robot_bundle/go1/trajectories/*.h5
  openprc/automod/robot_bundle/go1/reservoir_sims/*/output/simulation.h5
  ```

- The default experiment uses:

  ```text
  robot:       go1
  features:    strain,strain_rate,node_vel
  targets:     base_lin_vel,base_ang_vel,qvel
  target_fps:  50
  skip_seconds: 2.0
  ```

## Figure Map

The PDF figures correspond to these generated files:

```text
Figure 1: all_clips_heatmap/all_clips_r2_heatmap.svg
Figure 2: data_efficient_training/overlay_plots/ranking_heatmap.svg
Figure 3: data_efficient_training/overlay_plots/curve_overlay_<target>.svg
Figure 4: split_fraction_sweep/plots/split_fraction_overlay_<target>_strain.svg
```

For Figure 3, the paper uses three target panels:

```text
curve_overlay_base_lin_vel.svg
curve_overlay_base_ang_vel.svg
curve_overlay_qvel.svg
```

For Figure 4, the paper uses the successful strain readouts:

```text
split_fraction_overlay_base_ang_vel_strain.svg
split_fraction_overlay_qvel_strain.svg
```

## 1. Run Single-Clip and Ranked-Cumulative Training

This step trains one trajectory at a time, ranks clips by held-out R2, then
adds clips from best to worst for each feature-target pair.

```bash
python3 -m openprc.automod._tools.training.data_efficient_training \
  --bundle-dir openprc/automod/robot_bundle
```

Main outputs:

```text
openprc/automod/robot_bundle/go1/training/data_efficient_training/
  single_trajectory_ranking.csv
  cumulative_learning_curve.csv
  data_efficient_summary.json
  plots/
```

This CSV is also used by Figure 1 because the final cumulative row gives the
all-clips result for each feature-target pair.

## 2. Make Figure 1: All-Trajectory Heatmap

Figure 1 asks:

```text
When all selected clips are used, which reservoir feature predicts each target best?
```

Generate it with:

```bash
python3 -m openprc.automod._tools.training.plot_all_clips_heatmap \
  --bundle-dir openprc/automod/robot_bundle
```

Output:

```text
openprc/automod/robot_bundle/go1/training/all_clips_heatmap/
  all_clips_r2_heatmap.svg
```

The script reads:

```text
data_efficient_training/cumulative_learning_curve.csv
```

and keeps the largest `k` row for each feature-target pair.

## 3. Run Sequential Clip Accumulation

This step adds consecutive Go1 corridor clips in natural order:

```text
corridor_000
corridor_001
...
corridor_013
```

It tests whether a feature-target relation stays stable as the training data
expands from a local segment to broader trajectory coverage.

```bash
python3 -m openprc.automod._tools.training.sequential_clip_training \
  --bundle-dir openprc/automod/robot_bundle
```

Main outputs:

```text
openprc/automod/robot_bundle/go1/training/sequential_clip_training/
  sequential_learning_curve.csv
  sequential_summary.json
  plots/
```

## 4. Make Figure 2 and Figure 3 Overlays

This plotting script combines:

- `single_trajectory_ranking.csv` from `data_efficient_training`
- `sequential_learning_curve.csv` from `sequential_clip_training`
- `cumulative_learning_curve.csv` as a fallback for any missing pair

Run:

```bash
python3 -m openprc.automod._tools.training.data_efficient_overlay \
  --bundle-dir openprc/automod/robot_bundle
```

Outputs:

```text
openprc/automod/robot_bundle/go1/training/data_efficient_training/overlay_plots/
  ranking_heatmap.svg
  curve_overlay_base_lin_vel.svg
  curve_overlay_base_ang_vel.svg
  curve_overlay_qvel.svg
```

Use these as:

```text
Figure 2: ranking_heatmap.svg
Figure 3: curve_overlay_base_lin_vel.svg
          curve_overlay_base_ang_vel.svg
          curve_overlay_qvel.svg
```

## 5. Run Training-Data Scaling Sweeps

Figure 4 compares two fixed-test protocols for the strain readout:

1. In-clip fixed test:

   ```text
   train = first alpha fraction of each clip
   test  = last beta = 0.3 fraction of each clip
   ```

2. Global concatenated fixed test:

   ```text
   concatenate all selected clips first
   train = first alpha fraction of the concatenated matrix
   test  = last beta = 0.3 fraction of the concatenated matrix
   ```

The paper focuses on the stable strain readouts:

```text
base_ang_vel
qvel
```

Run the in-clip fixed-test sweep:

```bash
python3 -m openprc.automod._tools.training.split_fraction_sweep \
  --bundle-dir openprc/automod/robot_bundle \
  --features strain \
  --targets base_ang_vel,qvel \
  --split-method fixed-test
```

Run the global concat fixed-test sweep:

```bash
python3 -m openprc.automod._tools.training.split_fraction_sweep \
  --bundle-dir openprc/automod/robot_bundle \
  --features strain \
  --targets base_ang_vel,qvel \
  --split-method concat-fixed-test
```

Main outputs:

```text
openprc/automod/robot_bundle/go1/training/split_fraction_sweep/
  split_fraction_sweep.csv
  split_fraction_summary.json
  plots/
```

Important: `split_fraction_sweep.csv` is rewritten each time, but the actual
per-run training directories are separate because the run IDs include the split
method:

```text
splitfrac_fixed-test_<target>_strain_<pct>
splitfrac_concat-fixed-test_<target>_strain_<pct>
```

## 6. Make Figure 4 Overlay

After both scaling sweeps exist, overlay the two protocols:

```bash
python3 -m openprc.automod._tools.training.plot_split_fraction \
  --bundle-dir openprc/automod/robot_bundle \
  --features strain \
  --targets base_ang_vel,qvel \
  --split-methods fixed-test,concat-fixed-test \
  --labels in-clip,global-concat
```

Outputs:

```text
openprc/automod/robot_bundle/go1/training/split_fraction_sweep/plots/
  split_fraction_overlay_base_ang_vel_strain.svg
  split_fraction_overlay_qvel_strain.svg
```

Use these two files as Figure 4 panels:

```text
Figure 4a: split_fraction_overlay_base_ang_vel_strain.svg
Figure 4b: split_fraction_overlay_qvel_strain.svg
```

## Reusing Existing Runs

Most wrapper scripts skip completed training runs by default if the expected
`metrics.csv` already exists. To force retraining, pass:

```bash
--force-train
```

To regenerate plots or CSV summaries from existing run directories without
launching new training, pass:

```bash
--no-train
```

For example:

```bash
python3 -m openprc.automod._tools.training.split_fraction_sweep \
  --bundle-dir openprc/automod/robot_bundle \
  --features strain \
  --targets base_ang_vel,qvel \
  --split-method fixed-test \
  --no-train
```

## Suggested Minimal Reproduction Order

Run these in order:

```bash
cd /home/wensin/Documents/OpenPRC-dev/src

python3 -m openprc.automod._tools.training.data_efficient_training \
  --bundle-dir openprc/automod/robot_bundle

python3 -m openprc.automod._tools.training.plot_all_clips_heatmap \
  --bundle-dir openprc/automod/robot_bundle

python3 -m openprc.automod._tools.training.sequential_clip_training \
  --bundle-dir openprc/automod/robot_bundle

python3 -m openprc.automod._tools.training.data_efficient_overlay \
  --bundle-dir openprc/automod/robot_bundle

python3 -m openprc.automod._tools.training.split_fraction_sweep \
  --bundle-dir openprc/automod/robot_bundle \
  --features strain \
  --targets base_ang_vel,qvel \
  --split-method fixed-test

python3 -m openprc.automod._tools.training.split_fraction_sweep \
  --bundle-dir openprc/automod/robot_bundle \
  --features strain \
  --targets base_ang_vel,qvel \
  --split-method concat-fixed-test

python3 -m openprc.automod._tools.training.plot_split_fraction \
  --bundle-dir openprc/automod/robot_bundle \
  --features strain \
  --targets base_ang_vel,qvel \
  --split-methods fixed-test,concat-fixed-test \
  --labels in-clip,global-concat
```

After this, the paper-ready SVGs are under:

```text
openprc/automod/robot_bundle/go1/training/
```

