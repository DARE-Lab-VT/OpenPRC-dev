# Clean-IID IPC

IPC uses the **original uniform IID symbols that drove acquisition**, separately
from interpolated commands, laser measurements, or saved actuator positions.
A seed alone does not identify a sequence across generators or languages.

## Library contract

`compute_ipc_components` and `compute_ipc_components_gpu` in
`openprc.analysis.utils.training_utils` accept prepared states with one row per
original IID symbol. Align timestamps and perform optional temporal multiplexing
before calling them. These functions do not standardize states internally.
`MemoryBenchmark` retains its existing full-record z-score convention; its input
must already have one IID value per feature row. It cannot infer IID timing from
an arbitrary actuation waveform. For transformed rows on another clock, pass
`sample_dt` in seconds per row to MemoryBenchmark. Its default remains loader.dt.
The heatmap example reuses StateLoader, NodePositions, Trainer, Ridge and
MemoryBenchmark. The example calls NodePositions directly, subtracts the initial positions,
then aligns/multiplexes those displacements. It passes this unstandardized matrix
as prepared_states to MemoryBenchmark; sample_dt=1/IID_HZ controls duration
conversion. No custom node-position feature class is used. Standardization
and IPC fitting stay inside the existing benchmark. Results are saved using its
native metrics.h5 convention, alongside the exact/cumulative plots.

Trainer is the benchmark configuration object here; the IPC fits are performed
by MemoryBenchmark's existing memory_task path. Optional saved-readout retraining
still uses Trainer's readout implementation and is not changed by this example.
That optional path rejects a differing sample_dt/loader clock or externally
prepared_states, because Trainer would otherwise re-extract different features.
The default benchmark path without prepared_states is unchanged.

- `input_bounds=(-1,1)` describes the uniform distribution's population bounds.
  Use `(0,1)` for U(0,1), for example. No empirical min/max scaling.
- Normalized Legendre products span degrees 1..n_s and lag indices 0..tau_s,
  with physical delays equal to index * k_delay (in symbols).
- Every target uses the same start `max(washout, tau_s*k_delay)` and train_stop.
  Test rows follow training immediately; input history may cross this boundary.
- Separate target solves use trace-scaled ridge and an unpenalized intercept.
  State designs/Gram remain float32 for compatibility; targets are float64.
- Scores are held-out R² = 1 - SSE / centered SST. Only negative scores are
  clipped to zero. `return_raw=True` exposes unthresholded scores. There is no
  positive-capacity or effective-rank cutoff.
- CPU and Torch paths use the same target generation and windows. GPU results
  can differ slightly in floating-point arithmetic. GPU chunk_size is retained
  as a compatibility argument; computation now streams separate target solves.
- Mismatched lengths and invalid windows now raise instead of silently cropping.
  Alignment and multiplexing happen before IPC; no internal downsampling option.
- API cleanup: `epsilon`, `eps` and `interp_factor` have been removed. Remove
  these keywords from old calls. Optional solver arguments are now keyword-only
  to prevent old positional options being silently reinterpreted. MemoryBenchmark
  rejects unsupported keywords instead of silently ignoring them.
- Legacy monomial `compute_R2` is unchanged and is not orthogonal IPC.

`aggregate_ipc_heatmaps` produces (1) exact degree/max-delay **sums** and
(2) cumulative **means** over all targets in degree/delay rectangles, counting
zero capacities. These are not exponent-weighted memory cells, and neither is
required to be monotonic. Finite-sample estimates, clipping and regularization
mean summed estimates need not obey the ideal population rank bound exactly.
Full-record standardization is retained for compatibility, not presented as a
strict training-only preprocessing protocol or an intrinsic-capacity proof.

## Example

The example keeps an editable USER SETTINGS section for paths, original IID
symbols, feature selection, training windows and plotting. Set STATE_FILE there.
With IID_FILE=None (the default), the example generates uniform symbols using
IID_GENERATOR="numpy_pcg64", IID_SEED=42, IID_HZ=1/0.033 and IID_DURATION=30:
910 values in INPUT_BOUNDS (default [-1,1]). These defaults match the bundled
generation_5 simulation's saved input, verified by reconstructing its cubic
spline command. SNAPSHOT_PHASES=(0,) matches its one saved frame per symbol.
For another recording, change these settings to match its acquisition.
Generation includes symbol onsets before IID_DURATION; the last interval may
be partial, but every requested snapshot must remain within the recorded times.
For example, exactly 30 Hz and 120 seconds still produces 3,600 symbols.
To use a saved sequence instead, set IID_FILE; its values and length override
all generation settings. Then run without arguments:

```bash
python -m openprc.examples.run_plot_heatmap
```

Optional command-line arguments still override those settings. From the
checkout's src directory (or an installed OpenPRC environment):

```bash
python -m openprc.examples.run_plot_heatmap /path/to/output/simulation.h5 \
  --iid-file /path/to/original_symbols.npy --iid-hz 30 \
  --input-bounds -1 1 --iid-start 0 --phases 0 .5 \
  --washout 5 --train-fraction .5 --max-degree 4 --max-delay 30 --no-show
```

To generate without an IID file, omit --iid-file and optionally specify
`--iid-generator numpy_randomstate --iid-seed 42 --iid-duration 120 --iid-hz 30`.
No generated input file is written; the symbols are used in memory.

A one-dimensional H5 dataset is also accepted via `--iid-dataset`. Input files
are read only. The example retains native time_series/time and
 time_series/nodes/positions datasets, takes each node's initial position as
zero, and defaults to xy/all nodes. Phase interpolation supports noninteger
camera/IID rate ratios without extrapolation. It cannot create new independent
measurements or compensate unknown offsets. The supplied or regenerated IID values must match acquisition, including their
correct onset on the state clock. Matching seeds across different generators
(including MATLAB versus NumPy) does not establish matching values.

Both plots are saved under output/plots as heatmap_clean_iid and
heatmap_clean_iid_cumulative, in PNG and SVG, with two-decimal labels. No native
input/output H5 schema, physics solver, or file naming convention is changed.

## Tests

```bash
PYTHONPATH=src python -m unittest discover -s tests -p 'test_ipc.py' -v
```

Tests cover a known delay bank, a nonlinear product reservoir, an unrelated-IID
negative control, population scaling, common history windows, raw/clipped
scores, aggregation and CPU/Torch parity. CUDA parity also runs when available.
These are implementation checks, not evidence that a particular experiment's
IID provenance, timestamp alignment or asymptotic capacity is established.
