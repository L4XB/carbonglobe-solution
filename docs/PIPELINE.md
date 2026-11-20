# Pipeline map

Every prediction of the final file is `P = S + s_k (T - S)`. The table lists, per stage, the scripts, what they read and what
they write. Paths are relative to the folder that holds `scripts/` and `src/`. "stored" = shipped in `3_model_weights`.

## 1. Preprocessing and validation set-up

| script | does | writes |
|---|---|---|
| `scripts/verify_train_shapes.py` | checks the release layout: 41 year slots, slot 0 = initial state | - |
| `scripts/cache_arrays.py` | yearly trajectories, annual means and seasonal climatology of the training forcing | `work/cache/y_traj.npy`, `x_annual.npy`, `x_season.npy` |
| `src/cgb/holdout.py` | site hold-out: 6,171 sites in two halves, 47,981 sites to fit (seed 42) | split in memory |
| `src/cgb/shift.py`, `scripts/theory_y0_scale.py` | decode of the stored test states: physical = 1.14114 x (stored - floor) | constants `FLOORS`, `M_HEIGHT` |
| `scripts/geo/cell_identity.py`, `cell_identity_global.py` | static soil channels of test sites against training cells: 5,994 exact matches | `work/test_cell_exact.npz` |
| `scripts/geo/y0_identity.py`, `scripts/nex/*.py`, `scripts/repair_test_cache.py` | affine codes of the forcing channels, decoded test caches | `work/cal/`, `work/cache/test_*` |

## 2. Learned emulators (reference S)

| script | does | writes |
|---|---|---|
| `scripts/train_latent.py` (`src/cgb/latent.py`) | GRU latent rollout; AdamW, one-cycle 2e-3, weight decay 1e-4, clip 1, 32 sites per batch; loss = squared error in scored units, year 40 weighted 10x, unscored variables 0.25x | `work/models/latent_<tag>/model.pt`, `results.json` (stored: `emulators/`) |
| `scripts/direct_gbdt.py` | LightGBM, one booster per target (forty-year change of height / biomass) | `work/models/direct_<tag>/height.txt`, `agb.txt` (stored) |
| `scripts/predict_latent.py`, `scripts/predict_gbdt.py` | inference on the competition test files | `results/latent_*.csv`, `results/direct_*.csv` |
| `scripts/holdout_blend.py`, `scripts/eval_stage.py` | hold-out scores and blends | printed tables |

The chronosequence predictor needs no model: the fifteen `test_y0` values of a site, interpolated log-linearly in seed age and
read at age + 40 (`scripts/shift_apply.py`, `scripts/chrono_dynamics.py`).

## 3. Blend S and regional correction (public scores)

| script | does | writes |
|---|---|---|
| `scripts/fetch_scores.py`, `scripts/board_registry.py` | registry of public scores | `results/board_scores.json` (stored) |
| `scripts/file_solve.py`, `scripts/board_solve.py`, `scripts/block_solve.py`, `scripts/young_block_solve.py` | blend weights by ridge regression in prediction space on the scores | `results/fsolve_*.csv`, ... (stored as scored files) |
| `scripts/stack_blends.py` | convex stack of the scored blends | `results/stack_20261010.csv` (stored) |
| `scripts/spatial_probe.py`, `scripts/dir_probe.py`, `scripts/perp_probe.py`, `scripts/field_probe.py` | diagnostic files: a reference plus one pattern | `results/spat_*.csv`, `results/perp_spat_*.csv` (stored), `work/spat_centers.json` |
| `scripts/cluster_public_share.py` | public share of probes that sit on few sites | `work/cluster_ratio_pred.json` |
| `scripts/spatial_correction.py --ridge 1e-4 --scale 0.8` | joint ridge correction from all scored probes | `results/spatcorr_stack_20261110_s08.csv` = **S** (`reproduce.py blend`) |

## 4. Process targets T

| script | does | writes |
|---|---|---|
| `scripts/ed/run_site.py`, `scripts/ed/run_many.py`, `scripts/ed/launch_run.sh` | ED v3.0 per site: spin-up on the matched cell's training years (cycled, 415 ppm), then the 40 decoded test years for 15 seed ages and both scenarios; binaries built from the public source with `ed_patch/` | `work/edv3/<run>/csv/site*.csv` (stored) |
| `scripts/ed/dec2.py`, `scripts/ed/dec_eval.py` | row targets from ED: chronosequence at age + 40 x ED's change against its own chronosequence; memory decode; gates on the initial-state mismatch | used by the builder |
| `scripts/ed/forced_fire.py`, `scripts/ed/ff_rows.py` | ED runs with a site-tuned ignition strength at fire-possible sites; row values | `work/edv3/ff_*/csv`, `tune` (stored), `work/ff/ff_rows_day50.npz` (rebuilt) |
| `scripts/ornl/download_ornl2522.sh`, `scripts/ornl/ornl_rows.py` ... `ornl_rows6.py` | projection rows: chronosequence at age + 40 x exp(q), q from the external biomass projections at the test cells | `work/ornl2522/rows_v4.npz`, `rows_v6.npz` (stored) |
| `scripts/fire/fire_x.py`, `scripts/fire/emulator.py` | ED's ignition rule evaluated on site climates; site classes | `work/fire/max_factor_rule833.npy` (stored) |
| `scripts/ed/train_site.py`, `train_height_eval.py`, `train_decode_eval.py`, `train_memory_eval.py`, `ff_train_eval.py` | the same ED set-up on training cells with known outcome | `work/edv3/train_hv2/height_eval_rows.csv` (stored), printed tables |

## 5. Final file

| script | does | writes |
|---|---|---|
| `scripts/make_spec_next.py` | writes the recipe of a candidate and its probe files | `work/logs/day50_spec.json` (stored) |
| `scripts/design_run.py` | realises a recipe: builder runs, mixing of measured deviations, level tools | `results/cand50_A.csv` and four level probes |
| `scripts/build_candidate2.py` | `P = S + step[part] x (target - S)`; part labels and default steps are listed at the top of the file | intermediate files `results/td_*.csv` |
| `scripts/apply_ff.py`, `apply_cal.py`, `apply_dir.py`, `apply_level.py`, `apply_contrast.py`, `apply_group_level.py` | forced-fire heights, block levels, measured directions, probe changes | - |
| `scripts/direct_batch.py` | probe files: the scored base with one block changed | `results/cand5x_P*.csv` |
| `scripts/readout_joint.py` | public scores of the probes -> shrunk block levels (prints the steps for the second stage) | printed |
| `scripts/second_stage.py` | `final = base + sum_j m_j (probe_j - base)` | the final file |

## 6. Checks quoted in the report

| script | does |
|---|---|
| `scripts/report/train_cell_check.py` | ED targets on 120 training cells against the known year-40 states (0.010 against 0.022 for the chronosequence alone) |
| `scripts/report/composition.py` | share of the final file that comes from each source |
| `scripts/sim/synth_board.py`, `synth_stack.py`, `synth_two_stage.py` | synthetic leaderboards with the competition's site split and rounding: transfer of fitted corrections to the unseen half, null test of the shrinkage |
