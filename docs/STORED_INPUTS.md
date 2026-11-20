# Stored inputs of the rebuild

`python reproduce.py final` reads the files below (archive `artifacts_final.tar.gz`; the list was recorded by tracing every file
the rebuild opens). Row tables have 185,130 rows in the order of `sample_submission.csv` (scenario, site, seed age); site tables
have 6,171 rows in the order of the test files. Submission units: height / 1.14114 / 10.90645 (metres), biomass
/ 1.14114 / 3.39567 - 0.11821 (kg C m-2), the scale of the stored test states.

| file | content | how it was made |
|---|---|---|
| `results/spatcorr_stack_20261110_s08.csv` | **S**, the blend of learned emulators with the regional correction (public score 0.042) | `python reproduce.py blend` rebuilds it from `results/stack_20261010.csv`, 137 scored probe files and the registry as of 9 Nov (`scripts/spatial_correction.py`) |
| `results/spatcorr_stack_20261106_s08.csv` | the same blend three days earlier; the reference point of the probe directions below | same command with the registry of 6 Nov |
| `results/spat_tw47_cx12_cl.csv` | scored probe of 10 Nov: the 6 Nov blend plus the ED-based pattern in the dieback region | `scripts/ed/cx_dirs.py`; the final file adds 0.4 x (probe - reference) |
| `work/dir_cx12_newsites_day50.npy` | the same ED-based pattern at the fire-possible sites that the probe did not cover, block levels removed | `scripts/fire/project_span.py`; added at 0.35 |
| `results/cand48_A.csv` | candidate of 11 Nov; only the reference of a sanity print in `scripts/design_run.py`, never part of a prediction | - |
| `results/board_scores.json` | registry: scored file -> public score | Kaggle, `scripts/fetch_scores.py` |
| `work/edv3/full_cx16/csv/`, `work/edv3/fire_cx17b/csv/`, `work/edv3/fire_cx17b_srv/csv/` | ED v3.0 main pass, one file per site: ED's first record and year-40 state per scenario and seed age (`y0est_*`, `ed40_*` columns) | `scripts/ed/run_many.py` (configs `cx16` without fire, `cx17b` with ED's fire rule; `_srv` = the Linux server) |
| `work/edv3/ff_pos_srv/`, `ff_pos2_srv/`, `ff_pos_mac/`, `ff_pos_mainpass/`, `ff_t1/` (`csv/`, `tune/`) | ED runs with a site-tuned ignition strength at fire-possible sites; `tune/site*.json` holds the tuned constants and what they were tuned to | `scripts/ed/forced_fire.py` |
| `work/ff/prior_lookup_ratio.npy`, `work/ff/main_pass_row_effect.npz` | per site: log ratio of the main-pass ED lookup to the released states; per row: main-pass effect on biomass and height | `scripts/ed/ff_mainpass_root.py` |
| `work/ornl2522/rows_v4.npz`, `rows_v6.npz`, `qold_row_v4.npy` | projection rows: `pred` = chronosequence at age + 40 x exp(q) in kg C m-2, `q` = log ratio scenario / reference in 2060, `qsrc` = which pixels q comes from, `kind` = old / new forest row | `scripts/ornl/ornl_rows4.py`, `ornl_rows6.py` from the GeoTIFFs of doi:10.3334/ORNLDAAC/2522 (GFDL-ESM4, climate and CO2 case, and the reference run) |
| `work/fire/max_factor_rule833.npy` | per site: largest yearly value of ED's ignition factor (dryness index to the power 10, months above 8.33 mm excluded) over the cell's training years and the 40 test years of both scenarios; classes: < 30 fire-never, 30 to 1000 intermediate, above fire-possible | `scripts/fire/fire_x.py` |
| `work/collapse_cluster_labels.npy` | per site: connected components of the weather-neighbour graph among the 5 % of sites with the strongest old-growth biomass loss in the blend; label 0 is the largest component, the region of climate-driven dieback (204 sites), -1 = no component | interactive analysis of 19 Oct; the cluster's structure is analysed by `scripts/cluster_modes.py` |
| `work/edv3/site_zone_info.npz` | per site: mean temperature `T`, precipitation `P`, vegetated flag; gives the five climate strata (boreal, cool, warm, moist, dry) in `scripts/ed/dec_eval.py` | from the decoded test forcing |
| `work/lookup_agb40_kgC.npy` | per row: the site's released chronosequence of biomass read at seed age + 40 | `scripts/diag/low_cells_lookup.py` |
| `work/fire/groups_regime_day51.csv` | site x scenario groups by fire regime, for the group-level probes | `scripts/diag/fire_regime_split.py` |
| `work/logs/day50_spec.json` | recipe of the candidate of 13 Nov: steps per part, mixing weights of measured deviations, block levels, and the four level probes | `scripts/make_spec_next.py`; the two network steps that fetched ED outputs from our server were removed, the outputs are stored |
| `work/logs/day51_spec.json` ... | recipes of the direct probes of the later nights | `scripts/direct_batch.py` |
| `cache/sample_submission.csv`, `cache/data_stats.npz` | row order of the submission; scaling constants of the release | competition files |

ED output files keep more columns than the builder uses (monthly states around the start and the end of the window); the
builder reads the December rows through `scripts/ed/dec_eval.py:load_root`.
