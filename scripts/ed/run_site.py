#!/usr/bin/env python
"""Run ED v3.0 for one CarbonGlobe test site: inputs -> one spin-up -> 2 x 15 windows (year 0 + 40) -> CSVs.

Layout under --root:
  inputs/site<id>_<scen>/      netCDF inputs (deleted afterwards unless --keep)
  run/site<id>/                cfg files + ED outputs (deleted afterwards unless --keep)
  results/<site>_<scen>.csv    age,height_m,agb_kgC,soil,lai,gpp,npp,rh : December state of window year 40 (t = 41)
  results/<site>_<scen>_y0.csv the same columns after the year-0 step (t = 1): ED's own estimate of test_y0
  csv/site<id>.csv             detailed per-row table (ED states, spin-up state, decoded y0, lat/lon, scenario)
  csv/site<id>.json            timings and configuration

Configs (--config): b    = stored CO2, channels as delivered, spin-up CO2 = stored year 1 (364 ppm)
                    c400 = exact SSP CO2 (co2_real_paths.json), repaired channels, spin-up CO2 400 ppm
                    c417 = as c400 with the spin-up at the 2020 level (year-0 value of the json)
All: one spin-up per site from the mean of the first --spinup-years years of both scenarios, fire off,
treefall 0.012, 500 spin-up years with yearly dumps, each window restarted from the dump of its seed age
for 41 years (year 0 = the first window year's forcing at the 2020 CO2 level, then 2021-2060).
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "ed"))
import make_site_inputs as msi  # noqa: E402

EDV3 = ROOT / "work" / "edv3"
EDLU = Path(os.environ.get("ED_EDLU", str(EDV3 / "build" / "edlu")))
USE_MECH_CACHE = os.environ.get("ED_MECH_CACHE", "1") != "0"  # per-(site, scenario) cache of the yearly An tables across the 15 age windows
AGES = list(msi.SEED_AGES)
ENV_COMMON = {"ED_CO2_AS_IS": "1", "ED_TREEFALL": "0.012"}
T_Y0, T_END = 1.0, float(os.environ.get("ED_T_END", "41.0"))   # ED_T_END=1.0: pre-year only (diagnostics against test_y0)
CONFIGS = {
    "b": dict(co2="asis", repair=False, co2_spinup="first"),
    "c400": dict(co2="real", repair=True, co2_spinup="fixed", co2_spinup_fixed=400.0),
    "c417": dict(co2="real", repair=True, co2_spinup="fixed", co2_spinup_fixed=-1.0),  # -1 -> year-0 level of the json
    "brep": dict(co2="asis", repair=True, co2_spinup="first"),  # stored CO2 but repaired channels (control)
    "c400a": dict(co2="asis", repair=True, co2_spinup="fixed", co2_spinup_fixed=400.0),  # window CO2 = stored channels (as in training), spin-up 400
    "c400f": dict(co2="real", repair=True, co2_spinup="fixed", co2_spinup_fixed=400.0, fire_on=True),  # c400 + ED's internal fire
    "c400f_late": dict(co2="real", repair=True, co2_spinup="fixed", co2_spinup_fixed=400.0, fire_on=True, spinup_start=30),  # spin-up on window years 31-40 (training: 2011-2020)
    "c400f_fp01": dict(co2="real", repair=True, co2_spinup="fixed", co2_spinup_fixed=400.0, fire_on=True, fp1=0.1),  # weaker fire ignition
    "c400f_late_fp01": dict(co2="real", repair=True, co2_spinup="fixed", co2_spinup_fixed=400.0, fire_on=True, spinup_start=30, fp1=0.1),
    "c400fw": dict(co2="real", repair=True, co2_spinup="fixed", co2_spinup_fixed=400.0, fire_on=True, spinup_start=30, spinup_cycle=True),  # spin-up on cycled yearly weather (training: 2011-2020)
    "c400fw0": dict(co2="real", repair=True, co2_spinup="fixed", co2_spinup_fixed=400.0, fire_on=True, spinup_start=0, spinup_cycle=True),  # test sites: first 10 window years cycled
    "c400_sh30": dict(co2="real", repair=True, co2_spinup="fixed", co2_spinup_fixed=400.0, spinup_start=0, spinup_shift=30.0),  # first-10-year climatology moved +30 yr along the trend (training test of the shift)
    "c400_shm10": dict(co2="real", repair=True, co2_spinup="fixed", co2_spinup_fixed=400.0, spinup_start=0, spinup_shift=-10.0),  # test sites: climatology moved back 10 yr (2011-2020 proxy)
    "c400aw": dict(co2="asis", repair=True, co2_spinup="fixed", co2_spinup_fixed=400.0, window_repair=False),  # c400a, but the yearly window files from the channels as delivered
    # c400a + per-age seed selection at cold vegetated sites (first-decade T < 8 C): extra spin-ups with the climatology
    # moved +15/+30/+45 yr along the trend, each seed age starts from the spin-up state closest to y0 (select_seeds)
    "c400as": dict(co2="asis", repair=True, co2_spinup="fixed", co2_spinup_fixed=400.0, seed_shifts=[15.0, 30.0, 45.0], seed_cold_t=8.0),
    # training validation of the seed selection (train_site.py forces co2 asis, repair off): first-decade climatology
    # (1982-1991) plus spin-ups moved +10/+20/+30 yr along the 1981-2020 trend, every site qualifies
    # c400as + the geographic-twin spin-up (nearest training site's 2011-2020 climatology, make_site_inputs.geo_twin) as a
    # fourth seed candidate
    "c400asg": dict(co2="asis", repair=True, co2_spinup="fixed", co2_spinup_fixed=400.0, seed_shifts=[15.0, 30.0, 45.0], seed_cold_t=8.0, seed_geo_twin=True),
    # c400a with the spin-up climatology of the geographically nearest training site (MERRA-2 2011-2020) at every age:
    # cluster test of the precipitation-shock reading (the twin is ~1.3 K warmer and ~29% wetter than the ESM decade)
    "c400ag": dict(co2="asis", repair=True, co2_spinup="fixed", co2_spinup_fixed=400.0, spinup_geo_twin=True),
    # c400asg with the shift grid moved up: +15 never helps once +30 and the twin are candidates (32 cold sites: old rms
    # unchanged without it), +45 is the most frequent winner, so +60 replaces +15
    "c400asg2": dict(co2="asis", repair=True, co2_spinup="fixed", co2_spinup_fixed=400.0, seed_shifts=[30.0, 45.0, 60.0], seed_cold_t=8.0, seed_geo_twin=True),
    # c400asg2 + geographic-twin BASE spin-up at the collapse-cluster sites (wet MERRA-2 decade -> drier ESM windows)
    "c400asg2c": dict(co2="asis", repair=True, co2_spinup="fixed", co2_spinup_fixed=400.0, seed_shifts=[30.0, 45.0, 60.0], seed_cold_t=8.0, seed_geo_twin=True, geo_base="cluster"),
    # fire ON (ED's internal potential fire, the experiment's "_potfire"): with the twin (MERRA-2 2011-2020) spin-up,
    # the organisers' protocol as far as we can reconstruct it; and with the base (ESM first-decade) spin-up
    "c400agf": dict(co2="asis", repair=True, co2_spinup="fixed", co2_spinup_fixed=400.0, spinup_geo_twin=True, fire_on=True),
    "c400af": dict(co2="asis", repair=True, co2_spinup="fixed", co2_spinup_fixed=400.0, fire_on=True),
    # fire only in the windows (seed lookup without fire, windows "_potfire"): training test on the s30 protocol
    "c400fw_only": dict(co2="real", repair=True, co2_spinup="fixed", co2_spinup_fixed=400.0, fire_on=True, fire_spin=False, spinup_start=30),
    "c400fw_only_fp03": dict(co2="real", repair=True, co2_spinup="fixed", co2_spinup_fixed=400.0, fire_on=True, fire_spin=False, spinup_start=30, fp1=0.3),
    "c400fw_only_fp01": dict(co2="real", repair=True, co2_spinup="fixed", co2_spinup_fixed=400.0, fire_on=True, fire_spin=False, spinup_start=30, fp1=0.1),
    # fire in the windows only, and the windows' climatology (dryness index) = mean of the window years
    "c400fw_only_wclim": dict(co2="real", repair=True, co2_spinup="fixed", co2_spinup_fixed=400.0, fire_on=True, fire_spin=False, spinup_start=30, window_clim="window_mean"),
    # c400asg2c + fire in the windows only (spin-ups without fire) at the cluster (twin base spin-up) and at warm sites
    # (per-age seeds between the base and the twin spin-up); cold sites unchanged (fire off, shift/twin seeds)
    "c400fwx": dict(co2="asis", repair=True, co2_spinup="fixed", co2_spinup_fixed=400.0, seed_shifts=[30.0, 45.0, 60.0], seed_cold_t=8.0,
                    seed_geo_twin=True,
                    site_overrides=[("cluster", dict(spinup_geo_twin=True, fire_on=True, fire_spin=False, seed_shifts=[], seed_geo_twin=False)),
                                    ("warm", dict(fire_on=True, fire_spin=False, seed_shifts=[], seed_geo_twin=True, seed_cold_t=99.0))]),
    # training check of the latitude effect on ED's fire (PET band): fire in the windows only, true coordinates
    "c400fw_only_truell": dict(co2="real", repair=True, co2_spinup="fixed", co2_spinup_fixed=400.0, fire_on=True, fire_spin=False, spinup_start=30, latlon_mode="train_true"),
    # c400fwx with the grid latitude from estimate_lat_lon_v2 (ED's fire PET band); cold sites unaffected (fire off)
    "c400fwx2": dict(co2="asis", repair=True, co2_spinup="fixed", co2_spinup_fixed=400.0, seed_shifts=[30.0, 45.0, 60.0], seed_cold_t=8.0,
                     seed_geo_twin=True, latlon_mode="v2",
                     site_overrides=[("cluster", dict(spinup_geo_twin=True, fire_on=True, fire_spin=False, seed_shifts=[], seed_geo_twin=False)),
                                     ("warm", dict(fire_on=True, fire_spin=False, seed_shifts=[], seed_geo_twin=True, seed_cold_t=99.0))]),
    # fire in the windows only with a lower cap on the yearly burnt area (ED default fire_max_disturbance_rate 0.2)
    "c400fw_only_fmax10": dict(co2="real", repair=True, co2_spinup="fixed", co2_spinup_fixed=400.0, fire_on=True, fire_spin=False, spinup_start=30, params_over={"fire_max_disturbance_rate": 0.1}),
    "c400fw_only_fmax05": dict(co2="real", repair=True, co2_spinup="fixed", co2_spinup_fixed=400.0, fire_on=True, fire_spin=False, spinup_start=30, params_over={"fire_max_disturbance_rate": 0.05}),
    # selective fire (11-08 15:10): the Python replica of ED's fire rate fits the organisers' 3353 forested training
    # trajectories best at fp1 ~1e-4..3e-5 (R2 0.32 vs 0.22 at fp1 1; scripts/ed/fire_proxy_calib.py)
    "c400fw_only_fp4_cap10": dict(co2="real", repair=True, co2_spinup="fixed", co2_spinup_fixed=400.0, fire_on=True, fire_spin=False, spinup_start=30, fp1=1e-4, params_over={"fire_max_disturbance_rate": 0.1}),
    "c400fw_only_fp4_cap20": dict(co2="real", repair=True, co2_spinup="fixed", co2_spinup_fixed=400.0, fire_on=True, fire_spin=False, spinup_start=30, fp1=1e-4, params_over={"fire_max_disturbance_rate": 0.2}),
    # 11-09: selective fire in the SPIN-UP as well (the organisers' lookup is "potveg_potfire"; their dry-site y0 are tall-thin)
    "c400fw_all_fp4_cap20": dict(co2="real", repair=True, co2_spinup="fixed", co2_spinup_fixed=400.0, fire_on=True, fire_spin=True, spinup_start=30, fp1=1e-4, params_over={"fire_max_disturbance_rate": 0.2}),
    # c400fwx2 with the yearly burnt-area cap at 0.1 (ED default 0.2 burns ~2x too much on tropical training sites;
    # 0.1 matches their agb ratios: old 0.36 vs truth 0.43, mid 0.44 vs 0.48, young 0.85 vs 0.95)
    "c400fwx3": dict(co2="asis", repair=True, co2_spinup="fixed", co2_spinup_fixed=400.0, seed_shifts=[30.0, 45.0, 60.0], seed_cold_t=8.0,
                     seed_geo_twin=True, latlon_mode="v2", params_over={"fire_max_disturbance_rate": 0.1},
                     site_overrides=[("cluster", dict(spinup_geo_twin=True, fire_on=True, fire_spin=False, seed_shifts=[], seed_geo_twin=False)),
                                     ("warm", dict(fire_on=True, fire_spin=False, seed_shifts=[], seed_geo_twin=True, seed_cold_t=99.0))]),
    "c400_seedtr": dict(co2="real", repair=True, co2_spinup="fixed", co2_spinup_fixed=400.0, seed_shifts=[10.0, 20.0, 30.0], seed_cold_t=99.0),
    # c400fwx3 with the test precipitation floor (6.3517 mm/month, a release artefact) subtracted from the test forcing
    # (windows and test-climate spin-ups; the MERRA-2 twin climatology is clean): dry-month fire ignition as in the truth
    "c400fwx3p": dict(co2="asis", repair=True, co2_spinup="fixed", co2_spinup_fixed=400.0, seed_shifts=[30.0, 45.0, 60.0], seed_cold_t=8.0,
                      seed_geo_twin=True, latlon_mode="v2", params_over={"fire_max_disturbance_rate": 0.1}, pr_floor=6.3517,
                      site_overrides=[("cluster", dict(spinup_geo_twin=True, fire_on=True, fire_spin=False, seed_shifts=[], seed_geo_twin=False)),
                                      ("warm", dict(fire_on=True, fire_spin=False, seed_shifts=[], seed_geo_twin=True, seed_cold_t=99.0))]),
    # c400fwx3p with the real SSP CO2 in the windows (exact decompression of the stored channel, stored = 0.1296 real
    # + 309.74): every test-input artefact undone, as ED saw the clean inputs in training (9-sigma young scenario
    # contrast of 11-07 matches ED with real CO2; the 11-08 01:00 switch to stored CO2 predates the fire protocol)
    "c400fwx3pr": dict(co2="real", repair=True, co2_spinup="fixed", co2_spinup_fixed=400.0, seed_shifts=[30.0, 45.0, 60.0], seed_cold_t=8.0,
                       seed_geo_twin=True, latlon_mode="v2", params_over={"fire_max_disturbance_rate": 0.1}, pr_floor=6.3517,
                       site_overrides=[("cluster", dict(spinup_geo_twin=True, fire_on=True, fire_spin=False, seed_shifts=[], seed_geo_twin=False)),
                                       ("warm", dict(fire_on=True, fire_spin=False, seed_shifts=[], seed_geo_twin=True, seed_cold_t=99.0))]),
    # v4 = c400fwx3pr + selective fire (fp1 1e-4; the replica of ED's fire rate fits 3353 training trajectories best
    # there); cap 0.1 (c400fwx4) or 0.2 (c400fwx4c2); "_all" also lights the window fire at cold sites
    "c400fwx4": dict(co2="real", repair=True, co2_spinup="fixed", co2_spinup_fixed=400.0, seed_shifts=[30.0, 45.0, 60.0], seed_cold_t=8.0,
                     seed_geo_twin=True, latlon_mode="v2", params_over={"fire_max_disturbance_rate": 0.1}, pr_floor=6.3517, fp1=1e-4,
                     site_overrides=[("cluster", dict(spinup_geo_twin=True, fire_on=True, fire_spin=False, seed_shifts=[], seed_geo_twin=False)),
                                     ("warm", dict(fire_on=True, fire_spin=False, seed_shifts=[], seed_geo_twin=True, seed_cold_t=99.0))]),
    # 11-09 evening: EXACT inputs. Every channel mapped back through the exact-twin affine fits (cal_maps), the exact
    # training cell as position, real SSP CO2, fire-free lookup per SCENARIO (the test y0 differs by scenario).
    # Spin-up protocol variants (spin_only = lookup states only, compared with the decoded test y0):
    #   cxs_first  first window decade, CO2 417.5      cxs_win   window mean (2021-2060), CO2 window mean
    #   cxs_cent   window mean moved +20 y along the trend (~ the 2021-2100 mean), CO2 2021-2100 mean
    #   cxs_cent400  as cxs_cent with CO2 400
    "cxs_first": dict(co2="real", repair=False, cal_maps="work/cal/cal_maps_cal.npz", latlon_mode="exact", spin_per_scenario=True, spin_only=True,
                      spinup_both=False, spinup_years=10, co2_spinup="fixed", co2_spinup_fixed=417.5),
    "cxs_win": dict(co2="real", repair=False, cal_maps="work/cal/cal_maps_cal.npz", latlon_mode="exact", spin_per_scenario=True, spin_only=True,
                    spinup_both=False, spinup_years=0, co2_spinup="fixed", co2_spinup_by_scen={"ssp126": 454.1, "ssp585": 512.4}),
    "cxs_cent": dict(co2="real", repair=False, cal_maps="work/cal/cal_maps_cal.npz", latlon_mode="exact", spin_per_scenario=True, spin_only=True,
                     spinup_both=False, spinup_years=0, spinup_shift=20.0, co2_spinup="fixed", co2_spinup_by_scen={"ssp126": 459.0, "ssp585": 680.0}),
    "cxs_cent400": dict(co2="real", repair=False, cal_maps="work/cal/cal_maps_cal.npz", latlon_mode="exact", spin_per_scenario=True, spin_only=True,
                        spinup_both=False, spinup_years=0, spinup_shift=20.0, co2_spinup="fixed", co2_spinup_fixed=400.0),
    # common lookup for both scenarios (the test y0 stocks agree between scenarios to ~1%; only the year-0 fluxes differ)
    "cxs_c10": dict(co2="real", repair=False, cal_maps="work/cal/cal_maps_cal.npz", latlon_mode="exact", spin_only=True,
                    spinup_both=True, spinup_years=10, co2_spinup="fixed", co2_spinup_fixed=400.0),
    "cxs_c10f": dict(co2="real", repair=False, cal_maps="work/cal/cal_maps_cal.npz", latlon_mode="exact", spin_only=True,
                     spinup_both=True, spinup_years=10, co2_spinup="fixed", co2_spinup_fixed=400.0, fire_on=True, fp1=1e-4, params_over={"fire_max_disturbance_rate": 0.2}),
    "cxs_c10f1": dict(co2="real", repair=False, cal_maps="work/cal/cal_maps_cal.npz", latlon_mode="exact", spin_only=True,
                      spinup_both=True, spinup_years=10, co2_spinup="fixed", co2_spinup_fixed=400.0, fire_on=True),
    "cxs_c40": dict(co2="real", repair=False, cal_maps="work/cal/cal_maps_cal.npz", latlon_mode="exact", spin_only=True,
                    spinup_both=True, spinup_years=0, co2_spinup="fixed", co2_spinup_fixed=400.0),
    # 11-09 20:10 FULL exact-input protocol "cx1": lookup = fire-free spin-up on the exact twin's MERRA-2 2011-2020
    # climatology (reproduces the organisers' training lookup, which the test y0 follows at most sites), CO2 400; windows =
    # calibrated test drivers, real SSP CO2, ED potential fire with the training-calibrated fp1 1e-4 / cap 0.2 everywhere
    # (identical to fire off at cold sites).
    "cx1": dict(co2="real", repair=False, cal_maps="work/cal/cal_maps_cal.npz", latlon_mode="exact", spinup_both=True, spinup_years=10,
                co2_spinup="fixed", co2_spinup_fixed=400.0, spinup_geo_twin=True, fire_on=True, fire_spin=False, fp1=1e-4,
                params_over={"fire_max_disturbance_rate": 0.2}),
    # cx0 = cx1 without fire in the windows (is the future truth's moderate collapse fire at all?)
    "cx0": dict(co2="real", repair=False, cal_maps="work/cal/cal_maps_cal.npz", latlon_mode="exact", spinup_both=True, spinup_years=10,
                co2_spinup="fixed", co2_spinup_fixed=400.0, spinup_geo_twin=True, fire_on=False),
    # cx3 = cx1 with the windows' MONTHLY climate (pr, T, soil T: water balance, PET, fire dryness) from the exact twin's
    # MERRA-2 years, the hourly mechanism drivers from the calibrated test forcing
    "cx3": dict(co2="real", repair=False, cal_maps="work/cal/cal_maps_cal.npz", latlon_mode="exact", spinup_both=True, spinup_years=10,
                co2_spinup="fixed", co2_spinup_fixed=400.0, spinup_geo_twin=True, fire_on=True, fire_spin=False, fp1=1e-4,
                params_over={"fire_max_disturbance_rate": 0.2}, window_monthly_twin=True),
    # cx5 = cx3 with ONE averaged MERRA-2 year (the twin's 1982-2021 mean) as the monthly climate of every window year:
    # constant fire dryness per site (ED's readEnvironmentalData falls back to climate_file_avg outside its yearly range)
    "cx5": dict(co2="real", repair=False, cal_maps="work/cal/cal_maps_cal.npz", latlon_mode="exact", spinup_both=True, spinup_years=10, co2_spinup="fixed", co2_spinup_fixed=400.0, spinup_geo_twin=True, fire_on=True, fire_spin=False, fp1=1e-4, params_over={"fire_max_disturbance_rate": 0.2}, window_monthly_twin=True, window_monthly_twin_clim=(0, 40)),
    "cx5b": dict(co2="real", repair=False, cal_maps="work/cal/cal_maps_cal.npz", latlon_mode="exact", spinup_both=True, spinup_years=10, co2_spinup="fixed", co2_spinup_fixed=400.0, spinup_geo_twin=True, fire_on=True, fire_spin=False, fp1=1e-4, params_over={"fire_max_disturbance_rate": 0.2}, window_monthly_twin=True, window_monthly_twin_clim=(30, 40)),
    # 11-10 cx7 = cx3 (window monthly climate file from the exact twin's MERRA-2 years k -> k) with the truth-calibrated
    # fire (ignition factors of the same years, scripts/fire/fire_x.py; needs the build_fire binary). cx7c: constant
    # 40-year climatology for the monthly file and the fire.
    "cx7": dict(co2="real", repair=False, cal_maps="work/cal/cal_maps_cal.npz", latlon_mode="exact", spinup_both=True, spinup_years=10, co2_spinup="fixed", co2_spinup_fixed=400.0, spinup_geo_twin=True, fire_on=True, fire_spin=False, fp1=1.64e-5, params_over={"fire_max_disturbance_rate": 0.2}, window_monthly_twin=True, fire_x="mer", fire_fuel_fix=True),
    "cx7c": dict(co2="real", repair=False, cal_maps="work/cal/cal_maps_cal.npz", latlon_mode="exact", spinup_both=True, spinup_years=10, co2_spinup="fixed", co2_spinup_fixed=400.0, spinup_geo_twin=True, fire_on=True, fire_spin=False, fp1=1.64e-5, params_over={"fire_max_disturbance_rate": 0.2}, window_monthly_twin=True, window_monthly_twin_clim=(0, 40), fire_x="clim40", fire_fuel_fix=True),
    "cx8": dict(co2="real", repair=False, cal_maps="work/cal/cal_maps_cal.npz", latlon_mode="exact", spinup_both=True, spinup_years=10, co2_spinup="fixed", co2_spinup_fixed=400.0, spinup_geo_twin=True, fire_on=True, fire_spin=False, fp1=1.64e-5, params_over={"fire_max_disturbance_rate": 0.2}, window_monthly_twin=True, fire_x="twinx", fire_fuel_fix=True),
    # 11-10 cx9: the monthly climate file and the fire see MERRA-2 1981-2015 recycled cyclically, as the shipped constants
    # MERRA2_START / MERRA2_END imply for 2021+ (2021 -> 1986, ..., 2050 -> 2015, 2051 -> 1981); the board screen puts the
    # minimum of the recycling offset exactly there. cx9t: truth-derived fire factors of the twin instead of the emulator's.
    "cx9": dict(co2="real", repair=False, cal_maps="work/cal/cal_maps_cal.npz", latlon_mode="exact", spinup_both=True, spinup_years=10, co2_spinup="fixed", co2_spinup_fixed=400.0, spinup_geo_twin=True, fire_on=True, fire_spin=False, fp1=1.64e-5, params_over={"fire_max_disturbance_rate": 0.2}, window_monthly_twin=True, window_monthly_twin_shift=5, window_monthly_twin_mod=35, fire_x="mer:cyc35", fire_fuel_fix=True),
    "cx9t": dict(co2="real", repair=False, cal_maps="work/cal/cal_maps_cal.npz", latlon_mode="exact", spinup_both=True, spinup_years=10, co2_spinup="fixed", co2_spinup_fixed=400.0, spinup_geo_twin=True, fire_on=True, fire_spin=False, fp1=1.64e-5, params_over={"fire_max_disturbance_rate": 0.2}, window_monthly_twin=True, window_monthly_twin_shift=5, window_monthly_twin_mod=35, fire_x="twinx:cyc35", fire_fuel_fix=True),
    # 11-10 driver sensitivity (no fire, twin lookup at CO2 400): cx0 with ONE driver group left exactly as delivered, and
    # everything as delivered (ra0 = what the model saw if the release transform was applied before the ED runs).
    "rv_co2": dict(co2="asis", repair=False, cal_maps="work/cal/cal_maps_cal.npz", latlon_mode="exact", spinup_both=True, spinup_years=10, co2_spinup="fixed", co2_spinup_fixed=400.0, spinup_geo_twin=True, fire_on=False),
    "rv_hus": dict(co2="real", repair=False, cal_maps="work/cal/cal_maps_cal.npz", cal_skip=("hus",), latlon_mode="exact", spinup_both=True, spinup_years=10, co2_spinup="fixed", co2_spinup_fixed=400.0, spinup_geo_twin=True, fire_on=False),
    "rv_ta": dict(co2="real", repair=False, cal_maps="work/cal/cal_maps_cal.npz", cal_skip=("ta_h", "ta_m", "tsl"), latlon_mode="exact", spinup_both=True, spinup_years=10, co2_spinup="fixed", co2_spinup_fixed=400.0, spinup_geo_twin=True, fire_on=False),
    "rv_rsds": dict(co2="real", repair=False, cal_maps="work/cal/cal_maps_cal.npz", cal_skip=("rsds",), latlon_mode="exact", spinup_both=True, spinup_years=10, co2_spinup="fixed", co2_spinup_fixed=400.0, spinup_geo_twin=True, fire_on=False),
    "rv_pr": dict(co2="real", repair=False, cal_maps="work/cal/cal_maps_cal.npz", cal_skip=("pr",), latlon_mode="exact", spinup_both=True, spinup_years=10, co2_spinup="fixed", co2_spinup_fixed=400.0, spinup_geo_twin=True, fire_on=False),
    "rv_soil": dict(co2="real", repair=False, cal_maps="work/cal/cal_maps_cal.npz", cal_skip=("soil",), latlon_mode="exact", spinup_both=True, spinup_years=10, co2_spinup="fixed", co2_spinup_fixed=400.0, spinup_geo_twin=True, fire_on=False),
    "ra0": dict(co2="asis", repair=False, latlon_mode="exact", spinup_both=True, spinup_years=10, co2_spinup="fixed", co2_spinup_fixed=400.0, spinup_geo_twin=True, fire_on=False),
    "ra0c": dict(co2="real", repair=False, latlon_mode="exact", spinup_both=True, spinup_years=10, co2_spinup="fixed", co2_spinup_fixed=400.0, spinup_geo_twin=True, fire_on=False),
    # 11-10 "the model saw the drivers as delivered": lookup = spin-up on the window's own first decade exactly as delivered
    # at the stored year-0 CO2 (363.9 ppm), windows as delivered with the stored CO2. rb0 no fire, rb1 truth fire rule on
    # the delivered monthly climate, rb2 the same with the shipped wet-month rule (8.33 mm), rb3 rule and ignition x0.3.
    "rb0": dict(co2="asis", repair=False, latlon_mode="exact", spinup_both=True, spinup_years=10, spinup_start=0, co2_spinup="fixed", co2_spinup_fixed=363.9, fire_on=False),
    "rb1": dict(co2="asis", repair=False, latlon_mode="exact", spinup_both=True, spinup_years=10, spinup_start=0, co2_spinup="fixed", co2_spinup_fixed=363.9, fire_on=True, fire_spin=False, fp1=1.64e-5, params_over={"fire_max_disturbance_rate": 0.2}, fire_x="rel:own", fire_fuel_fix=True),
    "rb2": dict(co2="asis", repair=False, latlon_mode="exact", spinup_both=True, spinup_years=10, spinup_start=0, co2_spinup="fixed", co2_spinup_fixed=363.9, fire_on=True, fire_spin=False, fp1=1.64e-5, params_over={"fire_max_disturbance_rate": 0.2}, fire_x="rel:own:rule8.33", fire_fuel_fix=True),
    # 11-10 00:50 "the FIRE (monthly climate file) sees ta_m / pr / tsl as delivered, everything else is real": cx1 protocol
    # (calibrated hourly drivers, real CO2, real soil, twin lookup) with the monthly channels left as delivered and the
    # truth fire rule on them. cx10 no wet-month rule, cx10r shipped rule 8.33 mm, cx10n no fire.
    "cx10": dict(co2="real", repair=False, cal_maps="work/cal/cal_maps_cal.npz", cal_skip=("monthly",), latlon_mode="exact", spinup_both=True, spinup_years=10, co2_spinup="fixed", co2_spinup_fixed=400.0, spinup_geo_twin=True, fire_on=True, fire_spin=False, fp1=1.64e-5, params_over={"fire_max_disturbance_rate": 0.2}, fire_x="rel:own", fire_fuel_fix=True),
    "cx10r": dict(co2="real", repair=False, cal_maps="work/cal/cal_maps_cal.npz", cal_skip=("monthly",), latlon_mode="exact", spinup_both=True, spinup_years=10, co2_spinup="fixed", co2_spinup_fixed=400.0, spinup_geo_twin=True, fire_on=True, fire_spin=False, fp1=1.64e-5, params_over={"fire_max_disturbance_rate": 0.2}, fire_x="rel:own:rule8.33", fire_fuel_fix=True),
    "cx10n": dict(co2="real", repair=False, cal_maps="work/cal/cal_maps_cal.npz", cal_skip=("monthly",), latlon_mode="exact", spinup_both=True, spinup_years=10, co2_spinup="fixed", co2_spinup_fixed=400.0, spinup_geo_twin=True, fire_on=False),
    # 11-10 01:15 cx11: the board-preferred fire structure (scripts/fire: per-month cap, shipped wet-month rule on the
    # as-delivered precipitation, ignition 30 x the training constant): monthly factors, fp1 4.9e-4, cap 0.43 per month
    # (emulator loss cap 0.30 / survivorship factor 0.7). cx11b: threshold 9 mm, cap 0.357.
    "cx11": dict(co2="real", repair=False, cal_maps="work/cal/cal_maps_cal.npz", cal_skip=("monthly",), latlon_mode="exact", spinup_both=True, spinup_years=10, co2_spinup="fixed", co2_spinup_fixed=400.0, spinup_geo_twin=True, fire_on=True, fire_spin=False, fp1=4.9e-4, params_over={"fire_max_disturbance_rate": 0.43}, fire_x="rel:own:rule8.33:monthly", fire_fuel_fix=True),
    "cx11b": dict(co2="real", repair=False, cal_maps="work/cal/cal_maps_cal.npz", cal_skip=("monthly",), latlon_mode="exact", spinup_both=True, spinup_years=10, co2_spinup="fixed", co2_spinup_fixed=400.0, spinup_geo_twin=True, fire_on=True, fire_spin=False, fp1=4.9e-4, params_over={"fire_max_disturbance_rate": 0.357}, fire_x="rel:own:rule9.0:monthly", fire_fuel_fix=True),
    # 11-10 01:35 cx12: cx11 with ignition 3 x the training constant (the MERRA-dry African sites must not burn: their
    # as-delivered dryness is low) and the cap that keeps the cluster level (emulator loss cap 0.35 -> 0.50).
    "cx12": dict(co2="real", repair=False, cal_maps="work/cal/cal_maps_cal.npz", cal_skip=("monthly",), latlon_mode="exact", spinup_both=True, spinup_years=10, co2_spinup="fixed", co2_spinup_fixed=400.0, spinup_geo_twin=True, fire_on=True, fire_spin=False, fp1=4.9e-5, params_over={"fire_max_disturbance_rate": 0.50}, fire_x="rel:own:rule8.33:monthly", fire_fuel_fix=True),
    # rc0: twin MERRA lookup at the stored year-0 CO2 (no CO2 step into the as-delivered windows), windows as delivered, no fire
    "rc0": dict(co2="asis", repair=False, latlon_mode="exact", spinup_both=True, spinup_years=10, co2_spinup="fixed", co2_spinup_fixed=363.9, spinup_geo_twin=True, fire_on=False),
    "cxs_twin364": dict(co2="real", repair=False, cal_maps="work/cal/cal_maps_cal.npz", latlon_mode="exact", spin_only=True, spinup_both=True, spinup_years=10, co2_spinup="fixed", co2_spinup_fixed=363.9, spinup_geo_twin=True),
    # lookup with fire: twin climatology at 400 ppm and the training fire rule on the climatological dryness (constant factor)
    "cxs_fs": dict(co2="real", repair=False, cal_maps="work/cal/cal_maps_cal.npz", latlon_mode="exact", spin_only=True, spinup_both=True, spinup_years=10, co2_spinup="fixed", co2_spinup_fixed=400.0, spinup_geo_twin=True, fire_on=True, fire_spin=True, fp1=1.64e-5, params_over={"fire_max_disturbance_rate": 0.2}, fire_spin_x="clim10"),
    "cxs_fs3": dict(co2="real", repair=False, cal_maps="work/cal/cal_maps_cal.npz", latlon_mode="exact", spin_only=True, spinup_both=True, spinup_years=10, co2_spinup="fixed", co2_spinup_fixed=400.0, spinup_geo_twin=True, fire_on=True, fire_spin=True, fp1=1.64e-5, params_over={"fire_max_disturbance_rate": 0.2}, fire_spin_x="clim10", fire_spin_scale=3.0),
    # 11-10 04:55 UTC: the TEST LOOKUP is a spin-up on the cell's real MERRA-2 weather 1981-2020, cycled (40-year cycle from
    # 1981 at age 0): the 40-year-cycle lookup reproduces the test_y0 age profiles of non-fire sites to 1-2 % incl. their
    # wiggles (the 2011-2020 climatology lookup: smooth and 8 % too heavy). cx14 = cx12 with that lookup (build_fire3).
    "cx14": dict(co2="real", repair=False, cal_maps="work/cal/cal_maps_cal.npz", cal_skip=("monthly",), latlon_mode="exact", spinup_both=True, spinup_years=10, co2_spinup="fixed", co2_spinup_fixed=400.0, spinup_geo_twin=True, spinup_cycle=True, spinup_twin_cycle=(0, 40),
                 fire_on=True, fire_spin=False, fp1=4.9e-5, params_over={"fire_max_disturbance_rate": 0.50}, fire_x="rel:own:rule8.33:monthly", fire_fuel_fix=True),
    # cycled-MERRA lookup WITH the shipped wet-month fire (monthly factors of the 40 MERRA years, per-month cap); parameters via CG_OVERRIDE
    "cxs_tcf": dict(co2="real", repair=False, cal_maps="work/cal/cal_maps_cal.npz", latlon_mode="exact", spin_only=True, spinup_both=True, spinup_years=10, co2_spinup="fixed", co2_spinup_fixed=400.0, spinup_geo_twin=True, spinup_cycle=True, spinup_twin_cycle=(0, 40),
                    fire_on=True, fire_spin=True, fp1=4.9e-5, params_over={"fire_max_disturbance_rate": 0.50}, fire_spin_x="merm:rule8.33"),
    # cx15 = cx14 with the lookup at 415 ppm: on 13 test sites the cycled lookup at 400 ppm is 5 % short in height and 3-6 % light,
    # at 415 ppm 2-3 % short and +1 % in agb, at 430 ppm -1 % / +4-5 % (work/edv3/sp_tc40_c415, _c430)
    "cx15": dict(co2="real", repair=False, cal_maps="work/cal/cal_maps_cal.npz", cal_skip=("monthly",), latlon_mode="exact", spinup_both=True, spinup_years=10, co2_spinup="fixed", co2_spinup_fixed=415.0, spinup_geo_twin=True, spinup_cycle=True, spinup_twin_cycle=(0, 40),
                 fire_on=True, fire_spin=False, fp1=4.9e-5, params_over={"fire_max_disturbance_rate": 0.50}, fire_x="rel:own:rule8.33:monthly", fire_fuel_fix=True),
    # cx16 = ALL window drivers calibrated (real), no fire, cycled-MERRA lookup at 415 ppm. 11-10 05:50 UTC: at 55 temperate cells
    # whose training leaf state follows the 10 C December rule, 41 % of the test start states are leafy; the December T of the
    # early window years predicts 36-42 % if ED saw the CALIBRATED monthly T and 8-15 % if it saw the delivered one. So the
    # monthly climate file is real as well ("as delivered" was an artefact of the fire-emulator fits). Fire is added per site
    # class once the rule is calibrated on the lookup profiles (real MERRA units).
    "cx16": dict(co2="real", repair=False, cal_maps="work/cal/cal_maps_cal.npz", latlon_mode="exact", spinup_both=True, spinup_years=10, co2_spinup="fixed", co2_spinup_fixed=415.0, spinup_geo_twin=True, spinup_cycle=True, spinup_twin_cycle=(0, 40), fire_on=False),
    # cx17 = cx16 + the SHIPPED fire structure in real units, in the lookup AND the windows: monthly ignition fuel x fp1 x (DI / 30000)^10,
    # zero in months above 8.33 mm, mean of the 12 months, THEN the yearly cap 0.2 (env ED_FIRE_X_YEARCAP=1, binary build_fire6), with a
    # much weaker ignition than training (fp1 1e-7 vs 1.64e-5): on the lookup this reproduces the event timing of y0 at 5147 (dips at
    # the ages that end on MERRA 2020) and leaves the African miombo sites unburned (work/edv3/fc_q1..q4). Variants via CG_OVERRIDE.
    "cx17": dict(co2="real", repair=False, cal_maps="work/cal/cal_maps_cal.npz", latlon_mode="exact", spinup_both=True, spinup_years=10, co2_spinup="fixed", co2_spinup_fixed=415.0, spinup_geo_twin=True, spinup_cycle=True, spinup_twin_cycle=(0, 40),
                 fire_on=True, fire_spin=True, fp1=1e-7, params_over={"fire_max_disturbance_rate": 0.2}, fire_spin_x="merm:rule8.33", fire_x="fut:own:rule8.33:monthly", fire_fuel_fix=True),
    # lookup on the exact twin's real MERRA-2 weather years, cycled (2011-2020 / all 40), instead of their climatology
    "cxs_tc10": dict(co2="real", repair=False, cal_maps="work/cal/cal_maps_cal.npz", latlon_mode="exact", spin_only=True, spinup_both=True, spinup_years=10, co2_spinup="fixed", co2_spinup_fixed=400.0, spinup_geo_twin=True, fire_on=False, spinup_cycle=True, spinup_twin_cycle=(30, 40)),
    "cxs_tc40": dict(co2="real", repair=False, cal_maps="work/cal/cal_maps_cal.npz", latlon_mode="exact", spin_only=True, spinup_both=True, spinup_years=10, co2_spinup="fixed", co2_spinup_fixed=400.0, spinup_geo_twin=True, fire_on=False, spinup_cycle=True, spinup_twin_cycle=(0, 40)),
    # lookup with the MONTHLY climate as delivered (first-decade / 40-year mean of the test's own monthly channels) and the hourly drivers of the exact twin
    "cxs_tm": dict(co2="real", repair=False, cal_maps="work/cal/cal_maps_cal.npz", cal_skip=("monthly",), latlon_mode="exact", spin_only=True, spinup_both=True, spinup_years=10, co2_spinup="fixed", co2_spinup_fixed=400.0, spin_swap_twin="ta_h,hus,rsds,wind", fire_on=False),
    "cxs_tm40": dict(co2="real", repair=False, cal_maps="work/cal/cal_maps_cal.npz", cal_skip=("monthly",), latlon_mode="exact", spin_only=True, spinup_both=True, spinup_years=40, co2_spinup="fixed", co2_spinup_fixed=400.0, spin_swap_twin="ta_h,hus,rsds,wind", fire_on=False),
    # lookup with the SHIPPED fire (wet-month rule 8.33 mm on the twin's climatological months, per-month cap as cx12)
    "cxs_fw": dict(co2="real", repair=False, cal_maps="work/cal/cal_maps_cal.npz", latlon_mode="exact", spin_only=True, spinup_both=True, spinup_years=10, co2_spinup="fixed", co2_spinup_fixed=400.0, spinup_geo_twin=True, fire_on=True, fire_spin=True, fp1=4.9e-5, params_over={"fire_max_disturbance_rate": 0.50}, fire_spin_x="climm10:rule8.33"),
    "cxs_fw2": dict(co2="real", repair=False, cal_maps="work/cal/cal_maps_cal.npz", latlon_mode="exact", spin_only=True, spinup_both=True, spinup_years=10, co2_spinup="fixed", co2_spinup_fixed=400.0, spinup_geo_twin=True, fire_on=True, fire_spin=True, fp1=4.9e-5, params_over={"fire_max_disturbance_rate": 0.25}, fire_spin_x="climm10:rule8.33"),
    # lookup test of the as-delivered world: spin-up only (rows spin_* vs y0_*), CO2 at the stored year-0 level / at 400
    "cxs_rb": dict(co2="asis", repair=False, latlon_mode="exact", spin_only=True, spinup_both=True, spinup_years=10, spinup_start=0, co2_spinup="fixed", co2_spinup_fixed=363.9),
    "cxs_rb400": dict(co2="asis", repair=False, latlon_mode="exact", spin_only=True, spinup_both=True, spinup_years=10, spinup_start=0, co2_spinup="fixed", co2_spinup_fixed=400.0),
    "cxs_rb40": dict(co2="asis", repair=False, latlon_mode="exact", spin_only=True, spinup_both=True, spinup_years=40, spinup_start=0, co2_spinup="fixed", co2_spinup_fixed=363.9),
    # cx3 with shifted year mapping (window year k reads MERRA year k + shift, cyclic)
    "cx3s20": dict(co2="real", repair=False, cal_maps="work/cal/cal_maps_cal.npz", latlon_mode="exact", spinup_both=True, spinup_years=10, co2_spinup="fixed", co2_spinup_fixed=400.0, spinup_geo_twin=True, fire_on=True, fire_spin=False, fp1=1e-4, params_over={"fire_max_disturbance_rate": 0.2}, window_monthly_twin=True, window_monthly_twin_shift=20),
    # 11-09 20:40 fire selectivity on TRAINING cells (train_site.py; lookup = the cell's 2011-2020 climatology, true position):
    # the organisers' yearly losses are on/off events of ~12-14% per burning year; fp1 sets which site-years ignite.
    "trf4": dict(co2="asis", repair=False, co2_spinup="fixed", co2_spinup_fixed=400.0, fire_on=True, fire_spin=False, spinup_start=30, latlon_mode="train_true", fp1=1e-4, params_over={"fire_max_disturbance_rate": 0.2}),
    "trf5": dict(co2="asis", repair=False, co2_spinup="fixed", co2_spinup_fixed=400.0, fire_on=True, fire_spin=False, spinup_start=30, latlon_mode="train_true", fp1=1e-5, params_over={"fire_max_disturbance_rate": 0.2}),
    # 11-10 truth-calibrated fire (notes 11-10): ignition = fuel x fp1 x X[year] with X from scripts/fire (rolling
    # 12-slot monthly array, January index 0, no wet-month rule), ED's own cap 0.2 and fire survivorship 0.3.
    "trx": dict(co2="asis", repair=False, co2_spinup="fixed", co2_spinup_fixed=400.0, fire_on=True, fire_spin=False, spinup_start=30, latlon_mode="train_true", fp1=1.64e-5, params_over={"fire_max_disturbance_rate": 0.2}, fire_x="roll"),
    "trxt": dict(co2="asis", repair=False, co2_spinup="fixed", co2_spinup_fixed=400.0, fire_on=True, fire_spin=False, spinup_start=30, latlon_mode="train_true", fp1=1.64e-5, params_over={"fire_max_disturbance_rate": 0.2}, fire_x="truthx", fire_fuel_fix=True),
    "trxf": dict(co2="asis", repair=False, co2_spinup="fixed", co2_spinup_fixed=400.0, fire_on=True, fire_spin=False, spinup_start=30, latlon_mode="train_true", fp1=1.64e-5, params_over={"fire_max_disturbance_rate": 0.2}, fire_x="roll", fire_fuel_fix=True),
    "trx2": dict(co2="asis", repair=False, co2_spinup="fixed", co2_spinup_fixed=400.0, fire_on=True, fire_spin=False, spinup_start=30, latlon_mode="train_true", fp1=1.0e-5, params_over={"fire_max_disturbance_rate": 0.2}, fire_x="roll"),
    "trx3": dict(co2="asis", repair=False, co2_spinup="fixed", co2_spinup_fixed=400.0, fire_on=True, fire_spin=False, spinup_start=30, latlon_mode="train_true", fp1=2.5e-5, params_over={"fire_max_disturbance_rate": 0.2}, fire_x="roll"),
    "trf6": dict(co2="asis", repair=False, co2_spinup="fixed", co2_spinup_fixed=400.0, fire_on=True, fire_spin=False, spinup_start=30, latlon_mode="train_true", fp1=1e-6, params_over={"fire_max_disturbance_rate": 0.2}),
    "trf7": dict(co2="asis", repair=False, co2_spinup="fixed", co2_spinup_fixed=400.0, fire_on=True, fire_spin=False, spinup_start=30, latlon_mode="train_true", fp1=1e-7, params_over={"fire_max_disturbance_rate": 0.2}),
    "trf8": dict(co2="asis", repair=False, co2_spinup="fixed", co2_spinup_fixed=400.0, fire_on=True, fire_spin=False, spinup_start=30, latlon_mode="train_true", fp1=1e-8, params_over={"fire_max_disturbance_rate": 0.2}),
    # diagnostics: spin-up climatology (or single driver groups of it) from the exact twin's MERRA-2 2011-2020 decade
    "cxs_twin": dict(co2="real", repair=False, cal_maps="work/cal/cal_maps_cal.npz", latlon_mode="exact", spin_only=True, spinup_both=True, spinup_years=10, co2_spinup="fixed", co2_spinup_fixed=400.0, spinup_geo_twin=True),
    # which MERRA-2 period / CO2 level reproduces the TEST lookup (it follows the training lookup but not exactly)?
    "cxs_tw0040": dict(co2="real", repair=False, cal_maps="work/cal/cal_maps_cal.npz", latlon_mode="exact", spin_only=True, spinup_both=True, spinup_years=10, co2_spinup="fixed", spinup_geo_twin=True, co2_spinup_fixed=400.0, twin_years=(0, 40)),
    "cxs_tw0010": dict(co2="real", repair=False, cal_maps="work/cal/cal_maps_cal.npz", latlon_mode="exact", spin_only=True, spinup_both=True, spinup_years=10, co2_spinup="fixed", spinup_geo_twin=True, co2_spinup_fixed=400.0, twin_years=(0, 10)),
    "cxs_tw1020": dict(co2="real", repair=False, cal_maps="work/cal/cal_maps_cal.npz", latlon_mode="exact", spin_only=True, spinup_both=True, spinup_years=10, co2_spinup="fixed", spinup_geo_twin=True, co2_spinup_fixed=400.0, twin_years=(10, 20)),
    "cxs_tw2030": dict(co2="real", repair=False, cal_maps="work/cal/cal_maps_cal.npz", latlon_mode="exact", spin_only=True, spinup_both=True, spinup_years=10, co2_spinup="fixed", spinup_geo_twin=True, co2_spinup_fixed=400.0, twin_years=(20, 30)),
    "cxs_tw3040c417": dict(co2="real", repair=False, cal_maps="work/cal/cal_maps_cal.npz", latlon_mode="exact", spin_only=True, spinup_both=True, spinup_years=10, co2_spinup="fixed", spinup_geo_twin=True, co2_spinup_fixed=417.0, twin_years=(30, 40)),
    "cxs_tw3040c370": dict(co2="real", repair=False, cal_maps="work/cal/cal_maps_cal.npz", latlon_mode="exact", spin_only=True, spinup_both=True, spinup_years=10, co2_spinup="fixed", spinup_geo_twin=True, co2_spinup_fixed=370.0, twin_years=(30, 40)),
    "cxs_sw_tah": dict(co2="real", repair=False, cal_maps="work/cal/cal_maps_cal.npz", latlon_mode="exact", spin_only=True, spinup_both=True, spinup_years=10, co2_spinup="fixed", co2_spinup_fixed=400.0, spin_swap_twin="ta_h"),
    "cxs_sw_tsl": dict(co2="real", repair=False, cal_maps="work/cal/cal_maps_cal.npz", latlon_mode="exact", spin_only=True, spinup_both=True, spinup_years=10, co2_spinup="fixed", co2_spinup_fixed=400.0, spin_swap_twin="tsl,soil_t_hourly"),
    "cxs_sw_hus": dict(co2="real", repair=False, cal_maps="work/cal/cal_maps_cal.npz", latlon_mode="exact", spin_only=True, spinup_both=True, spinup_years=10, co2_spinup="fixed", co2_spinup_fixed=400.0, spin_swap_twin="hus"),
    "cxs_sw_rsds": dict(co2="real", repair=False, cal_maps="work/cal/cal_maps_cal.npz", latlon_mode="exact", spin_only=True, spinup_both=True, spinup_years=10, co2_spinup="fixed", co2_spinup_fixed=400.0, spin_swap_twin="rsds"),
    "cxs_sw_pr": dict(co2="real", repair=False, cal_maps="work/cal/cal_maps_cal.npz", latlon_mode="exact", spin_only=True, spinup_both=True, spinup_years=10, co2_spinup="fixed", co2_spinup_fixed=400.0, spin_swap_twin="pr,ta_m"),
    "cxs_win400": dict(co2="real", repair=False, cal_maps="work/cal/cal_maps_cal.npz", latlon_mode="exact", spin_per_scenario=True, spin_only=True,
                       spinup_both=False, spinup_years=0, co2_spinup="fixed", co2_spinup_fixed=400.0),
    "cxs_cent40": dict(co2="real", repair=False, cal_maps="work/cal/cal_maps_cal.npz", latlon_mode="exact", spin_per_scenario=True, spin_only=True,
                       spinup_both=False, spinup_years=0, spinup_shift=40.0, co2_spinup="fixed", co2_spinup_fixed=400.0),
    # 11-09: v4 with the site positions (ED latitude and the geographic twin of the seeds / cluster spin-up) from the
    # NEX-GDDP weather-anomaly geolocation (test_latlon_nex3.npz) instead of estimate_lat_lon_v2 (longitudes ~4-5 deg off)
    "c400fwx4c2nx": dict(co2="real", repair=True, co2_spinup="fixed", co2_spinup_fixed=400.0, seed_shifts=[30.0, 45.0, 60.0], seed_cold_t=8.0,
                         seed_geo_twin=True, latlon_mode="nex3", params_over={"fire_max_disturbance_rate": 0.2}, pr_floor=6.3517, fp1=1e-4,
                         site_overrides=[("cluster", dict(spinup_geo_twin=True, fire_on=True, fire_spin=False, seed_shifts=[], seed_geo_twin=False)),
                                         ("warm", dict(fire_on=True, fire_spin=False, seed_shifts=[], seed_geo_twin=True, seed_cold_t=99.0))]),
    "c400fwx4c2": dict(co2="real", repair=True, co2_spinup="fixed", co2_spinup_fixed=400.0, seed_shifts=[30.0, 45.0, 60.0], seed_cold_t=8.0,
                       seed_geo_twin=True, latlon_mode="v2", params_over={"fire_max_disturbance_rate": 0.2}, pr_floor=6.3517, fp1=1e-4,
                       site_overrides=[("cluster", dict(spinup_geo_twin=True, fire_on=True, fire_spin=False, seed_shifts=[], seed_geo_twin=False)),
                                       ("warm", dict(fire_on=True, fire_spin=False, seed_shifts=[], seed_geo_twin=True, seed_cold_t=99.0))]),
    # c400fwx4c2 WITHOUT the artefact corrections (stored-channel CO2, precipitation floor kept): paired board test
    "c400fwx4c2asis": dict(co2="asis", repair=True, co2_spinup="fixed", co2_spinup_fixed=400.0, seed_shifts=[30.0, 45.0, 60.0], seed_cold_t=8.0,
                           seed_geo_twin=True, latlon_mode="v2", params_over={"fire_max_disturbance_rate": 0.2}, fp1=1e-4,
                           site_overrides=[("cluster", dict(spinup_geo_twin=True, fire_on=True, fire_spin=False, seed_shifts=[], seed_geo_twin=False)),
                                           ("warm", dict(fire_on=True, fire_spin=False, seed_shifts=[], seed_geo_twin=True, seed_cold_t=99.0))]),
    # 11-09: the v4 / v4asis protocols with density-scaled seeds (every cohort's nindivs x y0_agb / seed_agb at each seed age)
    "c400fwx4c2ds": dict(co2="real", repair=True, co2_spinup="fixed", co2_spinup_fixed=400.0, seed_shifts=[30.0, 45.0, 60.0], seed_cold_t=8.0,
                         seed_geo_twin=True, latlon_mode="v2", params_over={"fire_max_disturbance_rate": 0.2}, pr_floor=6.3517, fp1=1e-4, density_scale=True,
                         site_overrides=[("cluster", dict(spinup_geo_twin=True, fire_on=True, fire_spin=False, seed_shifts=[], seed_geo_twin=False)),
                                         ("warm", dict(fire_on=True, fire_spin=False, seed_shifts=[], seed_geo_twin=True, seed_cold_t=99.0))]),
    "c400fwx4c2asisds": dict(co2="asis", repair=True, co2_spinup="fixed", co2_spinup_fixed=400.0, seed_shifts=[30.0, 45.0, 60.0], seed_cold_t=8.0,
                             seed_geo_twin=True, latlon_mode="v2", params_over={"fire_max_disturbance_rate": 0.2}, fp1=1e-4, density_scale=True,
                             site_overrides=[("cluster", dict(spinup_geo_twin=True, fire_on=True, fire_spin=False, seed_shifts=[], seed_geo_twin=False)),
                                             ("warm", dict(fire_on=True, fire_spin=False, seed_shifts=[], seed_geo_twin=True, seed_cold_t=99.0))]),
    # 11-09: v4 / v4asis with the calibrated selective fire also in the SPIN-UP (fire-shaped, tall-thin dry-site seeds)
    "c400fwx5": dict(co2="real", repair=True, co2_spinup="fixed", co2_spinup_fixed=400.0, seed_shifts=[30.0, 45.0, 60.0], seed_cold_t=8.0,
                     seed_geo_twin=True, latlon_mode="v2", params_over={"fire_max_disturbance_rate": 0.2}, pr_floor=6.3517, fp1=1e-4,
                     site_overrides=[("cluster", dict(spinup_geo_twin=True, fire_on=True, fire_spin=True, seed_shifts=[], seed_geo_twin=False)),
                                     ("warm", dict(fire_on=True, fire_spin=True, seed_shifts=[], seed_geo_twin=True, seed_cold_t=99.0))]),
    "c400fwx5asis": dict(co2="asis", repair=True, co2_spinup="fixed", co2_spinup_fixed=400.0, seed_shifts=[30.0, 45.0, 60.0], seed_cold_t=8.0,
                         seed_geo_twin=True, latlon_mode="v2", params_over={"fire_max_disturbance_rate": 0.2}, fp1=1e-4,
                         site_overrides=[("cluster", dict(spinup_geo_twin=True, fire_on=True, fire_spin=True, seed_shifts=[], seed_geo_twin=False)),
                                         ("warm", dict(fire_on=True, fire_spin=True, seed_shifts=[], seed_geo_twin=True, seed_cold_t=99.0))]),
    # 11-09: v4 with the test temperatures shifted to the NEX-GDDP GFDL-ESM4 level (make_site_inputs ta_nex)
    "c400fwx4c2tn": dict(co2="real", repair=True, co2_spinup="fixed", co2_spinup_fixed=400.0, seed_shifts=[30.0, 45.0, 60.0], seed_cold_t=8.0,
                         seed_geo_twin=True, latlon_mode="v2", params_over={"fire_max_disturbance_rate": 0.2}, pr_floor=6.3517, fp1=1e-4, ta_nex=True,
                         site_overrides=[("cluster", dict(spinup_geo_twin=True, fire_on=True, fire_spin=False, seed_shifts=[], seed_geo_twin=False)),
                                         ("warm", dict(fire_on=True, fire_spin=False, seed_shifts=[], seed_geo_twin=True, seed_cold_t=99.0))]),
    "c400fwx4c2tqn": dict(co2="real", repair=True, co2_spinup="fixed", co2_spinup_fixed=400.0, seed_shifts=[30.0, 45.0, 60.0], seed_cold_t=8.0,
                          seed_geo_twin=True, latlon_mode="v2", params_over={"fire_max_disturbance_rate": 0.2}, pr_floor=6.3517, fp1=1e-4, ta_nex=True, hus_nex=True,
                          site_overrides=[("cluster", dict(spinup_geo_twin=True, fire_on=True, fire_spin=False, seed_shifts=[], seed_geo_twin=False)),
                                          ("warm", dict(fire_on=True, fire_spin=False, seed_shifts=[], seed_geo_twin=True, seed_cold_t=99.0))]),
    "c400fwx4_all": dict(co2="real", repair=True, co2_spinup="fixed", co2_spinup_fixed=400.0, seed_shifts=[30.0, 45.0, 60.0], seed_cold_t=8.0,
                         seed_geo_twin=True, latlon_mode="v2", params_over={"fire_max_disturbance_rate": 0.1}, pr_floor=6.3517, fp1=1e-4,
                         fire_on=True, fire_spin=False,
                         site_overrides=[("cluster", dict(spinup_geo_twin=True, fire_on=True, fire_spin=False, seed_shifts=[], seed_geo_twin=False)),
                                         ("warm", dict(fire_on=True, fire_spin=False, seed_shifts=[], seed_geo_twin=True, seed_cold_t=99.0))]),
}
RESULT_COLS = ["age", "height_m", "agb_kgC", "soil", "lai", "gpp", "npp", "rh"]


def config_args(name: str, **over) -> argparse.Namespace:
    cfg = dict(CONFIGS[name])
    for k, v in dict(spinup_years=10, spinup_both=True).items():   # the historical defaults; a config may set its own
        cfg.setdefault(k, v)
    cfg.update(over)
    if os.environ.get("CG_OVERRIDE"):      # e.g. CG_OVERRIDE="co2_spinup_fixed=415.0;fp1=1e-4": config keys overridden for a test run
        import ast
        for item in os.environ["CG_OVERRIDE"].split(";"):
            if item.strip():
                k, v = item.split("=", 1); cfg[k.strip()] = ast.literal_eval(v.strip())
    args = msi.default_args(**cfg)
    if name == "c417":
        j = json.load(open(args.co2_json))
        v = j["decompressed"]["ssp126"]["site_mean_decompressed"]
        args.co2_spinup_fixed = float(2 * v[0] - v[1])
    return args


def parse_ed(path: Path) -> dict[float, dict[str, float]]:
    out: dict[float, dict[str, float]] = {}
    with open(path) as fh:
        for line in fh:
            toks = line.split()
            if len(toks) < 4 or toks[1] not in ("t=", "time"):
                continue
            d = {}
            kv = toks[3:]
            for i in range(0, len(kv) - 1, 2):
                try:
                    d[kv[i]] = float(kv[i + 1])
                except ValueError:
                    pass
            out[round(float(toks[2]), 4)] = d
    return out


def site_files(outdir: Path, exp: str) -> dict[str, Path]:
    files = {}
    for kind in ("biomass", "cfluxes", "sc"):
        cands = [p for p in outdir.glob(f"{exp}.*.{kind}") if "natural" not in p.name and "patch" not in p.name and ".aa." not in p.name]
        if not cands:
            raise FileNotFoundError(f"{exp}.*.{kind} in {outdir}")
        files[kind] = cands[0]
    wat = [p for p in outdir.glob(f"{exp}.*.natural.water") if "patch" not in p.name]
    if wat:
        files["water"] = wat[0]      # monthly site water balance (theta, w, rain, uptake, demand), diagnostics only
    return files


def monthly_diag(tabs: dict[str, dict], t: float) -> dict[str, float]:
    """Monthly rows before the yearly record at time t (row m holds the state at the END of month m; its leaf status was
    decided from the soil moisture at the end of month m - 1): December / November leaf area and fluxes, autumn theta."""
    out = {}
    for lab, back in (("m12", 1), ("m11", 2), ("m10", 3)):
        tt = round(t - back / 12.0, 4)
        b, c, w = tabs["biomass"].get(tt, {}), tabs["cfluxes"].get(tt, {}), tabs.get("water", {}).get(tt, {})
        out[f"lai_{lab}"] = b.get("lai", np.nan); out[f"agb_{lab}"] = b.get("tagb", np.nan)
        out[f"gpp_{lab}"] = c.get("gpp", np.nan); out[f"npp_{lab}"] = c.get("npp", np.nan); out[f"th_{lab}"] = w.get("theta=", np.nan)
        if lab == "m12":     # the organisers' yearly record IS the December row (leaf state of the training records, 11-10)
            out["h_m12"] = b.get("havg", np.nan); out["soil_m12"] = tabs["sc"].get(tt, {}).get("total_soil_c", np.nan); out["rh_m12"] = c.get("rh", np.nan)
    return out


def state(tabs: dict[str, dict], t: float) -> dict[str, float]:
    """State at time t (start of the step), fluxes of the preceding month (December when t is a whole year)."""
    b, s = tabs["biomass"].get(round(t, 4), {}), tabs["sc"].get(round(t, 4), {})
    c = tabs["cfluxes"].get(round(t - 1 / 12.0, 4), {})
    return {"height_m": b.get("havg", np.nan), "agb_kgC": b.get("tagb", np.nan), "soil": s.get("total_soil_c", np.nan), "lai": b.get("lai", np.nan),
            "gpp": c.get("gpp", np.nan), "npp": c.get("npp", np.nan), "rh": c.get("rh", np.nan)}


def write_cfgs(run: Path, inp: dict[str, Path], params_src: Path, pft_src: Path, fire_on: bool = False, fp1: float = 1.0, params_over: dict | None = None, spin_cycle: bool = False, fire_spin: bool | None = None,
               spin_sc: str = "ssp126") -> None:
    shutil.copy(pft_src, run / "ED_pft.defaults.cfg")
    raw = params_src.read_text()

    def fire_text(on: bool) -> str:
        if on:   # ED's own fuel/dryness fire model (the seed protocol's "potential fire"); no GFED input exists for the test
            return raw.replace("fire_gfed                = 1;", "fire_gfed                = 0;")
        return raw.replace("fire_off                 = 0;", "fire_off                 = 1;")

    base = fire_text(fire_on)                                       # window parameters
    spin_base = base if fire_spin is None else fire_text(fire_spin)  # spin-up parameters (fire_spin overrides fire_on)
    def tweak(text: str) -> str:
        if fp1 != 1.0:
            assert text.count("fp1                      = 1.0;") == 1
            text = text.replace("fp1                      = 1.0;", f"fp1                      = {fp1};")
        for key, val in (params_over or {}).items():   # generic ED_params overrides, e.g. {"m3": 10.0, "water1": 80.0}
            import re
            text, n = re.subn(r"(?m)^(%s\s*=\s*)[^;]*;" % re.escape(key), r"\g<1>%s;" % val, text)
            assert n == 1, f"param {key}: {n} matches"
        return text

    base = tweak(base); spin_base = tweak(spin_base)
    (run / "ED_params_spin.cfg").write_text(spin_base)
    win = base.replace("restart              = 0;", "restart              = 1;").replace("tmax                 = 500.0;", f"tmax                 = {T_END:.1f};")
    win = win.replace('old_restart_exp_name = "test_glb_newPFT_soilgrids_v15_potveg_potfire";', 'old_restart_exp_name = "spinup";')
    (run / "ED_params_win.cfg").write_text(win)
    for key, sc in (("spin", spin_sc), ("w126", "ssp126"), ("w585", "ssp585")):
        d = str(inp[sc].resolve())
        io_text = msi.ED_IO_TEMPLATE.format(d=d, y0=1981, fire_off=1)
        io_text = io_text.replace(f'output_base_path = "{d}/ED_outputs/";', f'output_base_path = "{run.resolve()}/out/";')
        if key != "spin":
            io_text = io_text.replace('which_mech_to_use = "cgb_spinup";', 'which_mech_to_use = "cgb_window";')
            io_text = io_text.replace("print_system_state = 1;", "print_system_state = 0;")
        elif spin_cycle:
            io_text = io_text.replace('which_mech_to_use = "cgb_spinup";', 'which_mech_to_use = "cgb_spinw";')   # yearly-mech spin-up on cycled weather
        (run / f"ED_io_{key}.cfg").write_text(io_text)
        params = "ED_params_spin.cfg" if key == "spin" else "ED_params_win.cfg"
        (run / f"models_{key}.cfg").write_text(
            'ed = {\n   io_default = "ED_io_%s.cfg"; io_alternate = "ED_io_%s.cfg";\n   params_default = "%s"; params_alternate = "%s";\n'
            '   pfts_default = "ED_pft.defaults.cfg"; pfts_alternate = "ED_pft.defaults.cfg";\n};\n' % (key, key, params, params))
    shutil.copy(inp["ssp126"] / "sois.txt", run / "sois.txt")


def run_ed(run: Path, exp: str, models: str, env_extra: dict[str, str], log: Path) -> float:
    env = dict(os.environ); env.update(ENV_COMMON); env.update(env_extra)
    t0 = time.time()
    with open(log, "w") as fh:
        rc = subprocess.call([str(EDLU), exp, models], cwd=str(run), env=env, stdout=fh, stderr=subprocess.STDOUT)
    if rc != 0:
        raise RuntimeError(f"edlu {exp} failed with {rc}; see {log}")
    return time.time() - t0


_GEO_BASE: dict[str, frozenset] = {}


def geo_base_sites(spec: str) -> frozenset:
    """Sites whose BASE spin-up uses the geographic twin (config key ``geo_base``): "cluster" = the collapse cluster
    (work/collapse_cluster_labels.npy == 0), otherwise a path to a file with one site id per line."""
    if spec not in _GEO_BASE:
        if spec == "cluster":
            _GEO_BASE[spec] = frozenset(int(s) for s in np.flatnonzero(np.load(ROOT / "work" / "collapse_cluster_labels.npy") == 0))
        else:
            _GEO_BASE[spec] = frozenset(int(l) for l in open(spec) if l.strip())
    return _GEO_BASE[spec]

def site_in_group(site: int, spec: str, in_args: argparse.Namespace) -> bool:
    """Group membership for config key site_overrides: "cluster" (collapse cluster), "warm" / "cold" (non-cluster, first-
    decade T at or above / below seed_cold_t, default 8 C), otherwise a file with one site id per line."""
    if spec == "cluster":
        return site in geo_base_sites("cluster")
    if spec in ("warm", "cold"):
        if site in geo_base_sites("cluster"):
            return False
        cold = first_decade_temperature(site) < float(getattr(in_args, "seed_cold_t", 8.0))
        return cold if spec == "cold" else not cold
    return site in geo_base_sites(spec)

def first_decade_temperature(site: int) -> float:
    """Mean monthly air temperature (deg C) of the first 10 window years, both scenarios."""
    vals = []
    for sc in ("ssp126", "ssp585"):
        x, _ = msi.io.open_test(sc)
        vals.append(float(np.asarray(x[site, :10, :, 0], dtype=np.float64).mean()) - 273.15)
    return float(np.mean(vals))


def seed_loss(st: dict[str, float], yh: float, ya: float) -> float:
    """Squared offset-log misfit of a spin-up state against the decoded y0 (height eps 0.1 m, agb eps 0.01 kgC)."""
    h, a = st.get("height_m", np.nan), st.get("agb_kgC", np.nan)
    if not (np.isfinite(h) and np.isfinite(a)):
        return float("inf")
    return float(np.log((max(h, 0.0) + 0.1) / (yh + 0.1)) ** 2 + np.log((max(a, 0.0) + 0.01) / (ya + 0.01)) ** 2)


def select_seeds(site: int, root: Path, in_args: argparse.Namespace, run: Path, spin: dict, metas: dict, floors: np.ndarray,
                 latlon: tuple[float, float], env_spin: dict[str, str], timings: dict, keep: bool) -> tuple[dict, dict]:
    """Per-age seed selection among trend-shifted spin-ups (config keys ``seed_shifts``, ``seed_cold_t``; with
    ``seed_geo_twin`` also the geographic-twin spin-up, key "geo").

    For a vegetated site whose first-decade mean temperature is below ``seed_cold_t``, extra spin-ups are run with the
    climatology moved ``s`` years along the window trend (``spinup_shift = s`` for each s in ``seed_shifts``); for each
    seed age the spin-up state closest to the decoded y0 (mean of both scenarios, ``seed_loss``) is chosen and its
    restart dump (.css/.pss) is copied over the base dump, so the windows start from the best-matching stand.
    Returns ({age: chosen shift}, {shift: parsed spin-up tables}); empty dicts when the site does not qualify."""
    shifts = [float(v) for v in (getattr(in_args, "seed_shifts", None) or [])]
    geo = bool(getattr(in_args, "seed_geo_twin", False))
    if not shifts and not geo:
        return {}, {}
    y_h = np.mean([np.asarray(metas[sc]["y0_decoded"]["height"], dtype=np.float64) for sc in metas], axis=0)
    y_a = np.mean([np.asarray(metas[sc]["y0_decoded"]["agb"], dtype=np.float64) for sc in metas], axis=0)
    if float(np.max(y_a)) < 0.05 or first_decade_temperature(site) >= float(getattr(in_args, "seed_cold_t", 8.0)):
        return {}, {}
    base_shift = float(getattr(in_args, "spinup_shift", 0.0))
    cands = {base_shift: spin}
    dirs = {base_shift: run / "out" / "spinup" / "RESTART"}
    extra_runs, extra_inputs = [], []
    t_extra = 0.0
    # candidate key: the trend shift (float) or "geo" (spin-up climatology of the geographically nearest training site,
    # 2011-2020, make_site_inputs.spinup_geo_twin); ties go to the earlier candidate (the base first)
    plan = [(s, {"spinup_shift": s}) for s in shifts if s != base_shift] + ([("geo", {"spinup_shift": 0.0, "spinup_geo_twin": True})] if geo else [])
    for s, over in plan:
        a2 = argparse.Namespace(**vars(in_args))
        for k, v in over.items():
            setattr(a2, k, v)
        tag = "geo" if s == "geo" else f"s{s:+.0f}".replace("+", "p").replace("-", "m")
        a2.out = str(root / "inputs" / f"seed_{site:05d}_{tag}")
        msi.build_site(site, "ssp126", a2, floors, latlon=latlon)
        inp2 = Path(a2.out) / f"site{site:05d}_ssp126"
        run2 = root / "run" / f"site{site:05d}_{tag}"
        run2.mkdir(parents=True, exist_ok=True); (run2 / "out").mkdir(exist_ok=True)
        write_cfgs(run2, {"ssp126": inp2, "ssp585": inp2}, EDV3 / "EDv3_code" / "ED_params.defaults.cfg", EDV3 / "EDv3_code" / "ED_pft.defaults.cfg",
                   fire_on=bool(getattr(in_args, "fire_on", False)), fp1=float(getattr(in_args, "fp1", 1.0)), params_over=getattr(in_args, "params_over", None),
                   spin_cycle=bool(getattr(in_args, "spinup_cycle", False)), fire_spin=getattr(in_args, "fire_spin", None))
        t_extra += run_ed(run2, "spinup", "models_spin.cfg", env_spin, run2 / "spinup.log")
        cands[s] = {k: parse_ed(v) for k, v in site_files(run2 / "out" / "spinup", "spinup").items()}
        dirs[s] = run2 / "out" / "spinup" / "RESTART"
        extra_runs.append(run2); extra_inputs.append(Path(a2.out))
    timings["seed_spinups"] = t_extra
    choice: dict[int, float] = {}
    for i, a in enumerate(AGES):
        losses = {s: seed_loss(state(tabs, float(a)), float(y_h[i]), float(y_a[i])) for s, tabs in cands.items()}
        rank = {s: k for k, s in enumerate(cands)}
        best = min(losses, key=lambda s: (losses[s], rank[s]))
        choice[a] = best
        if best != base_shift:
            for ext in ("css", "pss"):
                src = sorted(dirs[best].glob(f"*_y{a}.{ext}"))
                dst = sorted(dirs[base_shift].glob(f"*_y{a}.{ext}"))
                if len(src) != 1 or len(dst) != 1:
                    raise FileNotFoundError(f"seed dump _y{a}.{ext}: {len(src)} candidate / {len(dst)} base files")
                shutil.copyfile(src[0], dst[0])
    if not keep:
        for d in extra_runs + extra_inputs:
            shutil.rmtree(d, ignore_errors=True)
    return choice, cands


def density_scale_dumps(run: Path, spin: dict, seed_choice: dict, seed_cands: dict, metas: dict, lo: float = 0.25, hi: float = 4.0) -> dict:
    """Scale every cohort's nindivs in the spin-up dump of each seed age (RESTART/*_y<age>.css) so that the dump's agb
    equals the decoded y0 agb (mean of both scenarios, as the seed selection uses); heights and the cohort structure stay.
    Config key ``density_scale`` (11-09: test of the seed hypothesis). Returns {age: factor}; bare targets/dumps skipped."""
    restart = run / "out" / "spinup" / "RESTART"
    y_a = np.mean([np.asarray(metas[sc]["y0_decoded"]["agb"], dtype=np.float64) for sc in metas], axis=0)
    factors: dict[int, float] = {}
    for i, a in enumerate(AGES):
        tabs = seed_cands[seed_choice[a]] if seed_choice else spin
        e = float(state(tabs, float(a))["agb_kgC"])
        if not (np.isfinite(e) and e > 0.05 and y_a[i] > 0.05):
            continue
        f = float(np.clip(y_a[i] / e, lo, hi))
        css = sorted(restart.glob(f"*_y{a}.css"))
        if len(css) != 1:
            raise FileNotFoundError(f"seed dump _y{a}.css: {len(css)} files")
        lines = css[0].read_text().splitlines()
        j = lines[0].split().index("nindivs")
        out = [lines[0]]
        for ln in lines[1:]:
            t = ln.split()
            if len(t) > j:
                t[j] = repr(float(t[j]) * f)
            out.append(" ".join(t) + " ")
        css[0].write_text("\n".join(out) + "\n")
        factors[a] = f
    return factors


CLIM_VARS = ("precipitation", "temperature", "soil_temp1", "soil_temp2", "soil_temp3", "soil_temp4", "soil_temp5")


def rewrite_window_climatology(inp_dir: Path) -> None:
    """Replace climate_avg.nc (the climatology ED reads at site init: PET, the fire dryness index) by the mean of the
    yearly climate_<yr>.nc window files, so the windows' fire sees the window-period climate (config key window_clim)."""
    from scipy.io import netcdf_file
    files = sorted(inp_dir.glob("climate_[0-9]*.nc"))
    if not files:
        raise FileNotFoundError(f"no yearly climate files in {inp_dir}")
    acc: dict[str, np.ndarray] = {}
    for f in files:
        with netcdf_file(str(f), "r", mmap=False) as nc:
            for k in CLIM_VARS:
                if k in nc.variables:
                    acc[k] = acc.get(k, 0.0) + np.array(nc.variables[k][:], dtype=np.float64)
    with netcdf_file(str(inp_dir / "climate_avg.nc"), "a", mmap=False) as nc:
        for k, v in acc.items():
            nc.variables[k][:] = (v / len(files)).astype(nc.variables[k].data.dtype)

def results_exist(site: int, root: Path) -> bool:
    if (root / "SPIN_ONLY").exists():                       # lookup-only roots hold the per-site csv alone
        return (root / "csv" / f"site{site:05d}.csv").exists()
    return all((root / "results" / f"{site}_{sc}.csv").exists() for sc in ("ssp126", "ssp585")) and (root / "csv" / f"site{site:05d}.csv").exists()


def run_site(site: int, root: Path, in_args: argparse.Namespace, keep: bool = False) -> dict:
    root.mkdir(parents=True, exist_ok=True)
    if getattr(in_args, "geo_base", "") and site in geo_base_sites(in_args.geo_base):
        in_args = argparse.Namespace(**vars(in_args)); in_args.spinup_geo_twin = True   # base spin-up on the twin's MERRA-2 decade
    for spec, over in (getattr(in_args, "site_overrides", None) or []):   # per-group overrides: "cluster", "warm", "cold" or a file
        if site_in_group(site, spec, in_args):
            in_args = argparse.Namespace(**vars(in_args))
            for k, v in over.items():
                setattr(in_args, k, v)
            break
    for sub in ("csv", "results", "inputs", "run"):
        (root / sub).mkdir(exist_ok=True)
    if getattr(in_args, "spin_only", False):
        (root / "SPIN_ONLY").touch()
    floors = msi.rsds_channel_floors()
    inp, metas = {}, {}
    in_args.out = str(root / "inputs")
    latlon = None
    for sc in ("ssp126", "ssp585"):
        metas[sc] = msi.build_site(site, sc, in_args, floors, latlon=latlon)
        latlon = (metas[sc]["lat_est"], metas[sc]["lon_est"])
        inp[sc] = root / "inputs" / f"site{site:05d}_{sc}"
    run = root / "run" / f"site{site:05d}"
    run.mkdir(parents=True, exist_ok=True)
    (run / "out").mkdir(exist_ok=True)
    write_cfgs(run, inp, EDV3 / "EDv3_code" / "ED_params.defaults.cfg", EDV3 / "EDv3_code" / "ED_pft.defaults.cfg", fire_on=bool(getattr(in_args, "fire_on", False)), fp1=float(getattr(in_args, "fp1", 1.0)), params_over=getattr(in_args, "params_over", None),
               spin_cycle=bool(getattr(in_args, "spinup_cycle", False)), fire_spin=getattr(in_args, "fire_spin", None))
    env_spin = {"ED_DUMP_YEAR_SUFFIX": "1"}
    if getattr(in_args, "fire_spin_x", None):    # fire in the lookup: constant ignition factor of the lookup climatology (training rule, fp1 1.64e-5)
        sys.path.insert(0, str(ROOT / "scripts" / "fire"))
        import fire_x as _fxs
        xs_spin = np.asarray(_fxs.series(in_args.fire_spin_x, site, "ssp126", train=getattr(in_args, "latlon_mode", "") == "train_true"), dtype=float)
        if xs_spin.size >= 360:    # monthly factors (wet-month rule, per-month cap): used with the configured fp1 as they are
            np.savetxt(run / "fire_x_spin.txt", xs_spin * float(getattr(in_args, "fire_spin_scale", 1.0)), fmt="%.8g")
            if str(in_args.fire_spin_x).startswith("merm"):   # cycled MERRA lookup: factor of cycle year k at the January step of year k
                env_spin.update({"ED_FIRE_X_SHIFT": "0", "ED_FIRE_X_CYCLE": "1"})
        else:
            xc = float(xs_spin[0]) * 1.64e-5 / float(getattr(in_args, "fp1", 1.64e-5)) * float(getattr(in_args, "fire_spin_scale", 1.0))
            np.savetxt(run / "fire_x_spin.txt", [xc], fmt="%.8g")
        env_spin["ED_FIRE_X_FILE"] = str((run / "fire_x_spin.txt").resolve())
    if getattr(in_args, "spinup_cycle", False):
        n_cycle = metas["ssp126"]["spinup_climatology"]["cycle_files"]
        env_spin.update({"ED_WINDOW_YEAR0": "1981", "ED_YEAR_CYCLE": str(n_cycle)})
        if USE_MECH_CACHE:     # the cycle repeats its yearly tables every n_cycle years (binary with mech_cache.cc: build_fire4)
            (run / "cache_spin").mkdir(exist_ok=True)
            env_spin["ED_MECH_CACHE_DIR"] = str((run / "cache_spin").resolve())
    timings = {"spinup": run_ed(run, "spinup", "models_spin.cfg", env_spin, run / "spinup.log")}
    spin = {k: parse_ed(v) for k, v in site_files(run / "out" / "spinup", "spinup").items()}
    seed_choice, seed_cands = select_seeds(site, root, in_args, run, spin, metas, floors, latlon, env_spin, timings, keep)
    density = density_scale_dumps(run, spin, seed_choice, seed_cands, metas) if getattr(in_args, "density_scale", False) else {}
    if getattr(in_args, "window_clim", "") == "window_mean":   # windows read climate_avg.nc of their scenario dir
        for d in inp.values():
            rewrite_window_climatology(d)
    rows = []
    per_sc = bool(getattr(in_args, "spin_per_scenario", False))
    spin_only = bool(getattr(in_args, "spin_only", False))
    for sc, key in (("ssp126", "w126"), ("ssp585", "w585")):
        if per_sc and sc == "ssp585":
            # scenario-specific lookup: redo the spin-up on the ssp585 spin-up climatology / CO2, then its windows
            shutil.rmtree(run / "out" / "spinup", ignore_errors=True)
            write_cfgs(run, inp, EDV3 / "EDv3_code" / "ED_params.defaults.cfg", EDV3 / "EDv3_code" / "ED_pft.defaults.cfg", fire_on=bool(getattr(in_args, "fire_on", False)),
                       fp1=float(getattr(in_args, "fp1", 1.0)), params_over=getattr(in_args, "params_over", None),
                       spin_cycle=bool(getattr(in_args, "spinup_cycle", False)), fire_spin=getattr(in_args, "fire_spin", None), spin_sc="ssp585")
            timings["spinup585"] = run_ed(run, "spinup", "models_spin.cfg", env_spin, run / "spinup585.log")
            spin = {k: parse_ed(v) for k, v in site_files(run / "out" / "spinup", "spinup").items()}
            seed_choice, seed_cands = {}, {}
        if spin_only:
            y = metas[sc]["y0_decoded"]
            for i, a in enumerate(AGES):
                sp = state(spin, float(a))
                rows.append({"site": site, "scenario": sc, "age": a, **{f"spin_{k}": v for k, v in sp.items()},
                             **{f"y0_{n}": y[n][i] for n in ("height", "agb", "soil", "lai", "gpp", "npp", "rh")},
                             "lat_est": metas[sc]["lat_est"], "lon_est": metas[sc]["lon_est"]})
            timings[key] = 0.0
            continue
        tw = 0.0
        res40, resy0 = [], []
        env_win = {"ED_WINDOW_YEAR0": "1981"}
        if getattr(in_args, "fire_x", None):     # truth-calibrated fire: yearly ignition factors computed outside ED (scripts/fire/fire_x.py)
            sys.path.insert(0, str(ROOT / "scripts" / "fire"))
            import fire_x as _fx
            xs = _fx.series(in_args.fire_x, site, sc, train=getattr(in_args, "latlon_mode", "") == "train_true")
            xs = np.asarray(xs, dtype=float) * float(getattr(in_args, f"fire_x_scale_{key}", 1.0))      # forced-fire runs: window fire strength per scenario (scripts/ed/forced_fire.py)
            np.savetxt(run / f"fire_x_{key}.txt", xs, fmt="%.8g")
            env_win["ED_FIRE_X_FILE"] = str((run / f"fire_x_{key}.txt").resolve())
        if USE_MECH_CACHE:
            (run / f"cache_{key}").mkdir(exist_ok=True)
            env_win["ED_MECH_CACHE_DIR"] = str((run / f"cache_{key}").resolve())
        ages_only = getattr(in_args, "ages_only", None)       # forced-fire tuning runs: windows for these seed ages only (the csv then holds only their rows)
        for i, a in enumerate(AGES):
            if ages_only and int(a) not in ages_only:
                continue
            exp = f"{key}_a{a}"
            env_age = {**env_win, "ED_RESTART_SUFFIX": f"_y{a}"}
            if getattr(in_args, "fire_x", None) and getattr(in_args, "fire_fuel_fix", False):
                # the truth's ignition scales with ITS fuel (agb): correct the factor for the seed's miss of y0
                seed_agb = state(seed_cands[seed_choice[a]] if seed_choice else spin, float(a))["agb_kgC"]
                fs = float(np.clip((float(metas[sc]["y0_decoded"]["agb"][i]) + 0.05) / (seed_agb + 0.05), 0.67, 1.5)) if np.isfinite(seed_agb) else 1.0
                np.savetxt(run / f"fire_x_{key}_a{a}.txt", xs * fs, fmt="%.8g")
                env_age["ED_FIRE_X_FILE"] = str((run / f"fire_x_{key}_a{a}.txt").resolve())
            tw += run_ed(run, exp, f"models_{key}.cfg", env_age, run / f"{exp}.log")
            tabs = {k: parse_ed(v) for k, v in site_files(run / "out" / exp, exp).items()}
            if os.environ.get("ED_SAVE_TRAJ"):   # yearly December states of the window (year 0 = window start), for the fire calibration
                (root / "traj").mkdir(exist_ok=True)
                with open(root / "traj" / f"{site}_{sc}_a{a}.csv", "w", newline="") as fh:
                    recs = [{"year": k, **state(tabs, T_Y0 + k), **monthly_diag(tabs, T_Y0 + k)} for k in range(int(T_END - T_Y0) + 1)]
                    w = csv.DictWriter(fh, fieldnames=list(recs[0])); w.writeheader()
                    w.writerows(recs)
            s41, s1, sp = state(tabs, T_END), state(tabs, T_Y0), state(seed_cands[seed_choice[a]] if seed_choice else spin, float(a))
            res40.append({"age": a, **s41}); resy0.append({"age": a, **s1})
            y = metas[sc]["y0_decoded"]
            rows.append({"site": site, "scenario": sc, "age": a, **{f"ed40_{k}": v for k, v in s41.items()}, **{f"y0est_{k}": v for k, v in s1.items()},
                         **{f"pre_{k}": v for k, v in monthly_diag(tabs, T_Y0).items()}, **{f"end_{k}": v for k, v in monthly_diag(tabs, T_END).items()},
                         "spin_height_m": sp["height_m"], "spin_agb_kgC": sp["agb_kgC"], "spin_lai": sp["lai"], "spin_soil": sp["soil"],
                         "y0_height": y["height"][i], "y0_agb": y["agb"][i], "y0_lai": y["lai"][i], "y0_soil": y["soil"][i], "y0_gpp": y["gpp"][i],
                         "lat_est": metas[sc]["lat_est"], "lon_est": metas[sc]["lon_est"]})
            shutil.rmtree(run / "out" / exp, ignore_errors=True)
        timings[key] = tw
        for name, res in ((f"{site}_{sc}.csv", res40), (f"{site}_{sc}_y0.csv", resy0)):
            with open(root / "results" / name, "w", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=RESULT_COLS); w.writeheader(); w.writerows(res)
    with open(root / "csv" / f"site{site:05d}.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    info = {"site": site, "timings_s": timings, "config": vars(in_args), "co2": {sc: metas[sc]["co2"] for sc in metas},
            "seed_choice": {str(a): v for a, v in seed_choice.items()}, "density_factors": {str(a): v for a, v in density.items()}}
    json.dump(info, open(root / "csv" / f"site{site:05d}.json", "w"), indent=1)
    if not keep:
        shutil.rmtree(run, ignore_errors=True)
        for d in inp.values():
            shutil.rmtree(d, ignore_errors=True)
    return info


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--site", type=int, required=True)
    ap.add_argument("--root", default=str(EDV3 / "full"))
    ap.add_argument("--config", default="b", choices=sorted(CONFIGS))
    ap.add_argument("--keep", action="store_true")
    args = ap.parse_args()
    t0 = time.time()
    info = run_site(args.site, Path(args.root), config_args(args.config), keep=args.keep)
    print(f"site {args.site} config {args.config}: done in {time.time() - t0:.1f} s (spin-up {info['timings_s']['spinup']:.1f} s, windows {info['timings_s']['w126'] + info['timings_s']['w585']:.1f} s)")


if __name__ == "__main__":
    main()
