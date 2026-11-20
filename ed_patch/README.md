# Changes to the public ED v3.0 source

The model source is the public archive of Ma et al. (2022, Geosci. Model Dev. 15, 1971-1994; Zenodo). It is not
redistributed here. Apply a patch inside a copy of that source (`patch -p1 < build_fire6.patch`), add the new files of the
same folder and build with the patched Makefile (netCDF, TBB, Berkeley DB and libconfig are needed).

| build | used for | changes |
|---|---|---|
| `build_fire4.patch` | main pass without fire (config cx16) | forcing years cycled during the spin-up (`ED_YEAR_CYCLE`), yearly mechanism tables cached (`mech_cache.cc`), start year of the window (`ED_WINDOW_YEAR0`), CO2 passed through as given (`ED_CO2_AS_IS`), restart and dump suffixes, optional ignition factor per window year from a file (`ED_FIRE_X_FILE`, `ED_FIRE_X_SHIFT`) |
| `build_fire6.patch` | main pass with ED's fire rule (config cx17b) and the runs with a site-tuned ignition strength | build_fire4 plus monthly ignition factors with the yearly cap applied after the monthly mean (`ED_FIRE_X_YEARCAP`) and cycled ignition factors during the spin-up (`ED_FIRE_X_CYCLE`) |

The switches are environment variables read by the patched code; `scripts/ed/run_site.py` sets them per configuration
(`ED_EDLU` = path of the binary). All changes are marked with the comment `CarbonGlobe` in the patch.
