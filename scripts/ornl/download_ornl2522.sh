#!/bin/bash
# Download a subset of ORNL DAAC 2522 (Ma & Hurtt 2026, "Global Biomass Projections and Forest Carbon Sequestration
# Potential, 2020-2100", doi:10.3334/ORNLDAAC/2522) with the user's Earthdata login in ~/.netrc (never read here;
# curl -n uses it). Files land in work/ornl2522/ (resumable: complete files are skipped).
#   [FOREST=existing|new] bash scripts/ornl/download_ornl2522.sh [ESM] [CASE...]      default: existing, GFDL-ESM4 CLIM_CO2
set -u
cd "$(dirname "$0")/../.."
ESM=${1:-GFDL-ESM4}; shift || true
CASES=${*:-CLIM_CO2}
OUT=work/ornl2522; mkdir -p $OUT
FOREST=${FOREST:-existing}
BASE=https://data.ornldaac.earthdata.nasa.gov/protected/cms/Global_Forests_Future_Carbon/data
JAR=$OUT/.cookies
until [ -f "$HOME/.netrc" ]; do sleep 30; done
files=""
for c in $CASES; do
  for s in SSP126 SSP585; do
    for d in 2021_2030 2051_2060; do files="$files Biomass_${ESM}_${s}_${c}_${FOREST}_forest_${d}.tif"; done
  done
done
for d in 2021_2030 2051_2060; do files="$files Biomass_REF_${FOREST}_forest_${d}.tif"; done
for f in $files; do
  if [ -f "$OUT/$f" ]; then echo "have $f"; continue; fi
  echo "$(date -u +%H:%M) get $f"
  curl -sS -L -n -c "$JAR" -b "$JAR" --fail --retry 3 --retry-delay 20 -o "$OUT/$f.part" "$BASE/$f" && mv "$OUT/$f.part" "$OUT/$f" || echo "FAILED $f"
  ls -la "$OUT/$f" 2>/dev/null | awk '{print "   ", $5, "bytes"}'
done
echo "done $(date -u +%H:%M)"
