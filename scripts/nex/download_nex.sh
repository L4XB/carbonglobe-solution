#!/bin/bash
# NEX-GDDP-CMIP6 (public AWS open data, no login) daily files -> work/nexgddp/. Usage: bash scripts/nex/download_nex.sh VAR SCEN Y0 Y1 [suffix]
# suffix "" (v1) or "_v2.0" (fixed tas). Resumable.
set -u
cd "$(dirname "$0")/../.."
V=$1; SC=$2; Y0=$3; Y1=$4; SUF=${5:-}
B=https://nex-gddp-cmip6.s3.us-west-2.amazonaws.com/NEX-GDDP-CMIP6/GFDL-ESM4/$SC/r1i1p1f1/$V
for y in $(seq $Y0 $Y1); do
  f=${V}_day_GFDL-ESM4_${SC}_r1i1p1f1_gr1_${y}${SUF}.nc
  [ -f work/nexgddp/$f ] && continue
  curl -sS --fail --retry 3 --max-time 1200 -o work/nexgddp/$f.part "$B/$f" && mv work/nexgddp/$f.part work/nexgddp/$f || echo "FAILED $f"
done
echo "done $V $SC $Y0-$Y1$SUF"
