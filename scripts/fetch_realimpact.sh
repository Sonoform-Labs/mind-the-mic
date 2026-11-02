#!/usr/bin/env bash
set -u
PKG="$(cd "$(dirname "$0")/.." && pwd)"
DEST="${MTM_REALIMPACT_DIR:-$PKG/data/realimpact}"
BASE="http://downloads.cs.stanford.edu/viscam/RealImpact"
OBJECTS="3_CeramicKoiBowl 10_bowl 17_IronSkillet 19_Pan 27_WoodPlate 31_WoodSlab 32_WoodChalice 34_WoodMug 43_IronMortar 48_PlasticBowl 49_PlasticBowl 61_PlanterCeramic 64_CeramicMug 67_IronPlate 73_PlasticBin 78_CeramicCup 83_WoodVase 91_MetalSpoon 94_GlassGoblet 97_PlasticScoop"
mkdir -p "$DEST"
for o in $OBJECTS; do
  if [ -f "$DEST/$o/preprocessed/deconvolved_0db.npy" ]; then
    echo "skip $o"
    continue
  fi
  echo "get $o"
  if ! curl -L -f -C - -o "$DEST/$o.zip" "$BASE/$o.zip"; then
    echo "failed $o"
    continue
  fi
  unzip -q -o "$DEST/$o.zip" -d "$DEST" && rm -f "$DEST/$o.zip"
done
n=0
for o in $OBJECTS; do
  [ -f "$DEST/$o/preprocessed/deconvolved_0db.npy" ] && n=$((n+1))
done
echo "$n of 20 objects ready in $DEST"
