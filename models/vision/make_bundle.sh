#!/usr/bin/env bash
# Zip the committed repo for the Colab notebook (models/vision/train.ipynb).
#
#   bash models/vision/make_bundle.sh      # -> models/vision/out/waste_segregation.zip
#
# Only committed files go in (git archive), so no models, data or secrets. Colab downloads
# TACO itself, which is faster than uploading the images.
set -euo pipefail
cd "$(dirname "$0")/../.."
mkdir -p models/vision/out
if [[ -n "$(git status --porcelain)" ]]; then
    echo "note: uncommitted changes are NOT in the zip (git archive uses HEAD)"
fi
git archive -o models/vision/out/waste_segregation.zip HEAD
echo "wrote models/vision/out/waste_segregation.zip ($(du -h models/vision/out/waste_segregation.zip | cut -f1)) from $(git rev-parse --short HEAD)"
