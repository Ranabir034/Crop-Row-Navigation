#!/bin/bash
mkdir -p Results

for f in images/ZED_imageimage_*.png; do
    echo "Processing $f ..."
    
    base=$(basename "$f" .png)

    python3 crop_segmentation.py \
        --image "$f" \
        --out "Results/${base}_out.png"
done

