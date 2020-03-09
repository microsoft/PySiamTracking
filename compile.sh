#!/usr/bin/env bash

PYTHON=${PYTHON:-"python"}

echo "Building crop_and_resize op..."
cd siam_tracker/ops/crop_and_resize
if [ -d "build" ]; then
    rm -r build
fi
$PYTHON setup.py build_ext --inplace

echo "Building roi align op..."
cd ../roi_align
if [ -d "build" ]; then
    rm -r build
fi
$PYTHON setup.py build_ext --inplace

echo "Building NMS op..."
cd ../nms
if [ -d "build" ]; then
    rm -r build
fi
$PYTHON setup.py build_ext --inplace
$PYTHON setup_cython.py build_ext --inplace

echo "Building toolkit op..."
cd ../../benchmarks/utils/vot/region
if [ -d "build" ]; then
    rm -r build
fi
$PYTHON setup.py build_ext --inplace