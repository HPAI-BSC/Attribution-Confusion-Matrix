#!/bin/bash
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PROJECT_DIR="$( dirname $( dirname $( dirname $SCRIPT_DIR ) ) )"
cd $PROJECT_DIR
PYTHONPATH=$PROJECT_DIR python explainability/explainability_mosaic.py mit670_mosaic vgg16 mit67_lrfs_pt_places_vgg16_lr1e5_e28.ckpt lime