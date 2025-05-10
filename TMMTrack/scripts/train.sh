#!/bin/bash
CONFIG=${1:-configs/default.yaml}
WORK_DIR=./work_dir
mkdir -p $WORK_DIR
python -u train.py --cfg $CONFIG --work-dir $WORK_DIR