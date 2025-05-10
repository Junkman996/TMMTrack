#!/bin/bash
CFG=${1:-configs/default.yaml}
CKPT=${2}
python -u evaluate.py --cfg $CFG --ckpt $CKPT