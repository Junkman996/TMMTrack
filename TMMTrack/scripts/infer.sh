CFG=${1:-configs/default.yaml}
CKPT=${2}
VIDEO=${3}
python -u infer.py --cfg $CFG --ckpt $CKPT --video $VIDEO