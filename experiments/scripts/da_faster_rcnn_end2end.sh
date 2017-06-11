#!/bin/bash
# Usage:
# ./experiments/scripts/faster_rcnn_end2end_da.sh GPU NET LOSS [options args to {train,test}_net.py]
# DATASET is either pascal_voc or coco.
#
# Example:
# ./experiments/scripts/faster_rcnn_end2end_da.sh 0 VGG16 mmd \
#   EXP_DIR foobar RNG_SEED 42 TRAIN.SCALES "[400, 500, 600, 700]"

set -x
set -e

export PYTHONUNBUFFERED="True"

GPU_ID=$1
NET=$2
NET_lc=${NET,,}
ADAPTATION_LOSS=$3
TEST_IMDB=minicoco_2014_minival
# SRC_DATASET=$3
# TARGET_DATASET=$4

array=( $@ )
len=${#array[@]}
EXTRA_ARGS=${array[@]:3:$len}
EXTRA_ARGS_SLUG=${EXTRA_ARGS// /_}


case $ADAPTATION_LOSS in
  grl)
    PT_DIR="adaptation/grl"
    ITERS=350000
    WEIGHTS=data/imagenet_models/${NET}.v2.caffemodel
    EXTRA_ARGS="$EXTRA_ARGS TRAIN.ADAPTATION_LOSS DC_LOSS"
    ;;
  mmd)
    PT_DIR="adaptation/mkmmd"
    ITERS=70000
    WEIGHTS=data/faster_rcnn_models/${NET}_faster_rcnn_final.caffemodel
    EXTRA_ARGS="$EXTRA_ARGS TRAIN.ADAPTATION_LOSS MMD_LOSS TRAIN.RPN_POST_NMS_TOP_N 200"
    ;;
  *)
    echo "No Loss choice given"
    exit
    ;;
esac

if [[ ! -z $EXTRA_ARGS ]] ; then
  EXTRA_ARGS="--set $EXTRA_ARGS"
fi
echo $EXTRA_ARGS
echo $EXTRA_ARGS_SLUG

LOG="experiments/logs/faster_rcnn_end2end_${NET}_${ADAPTATION_LOSS}_${EXTRA_ARGS_SLUG}.txt.`date +'%Y-%m-%d_%H-%M-%S'`"
exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"

time python ./tools/train_da_net.py --gpu ${GPU_ID} \
  --solver models/${PT_DIR}/${NET}/faster_rcnn_end2end/solver.prototxt \
  --weights $WEIGHTS \
  --iters $ITERS \
  --cfg experiments/cfgs/faster_rcnn_end2end_da.yml \
  ${EXTRA_ARGS}

set +x
NET_FINAL=`grep -B 1 "done solving" ${LOG} | grep "Wrote snapshot" | awk '{print $4}'`
set -x

time ./tools/test_net.py --gpu ${GPU_ID} \
  --def models/${PT_DIR}/${NET}/faster_rcnn_end2end/test.prototxt \
  --net ${NET_FINAL} \
  --imdb ${TEST_IMDB} \
  --cfg experiments/cfgs/faster_rcnn_end2end.yml \
  ${EXTRA_ARGS}
