#!/bin/sh
set -e

CORPUS=../wiki_LEMMA_parsed.txt
MODEL=  # use for finetuning
SG=
CBOW_SUM=
VECTOR_LEN=300
WINDOW=7
VOCAB_MIN_COUNT=10
SEED=42
EPOCHS=1000  # for current train
MAX_EPOCHS=1000  # for all trains, counting finetuning
CHECKPOINT_EVERY=20
WORKERS=+1
SAVE_MODEL=1

__SG=
if [ $SG ]; then
    __SG=--sg
fi
__CBOW_SUM=
if [ $CBOW_SUM ]; then
    __CBOW_SUM=--cbow_sum
fi
__MODEL=
if [ $MODEL ]; then
    __MODEL=--model
fi
__SAVE_MODEL=
if [ $SAVE_MODEL ]; then
    __SAVE_MODEL=--save_model
fi

python w2v_train.py --corpus $CORPUS $__MODEL $MODEL $__SG $__CBOW_SUM \
       --vector_len $VECTOR_LEN --window $WINDOW \
       --vocab_min_count $VOCAB_MIN_COUNT --seed $SEED --epochs $EPOCHS \
       --max_epochs $MAX_EPOCHS --checkpoint_every $CHECKPOINT_EVERY \
       --workers $WORKERS $__SAVE_MODEL
