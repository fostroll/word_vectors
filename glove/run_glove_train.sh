#!/bin/sh
set -e

CORPUS=../wiki_LEMMA_parsed.txt
VECTOR_SIZE=300
WINDOW_SIZE=15
SYMMETRIC=1
VOCAB_MIN_COUNT=10
SEED=42
EPOCHS=1000
CHECKPOINT_EVERY=20
NUM_THREADS=7
VOCAB_FILE=glove_vocab.txt
COOCCURRENCE_FILE=glove_cooccurrence.bin
COOCCURRENCE_SHUF_FILE=glove_cooccurrence.shuf.bin
SAVE_GRADSQ=0
GRADSQ_FILE=glove_gradsq.bin
INIT_GRADSQ_FILE=glove_gradsq.bin
LOAD_INIT_GRADSQ=0
BUILDDIR=glove/build
VERBOSE=2
MEMORY=24.0
BINARY=0 # 0 - text, 1 - binary, 2 - both
X_MAX=100.0
SAVE_FILE=glove_$VECTOR_SIZE
W2V_FORMAT=1
REUSE_VOCAB=1

if [ ! -d glove ]; then
    git clone http://github.com/stanfordnlp/glove && cd glove && make && cd ..
fi

_GRADSQ_FILE=
if [ $SAVE_GRADSQ -eq 0 ]; then
    _GRADSQ_FILE=-gradsq_file
    GRADSQ_FILE=
fi

echo $REUSE_VOCAB $VOCAB_FILE $_a
if [ $REUSE_VOCAB -eq 0 ] || [ ! -f $VOCAB_FILE ]; then
    echo "$ $BUILDDIR/vocab_count -min-count $VOCAB_MIN_COUNT -verbose $VERBOSE < $CORPUS > $VOCAB_FILE"
    $BUILDDIR/vocab_count -min-count $VOCAB_MIN_COUNT -verbose $VERBOSE < $CORPUS > $VOCAB_FILE
fi
if [ $REUSE_VOCAB -eq 0 ] || [ ! -f $COOCCURRENCE_FILE ]; then
    echo "$ $BUILDDIR/cooccur -memory $MEMORY -vocab-file $VOCAB_FILE -verbose $VERBOSE -window-size $WINDOW_SIZE -symmetric=$SYMMETRIC < $CORPUS > $COOCCURRENCE_FILE"
    $BUILDDIR/cooccur -memory $MEMORY -vocab-file $VOCAB_FILE -verbose $VERBOSE -window-size $WINDOW_SIZE -symmetric=$SYMMETRIC < $CORPUS > $COOCCURRENCE_FILE
fi
if [ $REUSE_VOCAB -eq 0 ] || [ ! -f $COOCCURRENCE_SHUF_FILE ]; then
    echo "$ $BUILDDIR/shuffle -memory $MEMORY -verbose $VERBOSE -seed $SEED < $COOCCURRENCE_FILE > $COOCCURRENCE_SHUF_FILE"
    $BUILDDIR/shuffle -memory $MEMORY -verbose $VERBOSE -seed $SEED < $COOCCURRENCE_FILE > $COOCCURRENCE_SHUF_FILE
fi
echo "$ $BUILDDIR/glove -save-file $SAVE_FILE -threads $NUM_THREADS -eta 0.05 -alpha 0.75 -x-max $X_MAX -iter $EPOCHS -vector-size $VECTOR_SIZE -binary $BINARY -input-file $COOCCURRENCE_SHUF_FILE -vocab-file $VOCAB_FILE  -model 2-verbose $VERBOSE -seed $SEED -write-header $W2V_FORMAT -checkpoint-every $CHECKPOINT_EVERY $_GRADSQ_FILE $GRADSQ_FILE -load-init-gradsq $LOAD_INIT_GRADSQ -init-gradsq-file $INIT_GRADSQ_FILE"
$BUILDDIR/glove -save-file $SAVE_FILE -threads $NUM_THREADS -eta 0.05 -alpha 0.75 -x-max $X_MAX -iter $EPOCHS -vector-size $VECTOR_SIZE -binary $BINARY -input-file $COOCCURRENCE_SHUF_FILE -vocab-file $VOCAB_FILE -model 2 -verbose $VERBOSE -seed $SEED -write-header $W2V_FORMAT -checkpoint-every $CHECKPOINT_EVERY $_GRADSQ_FILE $GRADSQ_FILE -load-init-gradsq $LOAD_INIT_GRADSQ -init-gradsq-file $INIT_GRADSQ_FILE
