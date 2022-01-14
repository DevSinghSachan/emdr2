#!/bin/bash

BASE_DIR="/mnt/disks/project"

# BERT model configuration
CONFIG="base"

# Path of the checkpoint and evidence embeddings
CHECKPOINT_PATH="${BASE_DIR}/checkpoints/dualencoder-mss-dpr-${CONFIG}-epochs40-nq"
EMBEDDING_PATH="${BASE_DIR}/embedding-path/psgs_w100-dualencoder-mss-dpr-${CONFIG}-epochs40-nq.pkl"

rm -rf ${CHECKPOINT_PATH} ${EMBEDDING_PATH}

# To use MSS initialization, set the below variable to true
MSS_INIT="true"

# To train with hard negatives, mark this as true
TRAIN_WITH_NEG="true"

# Natural Questions data path
DATA_DIR="${BASE_DIR}/data/dpr/retriever"
TRAIN_DATA="${DATA_DIR}/nq-train.json"
VALID_DATA="${DATA_DIR}/nq-dev.json"

# BERT vocabulary path
VOCAB_FILE="${BASE_DIR}/bert-vocab/bert-large-uncased-vocab.txt"

# Wikipedia evidence path (from DPR code) and NQ evaluation questions
EVIDENCE_DATA_DIR="${BASE_DIR}/data/dpr/wikipedia_split/psgs_w100.tsv"
QA_FILE_DEV="${BASE_DIR}/data/dpr/retriever/qas/nq-dev.csv"
QA_FILE_TEST="${BASE_DIR}/data/dpr/retriever/qas/nq-test.csv"


DISTRIBUTED_ARGS="-m torch.distributed.launch --nproc_per_node 16 --nnodes 1 --node_rank 0 --master_addr localhost --master_port 6000"


function config_base() {
    export CONFIG_ARGS="--num-layers 12 \
--hidden-size 768 \
--num-attention-heads 12 \
--ffn-hidden-size 3072 \
--model-parallel-size 1"
}


if [ ${CONFIG} == "base" ]; then
    config_base
    export BERT_LOAD_PATH="${BASE_DIR}/checkpoints/bert_110m"
else
    echo "Invalid BERT model configuration"
    exit 1
fi


EXTRA_OPTIONS=""
if [ ${MSS_INIT} == "true" ]; then
    if [ ${CONFIG} == "base" ]; then
        PRETRAINED_CHECKPOINT="${BASE_DIR}/checkpoints/mss-emdr2-retriever-base-steps82k"
    fi
    echo "This will work because the code will first load BERT checkpoints (default) and then will load PRETRAINED_CHECKPOINT"
    export EXTRA_OPTIONS+=" --finetune --pretrained-checkpoint ${PRETRAINED_CHECKPOINT}"
fi


if [ ${TRAIN_WITH_NEG} == "true" ]; then
     export EXTRA_OPTIONS+=" --train-with-neg --train-hard-neg 1"
fi

OPTIONS=" \
          --task RETRIEVER \
          --tokenizer-type BertWordPieceLowerCase \
          --train-data ${TRAIN_DATA} \
          --valid-data ${VALID_DATA} \
          --save ${CHECKPOINT_PATH} \
          --load ${CHECKPOINT_PATH} \
          --qa-file-dev ${QA_FILE_DEV} \
          --qa-file-test ${QA_FILE_TEST} \
          --evidence-data-path ${EVIDENCE_DATA_DIR} \
          --embedding-path ${EMBEDDING_PATH} \
          --vocab-file ${VOCAB_FILE} \
          --bert-load ${BERT_LOAD_PATH} \
          --save-interval 5000 \
          --log-interval 20 \
          --eval-iters 100 \
          --indexer-log-interval 1000 \
          --distributed-backend nccl \
          --faiss-use-gpu \
          --DDP-impl torch \
          --fp16 \
          --num-workers 2 \
          --sample-rate 1.00 \
          --report-topk-accuracies 1 5 10 20 50 100 \
          --seq-length 512 \
          --seq-length-ret 256 \
          --max-position-embeddings 512 \
          --attention-dropout 0.1 \
          --hidden-dropout 0.1 \
          --retriever-score-scaling \
          --epochs 40 \
          --batch-size 8 \
          --eval-batch-size 16 \
          --indexer-batch-size 128 \
          --lr 2e-5 \
          --warmup 0.01 \
          --lr-decay-style linear \
          --weight-decay 1e-1 \
          --clip-grad 1.0 \
          --max-training-rank 16 "


COMMAND="WORLD_SIZE=16 python ${DISTRIBUTED_ARGS} tasks/run.py ${OPTIONS} ${CONFIG_ARGS} ${EXTRA_OPTIONS}"
eval ${COMMAND}
exit

set +x
