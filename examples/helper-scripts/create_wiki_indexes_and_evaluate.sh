#!/bin/bash

NPROC=16
BASE_DIR="/mnt/disks/project"
DATA_DIR="${BASE_DIR}/data/dpr/wikipedia_split/psgs_w100.tsv"
VOCAB_FILE="${BASE_DIR}/bert-vocab/bert-large-uncased-vocab.txt"

EMBEDDING_PATH="$BASE_DIR/embedding-path/nq-ssm-step9000.pkl"
# psgs_w100_univ-realm-dpr-retriever-base-steps82k_full-wikipedia_base.pkl"

CHECKPOINT_PATH="$BASE_DIR/checkpoints/ablations/nq-ssm-step9000"
# biencoder_dpr_cls-token_score-scaling_bert-base_hard-neg7_epochs40_univrealm-init82k_triviaqa"

DISTRIBUTED_ARGS="-m torch.distributed.launch --nproc_per_node ${NPROC} --nnodes 1 --node_rank 0 --master_addr localhost --master_port 6000"

QA_FILE_DEV="${BASE_DIR}/data/dpr/retriever/qas/nq-dev.csv"
QA_FILE_TEST="${BASE_DIR}/data/dpr/retriever/qas/nq-test.csv"

CREATE_EVIDENCE_INDEXES="true"
EVALUATE_RETRIEVER_RECALL="true"

OPTIONS="--num-layers 12 \
    --hidden-size 768 \
    --num-attention-heads 12 \
    --batch-size 128 \
    --checkpoint-activations \
    --seq-length 512 \
    --seq-length-ret 256 \
    --max-position-embeddings 512 \
    --load ${CHECKPOINT_PATH} \
    --evidence-data-path ${DATA_DIR} \
    --embedding-path ${EMBEDDING_PATH} \
    --indexer-log-interval 1000 \
    --indexer-batch-size 128 \
    --vocab-file ${VOCAB_FILE} \
    --num-workers 2 \
    --fp16 \
    --max-training-rank ${NPROC}"


if [ ${CREATE_EVIDENCE_INDEXES} == "true" ];
then
    COMMAND="WORLD_SIZE=16 python ${DISTRIBUTED_ARGS} create_doc_index.py ${OPTIONS}"
    eval "${COMMAND}"
fi
set +x


OPTIONS="--num-layers 12 \
    --hidden-size 768 \
    --num-attention-heads 12 \
    --checkpoint-activations \
    --seq-length 512 \
    --max-position-embeddings 512 \
    --load ${CHECKPOINT_PATH} \
    --evidence-data-path ${DATA_DIR} \
    --embedding-path ${EMBEDDING_PATH} \
    --batch-size 16 \
    --seq-length-ret 256 \
    --vocab-file ${VOCAB_FILE} \
    --qa-file-dev ${QA_FILE_DEV} \
    --qa-file-test ${QA_FILE_TEST} \
    --num-workers 2 \
    --faiss-use-gpu \
    --report-topk-accuracies 1 5 10 20 50 100 \
    --fp16 \
    --topk-retrievals 100 \
    --max-training-rank ${NPROC}"


if [ ${EVALUATE_RETRIEVER_RECALL} == "true" ];
then
COMMAND="WORLD_SIZE=${NPROC} python ${DISTRIBUTED_ARGS} evaluate_open_retrieval.py ${OPTIONS}"
eval "${COMMAND}"
fi

set +x
