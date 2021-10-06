#!/bin/bash

set -e

DATASET_DIR='/gf3/home/gks/data/wmt16_de_en/'

SEED=${1:-"1"}
TARGET=${2:-"24.00"}
RESUME_DIR='/gf3/home/gks/project/sketch_reduce/training-rnn_translator/pytorch/results/gnmt'
MATH=1

# run training
python -m torch.distributed.launch --nnode=4 --node_rank=3 --nproc_per_node=4 --master_addr="10.107.15.50" --master_port=1234 /gf3/home/gks/project/sketch_reduce/training-rnn_translator/pytorch/train_gnmt.py \
  --dataset-dir ${DATASET_DIR} \
  --seed $SEED \
  --target-bleu $TARGET \
  --math 'sketch_embed'