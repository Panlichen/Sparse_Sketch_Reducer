#!/bin/bash
ssh gn11
nohup python -m torch.distributed.launch --nnode=1 --node_rank=0 --nproc_per_node=4 --master_addr="10.107.16.40" --master_port=1234 /gf3/home/gks/project/sketch_reduce/train.py >> ./1G_network/lstm/6-27_lm_sketch.log 2>&1 &
ssh gn10
python -m torch.distributed.launch --nnode=2 --node_rank=1 --nproc_per_node=4 --master_addr="10.107.16.90" --master_port=1234 /gf3/home/gks/project/sketch_reduce/train.py 
