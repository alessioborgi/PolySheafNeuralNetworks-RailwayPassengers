#!/bin/sh

python -m exp.run \
    --add_hp=False \
    --add_lp=True \
    --d=4 \
    --dataset=ogbn-arxiv \
    --dropout=0.1 \
    --early_stopping=100 \
    --epochs=800 \
    --folds=10 \
    --hidden_channels=64 \
    --input_dropout=0.5 \
    --layers=5 \
    --lr=0.01 \
    --model=DiagSheaf \
    --second_linear=True \
    --sheaf_decay=0.001 \
    --weight_decay=0.0005 \
    --left_weights=True \
    --right_weights=True \
    --use_act=True \
    --normalised=True \
    --edge_weights=True \
    --stop_strategy=acc \
    --entity="${ENTITY}" 
