#!/bin/sh

python -m exp.run \
    --add_hp=False \
    --add_lp=True \
    --d=3 \
    --dataset=roman_empire \
    --dropout=0.1 \
    --early_stopping=100 \
    --epochs=1000 \
    --folds=10 \
    --hidden_channels=64 \
    --input_dropout=0.5 \
    --layers=5 \
    --lr=0.01 \
    --model=GeneralSheafPolynomial \
    --poly_layers_K=8 \
    --lambda_max_choice=analytic \
    --second_linear=True \
    --sheaf_decay=0.0005 \
    --weight_decay=0.0003 \
    --left_weights=True \
    --right_weights=True \
    --use_act=True \
    --normalised=True \
    --edge_weights=True \
    --stop_strategy=acc \
    --entity="${ENTITY}" 
