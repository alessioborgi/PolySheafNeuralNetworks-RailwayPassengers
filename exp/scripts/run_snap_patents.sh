#!/bin/sh

python -m exp.run \
    --add_hp=False \
    --add_lp=True \
    --d=3 \
    --dataset=snap-patents \
    --dropout=0.3 \
    --early_stopping=100 \
    --epochs=800 \
    --folds=10 \
    --hidden_channels=64 \
    --input_dropout=0.4 \
    --layers=4 \
    --lr=0.01 \
    --model=GeneralSheafPolynomial \
    --poly_layers_K=8 \
    --lambda_max_choice=analytic \
    --second_linear=True \
    --sheaf_decay=0.0005 \
    --weight_decay=0.0005 \
    --left_weights=True \
    --right_weights=True \
    --use_act=True \
    --normalised=True \
    --edge_weights=True \
    --stop_strategy=acc \
    --entity="${ENTITY}" 
