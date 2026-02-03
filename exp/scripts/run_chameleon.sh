#!/bin/sh

python -m exp.run \
    --add_hp=False \
    --add_lp=False \
    --d=2 \
    --dataset=chameleon \
    --dropout=0.7 \
    --early_stopping=200 \
    --epochs=500 \
    --folds=10 \
    --hidden_channels=32 \
    --input_dropout=0.8 \
    --layers=3 \
    --lr=0.01 \
    --model=BundleSheafPolynomial \
    --polynomial_type=ChebyshevType1 \
    --poly_layers_K=16 \
    --lambda_max_choice=analytic \
    --second_linear=True \
    --sheaf_decay=0.0001022 \
    --weight_decay=0.0010243 \
    --left_weights=True \
    --right_weights=True \
    --use_act=True \
    --normalised=True \
    --edge_weights=True \
    --stop_strategy=loss \
    --resource_analysis \
    --profile_flops \
    --cuda=1 \
    --entity="${ENTITY}" 





########################## PolySD VS NSD ##########################
# python -m exp.run \
#     --add_hp=1 \
#     --add_lp=0 \
#     --chebyshev_layers_K=6 \
#     --lambda_max_choice=analytic \
#     --d=4 \
#     --dataset=chameleon \
#     --dropout=0.7 \
#     --early_stopping=200 \
#     --epochs=500 \
#     --folds=10 \
#     --hidden_channels=16 \
#     --input_dropout=0.7 \
#     --layers=2 \
#     --lr=0.01 \
#     --model=GeneralSheafChebyshev \
#     --second_linear=True \
#     --sheaf_decay=0.000071535 \
#     --weight_decay=0.00013351 \
#     --left_weights=True \
#     --right_weights=True \
#     --use_act=True \
#     --normalised=True \
#     --edge_weights=True \
#     --stop_strategy=loss \
#     --entity="${ENTITY}" \
#     --cuda=0