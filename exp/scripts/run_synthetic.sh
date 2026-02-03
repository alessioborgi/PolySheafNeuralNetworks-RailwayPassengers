#!/bin/sh

#comment or uncomment to generate new data
# rm -f ./datasets/synthetic_exp/processed/*
# rm -f ./datasets/synthetic_exp/raw/*
# rm -f ./splits/synthetic_exp_split*

python -m exp.run \
    --dataset=synthetic_exp \
    --d=3 \
    --layers=3 \
    --epochs=500 \
    --early_stopping=200 \
    --hidden_channels=5 \
    --left_weights=False \
    --right_weights=False \
    --lr=0.01 \
    --maps_lr=0.01 \
    --weight_decay=0 \
    --input_dropout=0.0 \
    --dropout=0 \
    --use_act=False \
    --model=DiagSheaf \
    --normalised=True \
    --deg_normalised=False \
    --dual_normalised=False \
    --dual_diff_strength=1 \
    --learn_first_maps=False \
    --dual_linear=True \
    --dual_left_linear=False \
    --dual_right_linear=False \
    --sheaf_decay=0 \
    --dual_diag=True \
    --sparse_learner=False \
    --dual_left_linear=False \
    --dual_right_linear=False \
    --rotation_invariant_sheaf_learner=False \
    --node_edge_sims_time_dependent=False \
    --sheaf_init=False \
    --second_linear=False \
    --entity="${ENTITY}" \
    --feat_noise=0 \
    --edge_noise=0 \
    --num_classes=2 \
    --num_nodes=500 \
    --num_feats=15 \
    --het_coef=1 \
    --node_degree=5 \
    --ellipsoid_radius=1 \
    --use_epsilons=False \
    --use_embedding=True