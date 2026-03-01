#!/bin/sh

 # Possible models: [ DiagSheaf, BundleSheaf, GeneralSheaf, DiagSheafChebyshev, BundleSheafChebyshev, GeneralSheafChebyshev, EquivariantDiagSheaf, EquivariantBundleSheaf, EquivariantGeneralSheaf, EquivariantDiagSheafChebyshev] 

python -m exp.run \
    --dataset=tokyo_railway \
    --d=3 \
    --layers=2 \
    --hidden_channels=16 \
    --left_weights=True \
    --right_weights=True \
    --lr=0.02 \
    --maps_lr=0.005 \
    --input_dropout=0.0 \
    --dropout=0.7 \
    --use_act=True \
    --model=GeneralSheafPolynomial \
    --task=regression \
    --norm=global \
    --inductive=True \
    --polynomial_type="ChebyshevType1" \
    --normalised=True \
    --deg_normalised=False \
    --sparse_learner=True \
    --lambda_max_choice="iterative" \
    --chebyshev_layers_K=15 \
    --early_stopping=200 \
    --weight_decay=0.005 \
    --folds=2 \
    --cuda=0 \
    --entity="${ENTITY}" \
    --wandb_project="Tokyo_Railway" \
    --save_restriction_maps \
    --save_dir="../checkpoints/tokyo_railway/BundleSheafPolynomial_seed/fold0" \
    