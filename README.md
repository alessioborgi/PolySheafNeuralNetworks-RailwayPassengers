<div align="center">

# <b>Passenger Count Predictions on the Tokyo Railway Network by Deniz Alkan

[Deniz Alkan]

# <b>Polynomial Neural Sheaf Diffusion: A Spectral Polynomial Diffusion Approach on Cellular Sheaves



[Alessio Borgi](https://scholar.google.com/citations?hl=it&user=Ds4ktdkAAAAJ)<sup>*,</sup><sup>1,</sup><sup>2</sup>, [Fabrizio Silvestri](https://scholar.google.com/citations?user=pi985dQAAAAJ&hl=it&oi=ao)<sup>1</sup>, [Pietro Liò](https://scholar.google.com/citations?user=4YhNJBEAAAAJ&hl=it&oi=ao)<sup>2</sup>

<sup>1</sup>Sapienza University of Rome, <sup>2</sup> University of Cambridge
<p>
  <a href="mailto:borgi@diag.uniroma1.it">borgi@diag.uniroma1.it</a>,
  <a href="mailto:fsilvestri@diag.uniroma1.it">fsilvestri@diag.uniroma1.it</a>,
  <a href="mailto:pl219@cam.ac.uk">pl219@cam.ac.uk</a>
</p>

<p align="center"><sup>*</sup>Corresponding author</p>

<!--- ### <b>[CVPR 2025](https://cvpr.thecvf.com/) [Workshop on AI for Creative Visual Content Generation, Editing, and Understanding](https://cveu.github.io/)
      ### <b>[Published in 2025 IEEE-CVF Conference on Computer Vision and Pattern Recognition Workshops (CVPRW)](https://cveu.github.io/)
      ### <b>[Official CVPR 2025 Workshop Procedings](https://cveu.github.io/)
-->

</div>


</div>

<p align="center">
  <!---
  <a href="https://ieeexplore.ieee.org/document/11147719">
    <img src="https://img.shields.io/badge/CVPR%202025-IEEE CVF CVPRW-blue" alt="CVPR 2025 Workshop Accepted" style="height: 25px; margin-right: 5px;">
  </a>
  -->
  <a href="https://arxiv.org/html/2512.00242v1">
    <img src="https://img.shields.io/badge/arXiv-2512.00242v1-orange" alt="arXiv" style="height: 25px; margin-right: 5px;">
  </a>
  <!---
  <a href="https://alessioborgi.github.io/Z-SASLM.github.io/">
    <img src="https://img.shields.io/badge/Website-Project-green?logo=githubpages&logoColor=white" alt="Website" style="height: 25px; margin-right: 5px;">
  </a>
  -->
  <a href="https://huggingface.co/papers/2503.23234">
    <img src="https://img.shields.io/badge/HuggingFace-Papers-blue?logo=huggingface" alt="Hugging Face" style="height: 25px; margin-right: 5px;">
  </a>
  <a href="https://www.researchgate.net/publication/390303255_Z-SASLM_Zero-Shot_Style-Aligned_SLI_Blending_Latent_Manipulation">
    <img src="https://img.shields.io/badge/ResearchGate-Paper-00CCBB?logo=ResearchGate&logoColor=white" alt="ResearchGate" style="height: 25px; margin-right: 5px;">
  </a>
  <a href="https://paperswithcode.com/paper/z-saslm-zero-shot-style-aligned-sli-blending">
    <img src="https://img.shields.io/badge/Papers%20with%20Code-Enabled-9cf?logo=paperswithcode&logoColor=white" alt="Papers with Code" style="height: 25px; margin-right: 5px;">
  </a>
  <a href="https://www.academia.edu/128519694/Z_SASLM_Zero_Shot_Style_Aligned_SLI_Blending_Latent_Manipulation">
    <img src="https://img.shields.io/badge/Academia-Visit-blue" alt="Academia.edu" style="height: 25px;">
  </a>
</p>


---
### Copyright © 2025 Alessio Borgi, Fabrizio Silvestri, Pietro Liò


**Polynomial Sheaf Diffusion (PolySD)** is an optimisation of **Neural Sheaf Diffusion**, obtained by replacing fixed spectral filters with **orthogonal-polynomial spectral filters**. This lets you shape diffusion dynamics with families like **Chebyshev (Types I–IV), Legendre, Gegenbauer, and Jacobi**, on both **discrete** and **ODE (continuous-time)** sheaf diffusion variants, and across **Diagonal / Bundle / General** sheaf maps.

This optimisation allows to reach new **state-of-the-art accuracy performances** in **both Homophilic** and **Heterophilic Benchmarks**, and using way **less number of parameters** with higher performances with respect to Neural Sheaf Diffusion, even with NSD having more layers or hidden channels.

<p align="center">
  <img width="605" height="952" alt="Screenshot 2025-11-04 at 22 39 36" src="https://github.com/user-attachments/assets/d9f3d7cb-adc4-477b-b144-fbeef52fe300" />
</p>

---

## Table of Contents

- [1. Highlights](#highlights)
- [2. Installation](#installation)
- [3. Quick Start](#quick-start)
- [4. Project Layout](#project-layout)
- [5. Model Families](#model-families)
- [6. Polynomial Filters](#polynomial-filters)
- [7. Configuration (YAML) and CLI](#configuration-yaml-and-cli)
- [8. Hyperparameter Sweeps with W&B](#hyperparameter-sweeps-with-wb)
- [9. Datasets](#datasets)
- [10. Citing](#citing)
- [11. License](#license)

---

## 1. Highlights

- **Passenger Count Predictions**: For the baselines, run `tokyo.py`. For the sheaf extensions, run `run_tokyo.sh`. Instructions for specific details of running shell scripts and sweeps are provided in subsequent sections.
- **Polynomial Spectral Filters**: Chebyshev Types I–IV, Legendre, Gegenbauer, Jacobi.
- **Discrete & ODE (Continuous-time)** versions of Sheaf Diffusion.
- **Three Sheaf-map Regimes**: Diagonal, Bundle, General.
- **Heterophily-Ready**: Plug-and-play on benchmarks like **Texas**.
- **W&B Sweeps**: Reproducible experiment management.
- **Unit Tests**: Quick sanity checks (`pytest -v .`).

---

## 2. Installation

We used `CUDA 10.2`. Create and activate the environment:

```bash
conda env create --file=environment_gpu.yaml
conda activate nsd
```

- To use a different CUDA version, edit `environment_gpu.yaml`.
- For **CPU-only**, use `environment_cpu.yaml` instead.

Validate your setup:

```bash
pytest -v .
```

---

## 3. Quick Start

### Local run (without W&B)

Disable W&B once:

```bash
wandb disabled
```

Run the provided **Texas** example:

```bash
sh ./exp/scripts/run_texas.sh
```

### With W&B (recommended)

1) Log in:

```bash
wandb online
wandb login
```

2) Run the example (Texas):

```bash
export ENTITY=<WANDB_ACCOUNT_ID>
sh ./exp/scripts/run_texas.sh
```

> Additional dataset scripts live in `exp/scripts/`.

---

## 4. Project Layout

```
exp/
  run.py                     # Entry point for single runs
  scripts/
    run_texas.sh             # Example run script
    sweeps/                  # W&B sweep YAMLs

models/
  disc_models.py             # Discrete *Polynomial* sheaf diffusion models
  cont_models.py             # ODE (continuous) *Polynomial* sheaf diffusion models

utils/
  ...                        # Datasets, logging, transforms, helpers, etc.
```

---

## 5. Model Families

Use the `--model` flag (or the `parameters.model.value` in sweeps) to select a variant.

**Discrete (layered):**
- `DiagSheafPolynomial` – Diagonal sheaf maps (fastest).
- `BundleSheafPolynomial` – Block-structured (edge bundle) transports.
- `GeneralSheafPolynomial` – Full general transports (most expressive).

**Continuous-time (ODE):**
- `DiagSheafODEPolynomial`  
- `BundleSheafODEPolynomial`  
- `GeneralSheafODEPolynomial`

> Baselines (NSD): `DiagSheaf`, `BundleSheaf`, `GeneralSheaf`, and their `*ODE` counterparts (non-polynomial).

---

## 6. Polynomial Filters

PolySD applies a polynomial filter of degree `K` to a rescaled Laplacian operator. Indeed, every Polynomial Shead Diffusion layer, gets in input the sheaf laplacian (in its normalised or non normalised version), together with `lambda-max`, which is the upper bound of the spectrum. Its value depends on the typology of sheaf laplacian:
- If `sheaf-laplacian=normalised`: The lambda max is set as:  `lambda-max = 2`, since the sepctrum is bounded in `[0,2]`.
- If `sheaf-laplacian=unnormalised`: The lambda max gets the value depending on the choice we we here: 
    - `lambda_max_choice=analytic` Uses a known bound (e.g., **2** for normalized Laplacians) or closed-form where available. It is based on the Gershgorin's Theorem. 
    - `lambda_max_choice=iterative` Estimates \(\lambda_{\max}\) via power iteration, being a safer solution for **non-standard / sheaf Laplacians**.

**Supported Orthogonal Families**

| `polynomial_type`  | Symbol                   | Interval | Constraints                         |
|--------------------|--------------------------|----------|-------------------------------------|
| `ChebyshevType1`   | \(T_k\)                  | \([-1,1]\) | –                                   |
| `ChebyshevType2`   | \(U_k\)                  | \([-1,1]\) | –                                   |
| `ChebyshevType3`   | \(V_k\)                  | \([-1,1]\) | –                                   |
| `ChebyshevType4`   | \(W_k\)                  | \([-1,1]\) | –                                   |
| `Legendre`         | \(P_k\)                  | \([-1,1]\) | –                                   |
| `Gegenbauer`       | \(C_k^{(\lambda)}\)      | \([-1,1]\) | `gegenbauer_lambda > 0`             |
| `Jacobi`           | \(P_k^{(\alpha,\beta)}\) | \([-1,1]\) | `jacobi_alpha > -1`, `jacobi_beta > -1` |

**Practical Tips**
- Begin with `K ∈ {4, 8, 12}`; higher `K` increases capacity **and** cost.
- Prefer `iterative` lambda for **General** sheaf Laplacians or custom operators.
- Gegenbauer/Jacobi add response-shape control — scan a few values (e.g., λ ∈ {0.5, 1.0, 1.5}).
- In `homophilic settings` prefer `smaller K`, while for `heterophilic settings` prefer `larger K`. 

---

## 7: Configuration (YAML) and CLI

The YAML configurations are already set and are **sweep-friendly**. The part that concerns the Polynomial Filters is shown here. Remember to comment/uncomment the gagenbauer and jacobi parameters when using their family.  

```yaml
name: texas
program: exp/run.py
method: random

metric:
  name: val_acc
  goal: maximize

parameters:
  dataset:
    value: texas
  model:
    value: GeneralSheafODEPolynomial     # Options:
                                         # [DiagSheaf, BundleSheaf, GeneralSheaf,
                                         #  DiagSheafPolynomial, BundleSheafPolynomial, GeneralSheafPolynomial,
                                         #  DiagSheafODE, BundleSheafODE, GeneralSheafODE,
                                         #  DiagSheafODEPolynomial, BundleSheafODEPolynomial, GeneralSheafODEPolynomial]

  ########## POLYNOMIAL-ONLY PARAMETERS ####################################
  polynomial_type:
    value: ChebyshevType1                # [ChebyshevType1, ChebyshevType2, ChebyshevType3,
                                         #  ChebyshevType4, Legendre, Gegenbauer, Jacobi]

  lambda_max_choice:
    value: analytic                      # ["iterative", "analytic"]

  poly_layers_K:                         # Filter degree
    distribution: categorical
    values: [2, 3, 4, 5, 8, 12, 16]

  # Enable these when using the respective families:
  # gegenbauer_lambda:                    # Must be > 0
  #   distribution: categorical
  #   values: [0.5, 1.0, 1.5]
  # jacobi_alpha:                         # Must be > -1
  #   distribution: categorical
  #   values: [-0.5, 0.0, 0.5, 1.0]
  # jacobi_beta:                          # Must be > -1
  #   distribution: categorical
  #   values: [-0.5, 0.0, 0.5, 1.0]
```

### CLI one-liners

Here follows some examples for running in one line, a single run.

**Discrete, General sheaf, Legendre K=8 on Texas**
```bash
python exp/run.py \
  --dataset texas \
  --model GeneralSheafPolynomial \
  --polynomial_type Legendre \
  --poly_layers_K 8 \
  --lambda_max_choice analytic
```

**ODE (continuous), General sheaf, Chebyshev Type I K=12 on Texas**
```bash
python exp/run.py \
  --dataset texas \
  --model GeneralSheafODEPolynomial \
  --polynomial_type ChebyshevType1 \
  --poly_layers_K 12 \
  --lambda_max_choice analytic
```

**Gegenbauer (λ=1.0)**
```bash
python exp/run.py \
  --dataset texas \
  --model DiagSheafPolynomial \
  --polynomial_type Gegenbauer \
  --gegenbauer_lambda 1.0 \
  --poly_layers_K 8 \
  --lambda_max_choice iterative
```

**Jacobi (α=0.0, β=0.5)**
```bash
python exp/run.py \
  --dataset texas \
  --model BundleSheafPolynomial \
  --polynomial_type Jacobi \
  --jacobi_alpha 0.0 \
  --jacobi_beta 0.5 \
  --poly_layers_K 8 \
  --lambda_max_choice iterative
```

---

## 8: Hyperparameter Sweeps with W&B

**Create a sweep** (example project name `PolySD_Texas`): In order to find the best hyperparameter settings, it is useful to run a sweep and save them to your wandb account. 

```bash
export ENTITY=<WANDB_ACCOUNT_ID>
wandb sweep --project PolySD_Texas exp/scripts/sweeps/texas_sweep.yaml
```

**Run on a single GPU:**

```bash
wandb agent <ENTITY>/<PROJECT>/<SWEEP_ID>
```

**Run on multiple GPUs:**

```bash
sh run_sweeps.sh <SWEEP_ID>
```

> **Note:** If W&B warns about `log_uniform` using `min/max`, switch to `log_uniform_values` in your YAML to specify limit values directly.


In the following, we present a **Hyperparameter Full Reference**:

> All flags are parsed in `exp/run.py` via `argparse`.  
> **Booleans** accept `true/false`, `yes/no`, `1/0` (handled by `str2bool`).  
> **Lists** use comma-separated values (e.g., `--classes_corr 0.1,0.3,0.6`).

### 1) Optimisation

| Flag | Type / Choices | Default | Description |
|---|---|---:|---|
| `--epochs` | int | `1500` | Number of training epochs. |
| `--lr` | float | `0.01` | Learning rate. |
| `--weight_decay` | float | `0.0005` | Optimizer weight decay (L2). |
| `--sheaf_decay` | float or `None` | `None` | Optional decay/regularization for sheaf parameters. |
| `--early_stopping` | int | `200` | Patience for early stopping. |
| `--min_acc` | float | `0.0` | Minimum **test** acc on the first fold to continue training. |
| `--stop_strategy` | `loss` \| `acc` | `loss` | Validation criterion used for early stopping. |

---

### 2) Model Configuration

| Flag | Type / Choices | Default | Description |
|---|---|---:|---|
| `--d` | int | `2` | Base fiber dimension. |
| `--layers` | int | `2` | Number of diffusion layers. |
| `--normalised` | bool | `True` | Use a **normalised** Laplacian. |
| `--deg_normalised` | bool | `False` | Use a **degree-normalised** Laplacian. |
| `--linear` | bool | `False` | Learn a new Laplacian at each step. |
| `--hidden_channels` | int | `20` | Hidden channel size. |
| `--input_dropout` | float | `0.0` | Dropout on inputs. |
| `--dropout` | float | `0.0` | Dropout inside the network. |
| `--left_weights` | bool | `True` | Apply left linear layer. |
| `--right_weights` | bool | `True` | Apply right linear layer. |
| `--add_lp` | bool | `False` | **Adds fixed high-pass filter** in the restriction maps. *(as in source help)* |
| `--add_hp` | bool | `False` | **Adds fixed low-pass filter** in the restriction maps. *(as in source help)* |
| `--use_act` | bool | `True` | Apply activation in the layer. |
| `--second_linear` | bool | `False` | Add a second linear layer. |
| `--orth` | `matrix_exp` \| `cayley` \| `householder` \| `euler` | `householder` | Parametrisation for the orthogonal group. |
| `--sheaf_act` | str | `tanh` | Activation used in the sheaf learner. |
| `--edge_weights` | bool | `True` | Learn edge weights for connection Laplacian. |
| `--sparse_learner` | bool | `False` | Use sparse learner variant (if implemented). |

---

### 3) Experiment Setup

| Flag | Type / Choices | Default | Description |
|---|---|---:|---|
| `--dataset` | str | `texas` | Dataset name. |
| `--seed` | int | `43` | Random seed. |
| `--cuda` | int | `1` | CUDA device index. |
| `--folds` | int | `10` | Number of cross-validation folds. |
| `--model` | see list below | `None` | Model variant (discrete/ODE; diag/bundle/general; polynomial/non-polynomial). |
| `--entity` | str or `None` | `None` | Weights & Biases entity (account/workspace). |
| `--evectors` | int | `0` | Number of Laplacian positional-encoding eigenvectors to use. |

**Allowed `--model` values**

- **Discrete baseline:** `DiagSheaf`, `BundleSheaf`, `GeneralSheaf`  
- **Discrete polynomial:** `DiagSheafPolynomial`, `BundleSheafPolynomial`, `GeneralSheafPolynomial`  
- **ODE baseline:** `DiagSheafODE`, `BundleSheafODE`, `GeneralSheafODE`  
- **ODE polynomial:** `DiagSheafODEPolynomial`, `BundleSheafODEPolynomial`, `GeneralSheafODEPolynomial`

---

### 4) Polynomial Filters

| Flag | Type / Choices | Default | Constraints / Notes |
|---|---|---:|---|
| `--lambda_max_choice` | `analytic` \| `iterative` | `analytic` | How to estimate/upper-bound λ\_max. Prefer `iterative` for sheaf Laplacians. |
| `--polynomial_type` | `Chebyshev` \| `ChebyshevType1/2/3/4` \| `Legendre` \| `Gegenbauer` \| `Jacobi` | `ChebyshevType1` | Orthogonal family for spectral filtering. |
| `--poly_layers_K` | int | `3` | Order \(K\) of the polynomial filter (preferred). |
| `--chebyshev_layers_K` | int | `3` | **Deprecated** (legacy Chebyshev order; kept for compatibility). |
| `--gegenbauer_lambda` | float | `1.0` | Gegenbauer λ; **must be > 0**. |
| `--jacobi_alpha` | float | `0.0` | Jacobi α; **must be > −1**. |
| `--jacobi_beta` | float | `0.0` | Jacobi β; **must be > −1**. |

> **Tip:** For normalized Laplacians, `λ_max ≈ 2` → `--lambda_max_choice analytic` is often fine.  
> For **sheaf Laplacians** or custom operators, prefer `--lambda_max_choice iterative`.

---

### 5) ODE (Continuous-time) Controls

| Flag | Type / Choices | Default | Description |
|---|---|---:|---|
| `--max_t` | float | `1.0` | Maximum integration time. |
| `--int_method` | str | — | ODE solver: e.g., `dopri5`, `euler`, `rk4`, `midpoint`. |
| `--step_size` | float | `1` | Fixed step size for fixed-step solvers. |
| `--max_iters` | float | `100` | Maximum number of integration steps. |
| `--adjoint_method` | str | `adaptive_heun` | Backward solver for adjoint pass. |
| `--adjoint` | flag | `False` | Use adjoint ODE method to reduce memory footprint. |
| `--adjoint_step_size` | float | `1` | Fixed step size for adjoint solvers. |
| `--tol_scale` | float | `1.0` | Multiplier for absolute/relative tolerances. |
| `--tol_scale_adjoint` | float | `1.0` | Multiplier for adjoint tolerances. |
| `--max_nfe` | int | `1000` | Max number of function evaluations per epoch. |
| `--no_early` | flag | `False` | Disable early stopping of the ODE integrator at test time. |
| `--earlystopxT` | float | `3` | Multiplier for `T` to evaluate best model. |

---

### 6) Synthetic / Legacy Experiment Controls

| Flag | Type / Choices | Default | Description |
|---|---|---:|---|
| `--max_test_steps` | int | `100` | Max steps for `dopri5Early` test integrator (avoid OOM). |
| `--maps_lr` | float or `None` | `None` | Optional separate LR for map parameters. |
| `--classes_corr` | list\<float> | `None` | Class correlation vector (comma-separated). |
| `--num_nodes` | int | `200` | Synthetic graph: number of nodes. |
| `--num_classes` | int | `2` | Synthetic labels: number of classes. |
| `--het_coef` | float | `0.9` | Heterophily coefficient. |
| `--edge_noise` | float | `0.05` | Edge noise level. |
| `--node_degree` | int | `10` | Target node degree. |
| `--num_feats` | int | `10` | Number of node features. |
| `--new_synth_data` | bool | `False` | Regenerate synthetic features. |
| `--feat_noise` | float | `0.25` | Feature noise level. |
| `--new_synth_edges` | bool | `False` | Regenerate synthetic edges. |
| `--use_epsilons` | bool | `True` | Use epsilon regularization/offsets (if supported). |
| `--use_embedding` | bool | `True` | Use feature embedding. |
| `--ellipsoid_radius` | float | `1` | Synthetic ellipsoid radius. |
| `--just_add_noise` | bool | `False` | Only add noise to the synthetic setup. |
| `--ellipsoids` | bool | `True` | Use ellipsoids in synthetic generator. |
| `--rotation_invariant_sheaf_learner` | bool | `False` | Use RotationInvariantSheafLearner (O(d) maps). |
| `--node_edge_sims_time_dependent` | bool | `False` | Time-dependent node–edge similarity features (for rotation-invariant learner). |




---

## 9: Datasets

- Common heterophily datasets (e.g., **Texas**) are prepared by the scripts under `exp/scripts/`.
- To generate synthetic data:

```bash
python exp/run.py --dataset synthetic_exp  [other flags...]
```

---

## 9.5: Ablation Study: Dirichlet Energy Experiment
First thing first, we need to build the results with: 
```python visualizations/dirichlet.py collect  --device cuda:1```

Then, we can plot them out:
```python visualizations/dirichlet.py plot  --device cuda:1```

## 10: Oversquashing Experiment
First thing first, we need to build the results with: 
```python visualizations/oversquashing.py   --out_dir visualizations/figures/oversquashing/   --device cuda:0   --epochs 300   --early_stopping 50   --max_hops 10   --num_targets 128   --target_split test   --layers_nsd 6   --layers_poly 2   --poly_layers_K 10   --hidden_channels 32   --dropout 0.3   --lr 0.01   --weight_decay 5e-4```

## 11: PolySpecGNN Baseline
In order to run the baseline model, we will need to run: 
```wandb sweep models/tests_and_experiments/baseline/chebgnn_sweep.yaml```

## 12: Long-Range Benchmark: City-Network
In roder to run the benchmark CityNetwork, we will need to run: 
```wandb sweep -p CityNetwork_Influence_RF models/tests_and_experiments/city_networks_long_range/long_range.yaml```

## 13: Citing
For citing the **Polynomial Sheaf Diffusion** paper:

```
@misc{polysd2025,
  title={Polynomial Sheaf Diffusion},
  author={Alessio Borgi and collaborators},
  year={2025},
  note={Code: https://github.com/<user>/Polynomial-Sheaf-Diffusion}
}
```

This repository is based in part on the **Neural Sheaf Diffusion** paper:

```
@inproceedings{bodnar2022neural,
  title={Neural Sheaf Diffusion: A Topological Perspective on Heterophily and Oversmoothing in {GNN}s},
  author={Cristian Bodnar and Francesco Di Giovanni and Benjamin Paul Chamberlain and Pietro Li{\`o} and Michael M. Bronstein},
  booktitle={Advances in Neural Information Processing Systems},
  year={2022},
  url={https://openreview.net/forum?id=vbPsD-BhOZ}
}
```

---

## 10.5: DOCKER CONTAINER:**
1. Build the image:
   ```bash
   podman build -t hetero-polysd:latest .
   ```
2. Run a shell with GPU access (adjust volume paths for your system):
   ```bash
   podman run -it --rm \
     -v /home/$USER/PolySheafNeuralNetworks-RailwayPassengers:/work/project \
     -v /mnt/ssd2/$USER/hetero:/work/data \
     --device nvidia.com/gpu=all \
     --ipc host \
     hetero-polysd:latest \
     /bin/bash
   ```
3. Inside the container, launch sweeps as usual:
   ```bash
   wandb sweep --project PolySheafNeuralNetworks-RailwayPassengers sweeps/nc/dblp/diag_sheaf.yaml
   wandb agent sheaf_hypergraphs/PolySheafNeuralNetworks-RailwayPassengers/<SWEEP_ID>
   ```

For mac, instead: 
```
docker build --platform linux/amd64 -t hetero-polysd:latest .
```
and then: 
```
docker run --platform linux/amd64 --rm -it \
  -v /Users/alessioborgi/Documents/GitHub/Heterogeneous-Polynomial-Sheaf-Diffusion:/work/project \
  hetero-polysd:latest \
  /bin/bash
```

### Development Dependencies

Install additional development tools:
```bash
uv sync --group dev
```

This includes `pytest` and `pytest-cov` for testing.

---


## 11: License

See `LICENSE` for details.

## ACKNOWLEDGEMENTS

Thanks to Leo-Minh Kustermann for his help in cleaning the code!

















