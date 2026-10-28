# Neural Network Eigenfunction Analysis

This project studies how different neural-network parameterizations and geometric domains shape the spectrum and eigenfunctions of the Neural Tangent Kernel (NTK) and the Neural Network Gaussian Process (NNGP).

The code generates synthetic 2D binary classification datasets from several shapes, trains small networks in JAX/Flax, and then evaluates kernel alignment, eigenvalue spectra, continuous eigenfunctions, and diagonal diagnostics. It is designed for research experiments and visualization-heavy analysis rather than production deployment.

## Project purpose

The project explores the relationship between:

- input geometry (circle, heart, star, astroid, xor circles)
- network initialization scheme (standard, NTK, maximal update)
- network width and depth
- torus projection of 2D inputs into a 4D embedding
- kernel structure and spectral behavior during training

The intended use is to study whether different architectures produce different eigenfunction structures and whether particular parameterizations lead to more stable or more collapsed spectra.

## Main workflow

The execution flow in `main.py` is:

1. Build synthetic training, validation, and evaluation sets.
2. Compute signed margins to the shape boundary and sort the evaluation set by margin.
3. Optionally project 2D coordinates onto a torus feature map.
4. Instantiate one of three model classes.
5. Train the model with the `Trainer` loop.
6. Evaluate kernel matrices and eigenvalue spectra via `KernelEvaluator`.
7. Log metrics and figures to Weights & Biases.

## Repository structure

```text
nn_eigendecomposition/
├── main.py
├── README.md
├── pyproject.toml
├── experiments.txt
├── plots/
├── src/
│   ├── data.py
│   ├── eval.py
│   ├── geometry.py
│   ├── kernels.py
│   ├── model.py
│   ├── plots.py
│   └── trainer.py
└── wandb/
```

## Module overview

### `main.py`

This is the entry point for running experiments. It:

- parses command-line arguments with `argparse`
- prepares train/validation/evaluation datasets
- logs the label identity matrix
- chooses a model based on `--init`
- creates a `KernelEvaluator` and `Trainer`
- runs the experiment and logs to W&B

Relevant CLI arguments include:

- `--shape`: circle, xor_circles, heart, star, astroid
- `--init`: NTK, standard, max
- `--width`: hidden width
- `--learning_rate`
- `--batch_size`
- `--epochs`
- `--seed`
- `--num_hidden_layers`
- `--torus`: project points to torus features

Important note: in the current file, the W&B project name is still a placeholder:

```python
wandb.init(
    project="ENTER NAME OF PROJECT HERE",
    ...
)
```

You should replace this with your own W&B project before running the script.

### `src/data.py`

Responsible for synthetic data generation and batching.

Key functions/classes:

- `generate_data(num_samples, seed, noise=0, shape="circle")`
  - samples points uniformly on `[-2, 2]^2`
  - computes the boundary margin for the chosen shape
  - labels positive/negative by whether the point is outside the shape boundary
- `DataLoader`
  - iterable batching wrapper with optional shuffling and `drop_last`
- `get_toy(batch_size, num_samples, seed, shape)`
  - returns a dataset pair and a loader for toy experiments

### `src/geometry.py`

Defines the boundary margin function for each geometric shape.

Supported shapes:

- `circle`
- `heart`
- `star`
- `astroid`
- `xor_circles`

The function returns a signed margin, where:

- positive values indicate points outside the shape
- negative values indicate points inside the shape
- values near zero lie near the boundary

### `src/model.py`

Contains the network architectures used in the experiments.

Included model families:

- `NTKModel`: NTK-parameterized linear layers
- `StandardNetwork`: standard neural network with normal initialization
- `MaximalUpdateNetwork`: a maximal-update parameterization network using a custom first layer

The custom `NTKLinear` and `MuPFirstLayer` classes implement alternative scaling schemes to match the experiment design.

### `src/kernels.py`

Computes empirical kernel matrices for the model.

Functions:

- `get_ntk(model, x1, x2=None, **kwargs)`
  - returns the NTK matrix for `x1` and optionally a cross-kernel matrix with `x2`
- `get_nngp(model, x1, x2=None, **kwargs)`
  - computes the NNGP kernel matrix from flattened predictions

This is the core mathematical object used for the spectral analysis.

### `src/eval.py`

Implements the evaluation pipeline. `KernelEvaluator` computes:

- NTK and NNGP matrices
- kernel alignment metrics
- eigenvalue spectra
- effective rank
- continuous eigenfunctions on a dense grid
- diagonal slice diagnostics
- collapse metrics related to spectral concentration

Most visualization-heavy analysis happens here, and the evaluator sends images and scalars to W&B.

### `src/plots.py`

Provides plotting helpers for:

- kernel heatmaps
- label identity matrices
- boundary visualization overlays

It uses the `Agg` matplotlib backend so plots can be generated in headless environments.

### `src/trainer.py`

Contains the training loop and logging logic.

The `Trainer` class:

- holds the model, optimizer, data loaders, and evaluator
- runs training steps and validation steps
- logs training/validation loss and accuracy
- triggers deep evaluation periodically at fixed epoch intervals

The optimizer is SGD with a configurable learning rate.

## Dependencies

This project depends on:

- JAX + JAXLIB
- Flax
- Optax
- Matplotlib
- W&B
- NumPy / JAX NumPy
- `datasets` is also referenced in prior code patterns, though the current `main.py` workflow is primarily synthetic-data based

The environment is configured via `pyproject.toml`.

## Quick start

### 1. Install dependencies

```bash
cd nn_eigendecomposition
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

or with `uv`:

```bash
cd nn_eigendecomposition
uv sync
```

### 2. Login to W&B

```bash
wandb login
```

### 3. Run an experiment

Example:

```bash
python main.py --shape circle --init NTK --width 100 --learning_rate 1.0 --batch_size 100 --epochs 1000 --seed 42 --num_hidden_layers 1
```

Torus-enabled example:

```bash
python main.py --shape star --init standard --width 100 --learning_rate 1.0 --batch_size 100 --epochs 1000 --seed 42 --num_hidden_layers 2 --torus
```

## Expected outputs

The project logs the following to W&B:

- loss / accuracy curves
- kernel heatmaps
- eigenvalue spectrum plots
- continuous eigenfunction contours
- diagonal NTK diagnostics
- collapse metrics
