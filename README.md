# scPTG

Minimal, reproducible implementation of **scPTG** for single-cell clustering. This release contains only the formal method and the code required to train it and report ARI, NMI and ACC. Plotting, ablation variants and batch experiment scripts are intentionally excluded.

## Method

scPTG combines a masked denoising autoencoder, a rank-weighted PCA KNN graph, multiscale Louvain initialization, an anchored graph neural network, a shared cosine-prototype head and guarded trajectory readout.

## Repository layout

```text
scPTG/
├── config.yaml           # Default Zeisel hyperparameters
├── main.py               # Single training/evaluation entry point
├── requirements.txt
├── data/Zeisel.h5ad      # Example dataset
├── log/                  # Timestamped training logs
├── outputs/              # Metrics and assignments (created at runtime)
└── scptg/
    ├── data.py           # h5ad loading and preprocessing
    ├── graph.py          # KNN graph construction
    ├── community.py      # Multiscale community initialization
    ├── model.py          # AE, anchored GNN and prototype head
    ├── trainer.py        # Formal scPTG training path
    ├── metrics.py        # ARI, NMI and ACC
    └── utils.py          # Reproducibility, logging and JSON helpers
```

## Installation

Python 3.10 or newer is recommended.

```bash
pip install -r requirements.txt
```

For a CUDA-enabled GPU, install the matching PyTorch build first by following the official PyTorch instructions.

## Run

From the repository root:

```bash
python main.py
```

The default command runs Zeisel with seed 0. To select the device or perform a short installation check:

```bash
python main.py --device cuda:0
python main.py --device cpu --pretrain-epochs 1 --joint-epochs 1 --output smoke_outputs
```

Each run writes:

- `metrics.json`: ARI, NMI, ACC, runtime and graph/readout diagnostics;
- `history.csv`: epoch-level training losses;
- `assignments.npz`: predictions, labels, probabilities and final embeddings;
- `config.json`: the exact effective configuration;
- `log/*.log`: the complete console log.

The ground-truth labels are not used as an optimization target. They provide the known cluster count and are read after training to calculate evaluation metrics.

