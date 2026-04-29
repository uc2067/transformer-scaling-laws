# Scaling Laws for Language Models Trained on SVG Code

**CS-GY 6923: Machine Learning — Optional Project**
New York University Tandon School of Engineering

## Overview

This project investigates neural scaling laws for decoder-only Transformer language models trained on SVG (Scalable Vector Graphics) code. We train 5 models ranging from 1.87M to 91.54M parameters, fit power-law scaling curves, compare standard parameterization (SP) vs. Maximal Update Parameterization (µP), and generate SVG samples from our best model.

## Key Results

| Model | Params | SP Val Loss | µP Val Loss |
|-------|--------|------------|-------------|
| Tiny | 1.87M | 0.5878 | 0.6458 |
| Small | 4.29M | 0.5788 | 0.6671 |
| Medium | 13.89M | 0.6102 | 0.6571 |
| Large | 35.85M | 0.5967 | 0.6726 |
| XL | 91.54M | — | 0.6968 |

- **Scaling exponent α = 0.0003** (flat — all models compute-limited after 1 epoch)
- **Best SP LR: 0.01** | **Best µP LR: 0.03**
- **Test Perplexity: 1.84**
- **XML Validity Rate: 60%** | **SVG Render Rate: 60%**

## Project Structure

```
svg-scaling-project/
├── README.md
├── requirements.txt
├── Part1_Data_Pipeline.ipynb       # Data collection, cleaning, tokenization
├── Part2_Scaling_Study.ipynb       # LR sweep, model training, scaling plot
├── Part3_muP.ipynb                 # µP implementation, comparison, extrapolation
├── Part4_Generation.ipynb          # Sample generation and evaluation
└── results/
    ├── all_results.json            # SP training results
    └── all_mup_results.json        # µP training results
```

## Setup

### Requirements

```bash
pip install -r requirements.txt
```

### Requirements File

See `requirements.txt` for all dependencies.

### Data

Data is loaded automatically from HuggingFace in Part 1:
- `starvector/svg-icons-simple` (~64k SVGs)
- `starvector/svg-emoji-simple` (~2k SVGs)
- `starvector/svg-fonts-simple` (~500k SVGs, streamed)
- `starvector/svg-stack-simple` (~10k SVGs)

Total: 576,776 SVGs → 277M training tokens

## Reproducing Results

### Part 1: Data Pipeline
Run `Part1_Data_Pipeline.ipynb` in Google Colab. Saves processed data to Google Drive.

### Part 2: Scaling Study
Run `Part2_Scaling_Study.ipynb`. LR sweep runs in ~5 minutes. Full 1-epoch training was run on Kaggle T4 x2 GPU (~6 hours total). Checkpoints saved to Google Drive.

### Part 3: µP
Run `Part3_muP.ipynb` in Google Colab. µP LR sweep runs in ~5 minutes. Full training was run on Colab H100 (~3 hours total).

### Part 4: Generation
Run `Part4_Generation.ipynb` in Google Colab. Loads best model checkpoint from Drive and generates samples.

## Model Architecture

All models use a decoder-only Transformer (GPT-style):
- Vocabulary size: 4,096 (BPE tokenizer)
- Context window: 256 tokens
- Batch size: 32 sequences
- Optimizer: AdamW (β1=0.9, β2=0.95, wd=0.1)
- LR schedule: Cosine with 200 warmup steps
- Training: Exactly 1 epoch (33,826 iterations on 277M tokens)

## µP Implementation

µP (Maximal Update Parameterization) is implemented using the [microsoft/mup](https://github.com/microsoft/mup) package. Key changes from standard parameterization:
1. Attention scaling: `1/d` instead of `1/sqrt(d)`
2. Output layer: `MuReadout` instead of `nn.Linear`
3. Optimizer: `MuAdamW` instead of `AdamW`

## References

- Kaplan et al. (2020): Scaling Laws for Neural Language Models
- Hoffmann et al. (2022): Training Compute-Optimal Large Language Models (Chinchilla)
- Yang et al. (2022): Tensor Programs V: Zero-Shot Hyperparameter Transfer (µP)
- Rodriguez et al. (2023): StarVector: Generating SVG Code from Images and Text
- Karpathy (2022): nanoGPT