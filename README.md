# Scaling Laws for Language Models Trained on SVG Code

**CS-GY 6923: Machine Learning — Optional Project**  
New York University Tandon School of Engineering

## Overview

This project investigates neural scaling laws for decoder-only Transformer language models trained on SVG (Scalable Vector Graphics) code. We train 5 models ranging from 1.87M to 91.54M parameters, fit power-law scaling curves, compare standard parameterization (SP) vs. Maximal Update Parameterization (µP), and generate SVG samples from our best model.

SVG is an ideal testbed for scaling law research: it has a formal grammar with strict syntactic rules, a well-defined coordinate system, and — most importantly — outputs that can be instantly rendered and visually inspected, enabling both quantitative and qualitative evaluation.

---

## Key Results

| Model | Params | SP Val Loss | µP Val Loss | Train Time |
|-------|--------|------------|-------------|------------|
| Tiny | 1.87M | 0.5878 | 0.6458 | ~29 min (T4) |
| Small | 4.29M | 0.5788 | 0.6671 | ~67 min (T4) |
| Medium | 13.89M | 0.6102 | 0.6571 | ~153 min (T4) |
| Large | 35.85M | 0.5967 | 0.6726 | ~32 min (H100) |
| XL | 91.54M | — | 0.6968 | ~73 min (H100) |

| Metric | Value |
|--------|-------|
| Scaling exponent α | 0.0003 (flat — compute-limited regime) |
| Best SP Learning Rate | 0.01 |
| Best µP Learning Rate | 0.03 |
| Test Perplexity | 1.84 |
| XML Validity Rate | 60% |
| SVG Render Rate | 60% |

---

## Repository Structure

```
svg-scaling-project/
├── README.md
├── requirements.txt
├── Project.ipynb                   # Parts 1, 3, 4: Data pipeline, µP training, generation (Colab)
├── Kaggle-notebook.ipynb           # Part 2: LR sweep and full SP model training (Kaggle T4 x2)
└── results/
    ├── all_results.json            # SP training results (val_loss, params, curves for all models)
    ├── all_mup_results.json        # µP training results for all 5 model sizes
    └── part4_results.json          # Generation evaluation metrics
```

---

## Setup

```bash
pip install -r requirements.txt
```

Requires Python 3.9+ and a CUDA-capable GPU (T4 or better recommended).

---

## Data

Data is loaded automatically from HuggingFace in Part 1 of `Project.ipynb`:

| Dataset | SVGs Used | Description |
|---------|-----------|-------------|
| `starvector/svg-icons-simple` | ~64,811 | Simplified icon SVGs |
| `starvector/svg-emoji-simple` | ~1,965 | Simplified SVG emoji |
| `starvector/svg-fonts-simple` | ~500,000 | Font glyph SVGs (streamed) |
| `starvector/svg-stack-simple` | ~10,000 | Diverse SVGs (streamed) |
| **Total** | **576,776** | **277M training tokens** |

Preprocessing steps applied to all SVGs:
1. Strip XML comments
2. Normalize whitespace
3. Round coordinates to 1 decimal place (reduces length ~57%)
4. Filter by length (50–8,000 characters)
5. Validate with `lxml.etree`

---

## Reproducing Results

### Part 1: Data Pipeline
Run `Project.ipynb` (Cells 1–11) in **Google Colab**.
- Downloads datasets from HuggingFace
- Cleans and tokenizes SVGs
- Saves `train.bin`, `val.bin`, `test.bin` and tokenizer to Google Drive
- Takes ~30 minutes

### Part 2: SP Scaling Study
Run `Kaggle-notebook.ipynb` on **Kaggle (T4 × 2 GPU)**.
- Performs LR sweep on Tiny model (~5 minutes)
- Trains all 4 SP models for 1 full epoch (~6 hours total)
- Saves checkpoints and results to Google Drive

### Part 3: µP Training
Run `Project.ipynb` (Part 3 cells) in **Google Colab (H100 GPU)**.
- µP LR sweep on Tiny (~5 minutes)
- Trains all 5 µP models for 1 full epoch (~4 hours on H100)
- Saves `all_mup_results.json` to Google Drive

### Part 4: Generation and Evaluation
Run `Project.ipynb` (Part 4 cells) in **Google Colab**.
- Loads Large model checkpoint from Drive
- Generates 10 unconditional + 5 prefix-conditioned samples
- Computes test perplexity, XML validity, render rate

---

## Model Architecture

All models use a decoder-only Transformer (GPT-style), based on [nanoGPT](https://github.com/karpathy/nanoGPT):

| Hyperparameter | Value |
|---------------|-------|
| Vocabulary size | 2,838 (BPE) |
| Context window | 256 tokens |
| Batch size | 32 sequences (8,192 tokens/step) |
| Optimizer | AdamW (β1=0.9, β2=0.95, wd=0.1) |
| LR schedule | Cosine decay with 200 warmup steps |
| Training duration | 1 epoch = 33,826 iterations |
| Gradient clipping | 1.0 |

---

## µP Implementation

µP is implemented using the [microsoft/mup](https://github.com/microsoft/mup) package. Three changes from standard parameterization:

1. **Attention scaling**: `1/d` instead of `1/sqrt(d)`
2. **Output layer**: `MuReadout` instead of `nn.Linear`
3. **Optimizer**: `MuAdamW` instead of `AdamW`

Base shapes are set using a 64-dimensional proxy model with the same depth and number of heads as the target, ensuring only width varies.

---

## Key Findings

1. **Compute-limited scaling**: The nearly flat scaling exponent (α ≈ 0.0003) indicates all models are compute-limited after 1 epoch — consistent with Chinchilla (larger models are undertrained relative to their capacity).

2. **µP does not help at small scales**: SP outperforms µP at all model sizes tested. The LR transfer benefit of µP becomes significant only at much larger width ratios than explored here.

3. **SVG syntax is learnable**: The Large model achieves 60% XML validity and render rate, demonstrating that language models can learn SVG structure from data alone.

---

## References

- Kaplan et al. (2020): [Scaling Laws for Neural Language Models](https://arxiv.org/abs/2001.08361)
- Hoffmann et al. (2022): [Training Compute-Optimal Large Language Models (Chinchilla)](https://arxiv.org/abs/2203.15556)
- Yang et al. (2022): [Tensor Programs V: Zero-Shot Hyperparameter Transfer (µP)](https://arxiv.org/abs/2203.09789)
- Rodriguez et al. (2023): [StarVector: Generating SVG Code from Images and Text](https://arxiv.org/abs/2312.11556)
- Karpathy (2022): [nanoGPT](https://github.com/karpathy/nanoGPT)