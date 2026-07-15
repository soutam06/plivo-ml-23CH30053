# Run Log

## Run 1: Baseline
- **Hypothesis**: The baseline model uses standard components (LayerNorm, GELU, absolute positional embeddings, untied weights) and a constant learning rate (3e-4) with vanilla Adam. It should perform poorly because of the lack of learning rate scheduling and parameter inefficiencies.
- **Changes**: None.
- **Score (BPB)**: 2.3718
- **Conclusion**: As expected, the baseline is suboptimal. The constant learning rate and lack of weight decay hinder convergence.

## Run 2: AdamW + OneCycleLR
- **Hypothesis**: Switching to AdamW (with weight decay 0.01) and adding a OneCycleLR scheduler (warmup + cosine decay with max_lr=1e-3) and gradient clipping will allow the model to learn much faster within the 2000 step budget and regularize better.
- **Changes**: Modified `train.py` to use AdamW, OneCycleLR, and `clip_grad_norm_`.
- **Score (BPB)**: TBD
- **Conclusion**: TBD
