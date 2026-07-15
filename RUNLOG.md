# Run Log

## Run 1: Baseline
- **Hypothesis**: The baseline model uses standard components (LayerNorm, GELU, absolute positional embeddings, untied weights) and a constant learning rate (3e-4) with vanilla Adam. It should perform poorly because of the lack of learning rate scheduling and parameter inefficiencies.
- **Changes**: None.
- **Score (BPB)**: 2.3718
- **Conclusion**: As expected, the baseline is suboptimal. The constant learning rate and lack of weight decay hinder convergence.

## Run 2: AdamW + OneCycleLR
- **Hypothesis**: Switching to AdamW (with weight decay 0.01) and adding a OneCycleLR scheduler (warmup + cosine decay with max_lr=1e-3) and gradient clipping will allow the model to learn much faster within the 2000 step budget and regularize better.
- **Changes**: Modified `train.py` to use AdamW, OneCycleLR, and `clip_grad_norm_`.
- **Score (BPB)**: 2.2347
- **Conclusion**: The scheduler and AdamW provided a significant boost, dropping the BPB from 2.3718 to 2.2347.

## Run 3: Architectural Swaps (RMSNorm + SwiGLU + Tie Weights)
- **Hypothesis**: Replacing standard LayerNorm with RMSNorm and the GELU MLP with a SwiGLU activation (with parameter parity) improves parameter efficiency. Tying token and output embeddings frees up parameters that can be better utilized by the network.
- **Changes**: Modified `model.py` to use `RMSNorm`, replaced `Block.mlp` with `SwiGLU(cfg.n_embd)`, and set `tie_weights = True` in `Config`.
- **Score (BPB)**: TBD
- **Conclusion**: TBD
