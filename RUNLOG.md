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
- **Score (BPB)**: 2.2031
- **Conclusion**: The architectural swaps (SwiGLU, RMSNorm) and weight tying brought the parameter count down to 1.29M and improved the score to 2.2031.

## Run 4: QK-Norm and Parameter Scaling
- **Hypothesis**: Following "nanoGPT speedrun" best practices, applying RMSNorm to the Queries and Keys (QK-Norm) stabilizes attention, allowing for better gradient flow. Scaling the model parameters up from 1.29M to 1.83M (by increasing embedding dimensions to 192 and attention heads to 6) will utilize our 2M budget efficiently and maximize learning.
- **Changes**: Added QK-Norm to `SelfAttention`. Scaled `n_embd` to 192 and `n_head` to 6 in `Config`.
- **Score (BPB)**: 2.2648
- **Conclusion**: The BPB worsened! This was a classic ambitious failure. While the parameter count scaled perfectly to 1.84M, the model failed to converge as efficiently within the strict 2000-step budget. The larger model is less sample-efficient. Additionally, QK-Norm altered the attention logit scales, which would require a significantly retuned learning rate.

## Run 5: Reverting Scale and Pushing the Learning Rate
- **Hypothesis**: The 1.29M parameter model from Run 3 was our most sample-efficient architecture. By reverting the parameter scale and removing QK-Norm, we fix the loss observed in Run 4. To push beyond Run 3's score, we will increase the peak learning rate of our OneCycleLR scheduler from 1e-3 to 2e-3. Our RMSNorm and SwiGLU architecture should be robust enough to handle the faster learning rate without diverging.
- **Changes**: Reverted `n_embd=160` and `n_head=4` in `model.py`. Removed QK-Norm. Changed `max_lr=2e-3` in `train.py`.
- **Score (BPB)**: TBD
- **Conclusion**: TBD
