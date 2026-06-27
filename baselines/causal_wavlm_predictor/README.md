# causal_wavlm_predictor

A lightweight **causal** turn-taking predictor built on frozen WavLM
features, trained on individual speaker channels. Two model sizes are
provided: **WavLM-Base-Plus** (~94 M frozen backbone + 4 M predictor) and
**WavLM-Large** (~316 M frozen backbone + 4 M predictor).

## Architecture

```
WavLM (frozen, causal-masked) -> Conv1d subsample (stride 2) -> 4-layer causal Transformer (256d, 4h, ff1024) -> Linear -> 5 classes
```

- **Frontend:** CausalS3prlFrontend (WavLM with causal streaming mask
  injected into the self-attention), frozen during training.
- **Subsampling:** Conv1d stride-2, projects WavLM dim (768 or 1024) to 256.
- **Encoder:** 4-layer causal Transformer encoder (256d, 4 heads, FFN 1024).
- **Head:** Linear 256 -> 5 classes {C, NA, I, BC, T}.
- **Frame rate:** 25 Hz (40 ms stride), first prediction at 200 ms.

The model is fully causal: each frame's prediction depends only on audio
up to that frame. No future context, no sliding window, no mixing of
speaker channels. One forward pass per speaker channel produces all frame
predictions simultaneously.

## Training

Trained on TurnBench + Switchboard (train_joint) with hard labels from
`metric.scp`. WavLM backbone is frozen throughout (frozen-backbone
baseline); unfrozen fine-tuning is in progress.

- **Loss:** Cross-entropy on the last-frame prediction per utterance.
- **Optimizer:** AdamW, lr=1e-4, warmup 2000 steps, cosine decay.
- **Epochs:** 200 (Base converged ~ep10, Large ~ep12 with frozen backbone).

## Results (dev, official `eval.score`)

Operating point: EOT `tau_high=0.20`, INT `tau_high=0.15` (same as
`espnet_turntaking_perchannel` baseline), refractory 2.0 s. Commitment via
online hysteresis detector, identical to the per-channel baseline.

| Track | Model | recall | fp_rate | latency p50 (ms) |
| --- | --- | --- | --- | --- |
| **EOT** | WavLM-Base-Plus | 0.183 | 0.117 | 526 |
| **EOT** | **WavLM-Large** | **0.303** | **0.104** | **661** |
| **EOT** | espnet_turntaking_perchannel | 0.666 | 0.210 | 428 |
| **INT** | WavLM-Base-Plus | 0.591 | 0.033 | 996 |
| **INT** | **WavLM-Large** | **0.818** | **0.186** | **578** |
| **INT** | espnet_turntaking_perchannel | 0.695 | 0.217 | 527 |

### Threshold sweep (dev)

**WavLM-Large:**

| Track | tau_high | recall | fp_rate | latency p50 (ms) |
| --- | --- | --- | --- | --- |
| EOT | 0.05 | 0.612 | 0.422 | 224 |
| EOT | 0.10 | 0.604 | 0.380 | 343 |
| EOT | 0.15 | 0.489 | 0.205 | 496 |
| EOT | 0.20 | 0.303 | 0.104 | 661 |
| EOT | 0.25 | 0.170 | 0.045 | 814 |
| INT | 0.10 | 0.890 | 0.385 | 315 |
| INT | 0.15 | 0.818 | 0.186 | 578 |
| INT | 0.20 | 0.614 | 0.070 | 1186 |
| INT | 0.25 | 0.343 | 0.020 | 1427 |

**WavLM-Base-Plus:**

| Track | tau_high | recall | fp_rate | latency p50 (ms) |
| --- | --- | --- | --- | --- |
| EOT | 0.05 | 0.432 | 0.191 | 1508 |
| EOT | 0.10 | 0.366 | 0.216 | 989 |
| EOT | 0.15 | 0.269 | 0.179 | 573 |
| EOT | 0.20 | 0.183 | 0.117 | 526 |
| EOT | 0.25 | 0.106 | 0.060 | 368 |
| INT | 0.05 | 0.914 | 0.364 | 191 |
| INT | 0.10 | 0.879 | 0.121 | 515 |
| INT | 0.15 | 0.591 | 0.033 | 996 |
| INT | 0.20 | 0.294 | 0.011 | 986 |

### Observations

1. **Interruption detection is strong**, even with a frozen backbone.
   WavLM-Large achieves 0.818 recall at 0.186 fp_rate, beating the
   per-channel baseline's 0.695 recall at 0.217 fp_rate.
2. **EOT recall lags behind** the per-channel baseline (0.303 vs 0.666 at
   the default operating point). This is expected: the backbone is frozen,
   and the predictor head has limited capacity. Unfrozen fine-tuning
   (in progress) should close this gap.
3. **WavLM-Large consistently outperforms Base** across all thresholds,
   both for EOT and INT recall.
4. **Fully causal and streaming-compatible:** unlike the per-channel
   baseline (which uses 30 s sliding windows over a non-causal Whisper
   encoder), this model runs a single causal forward pass. Each frame's
   prediction depends only on past audio, making it suitable for real-time
   deployment.

## Files

- `predictions-dev-base.json` — dev predictions (WavLM-Base-Plus, frozen,
  EOT tau=0.20, INT tau=0.15).
- `predictions-dev-large.json` — dev predictions (WavLM-Large, frozen,
  EOT tau=0.20, INT tau=0.15).
- `README.md` — this file.
