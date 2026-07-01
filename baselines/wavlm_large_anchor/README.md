# wavlm_large_anchor

A turn-taking model based on the **ANCHOR** framework (AR Universa), using a
frozen **WavLM-Large** frontend (~315 M parameters) with a 4-layer Transformer
audio encoder and a 12-layer autoregressive Transformer decoder. This is the
**TT-only** variant — the decoder vocabulary contains only 6 turn-taking tokens
(5 classes + meta), no quality metrics. Trained on TurnBench with hard
turn-taking labels. Inference uses 4-second sliding windows at 40 ms stride.

## Architecture

```
WavLM-Large (frozen)
  → 4-layer Transformer audio encoder
  → 12-layer AR Transformer decoder (6-token TT-only vocabulary)
  → 5-class turn-taking distribution {C, NA, I, BC, T}
```

Total: ~628 M parameters (313 M trainable). Within each 4 s window, attention
is bidirectional, but windows are independent and each ends at the current time,
so no future audio is ever observed. Frame rate: 25 Hz (40 ms stride), first
prediction at 200 ms. Declared lookahead: **0 ms**.

## Probability signals

- **EOT:** P(NA) + P(T) — end-of-turn detected as a transition to silence.
- **INT:** 1 - P(C) — interruption detected as a drop in continuation
  probability. P(I) alone is too sparse for this model; 1-P(C) captures the
  same event more reliably.

## Operating point (rule 2: lowest latency at fp_rate ≤ 0.1)

```
θ_eot = 0.90   (from eval.sweep on probs-eot.json)
θ_int = 0.20   (from eval.sweep on probs-int.json)
```

Dev results at operating point:
- EOT: recall=0.813, fp_rate=0.081, latency p50=1141ms
- INT: recall=0.833, fp_rate=0.096, latency p50=992ms

## How to reproduce

### 1. Environment

```bash
pip install espnet s3prl soundfile numpy huggingface_hub
```

`CausalS3prlFrontend` is **not** in stock ESPnet — install it from the HF repo:

```bash
wget -P $(python -c "import espnet2; print(espnet2.__path__[0])")/asr/frontend/ \
    https://huggingface.co/ZhuoyanTao/causal-wavlm-turn-taking/resolve/main/espnet2/asr/frontend/causal_s3prl.py
```

### 2. Checkpoint

Available on HuggingFace: [`ZhuoyanTao/causal-wavlm-turn-taking`](https://huggingface.co/ZhuoyanTao/causal-wavlm-turn-taking)
(`universa_turn_taking_only_turn_a40/{config.yaml, valid.loss.best.pth}`).

Loaded via ESPnet's `UniversaTask.build_model_from_file`. Config.yaml, tokenizer
data, and WavLM-Large are downloaded automatically on first run.

### 3. Run predict.py

```bash
python -m baselines.wavlm_large_anchor.predict                   # score on dev
python -m baselines.wavlm_large_anchor.predict --out preds.json  # write predictions
```

### 4. Get operating point

```bash
uv run python -m eval.sweep baselines/wavlm_large_anchor/probs-eot.json   # → θ_eot = 0.90
uv run python -m eval.sweep baselines/wavlm_large_anchor/probs-int.json   # → θ_int = 0.20
```

## Files

- `predictions-dev.json` — committed events at θ_eot=0.90, θ_int=0.20.
- `predictions-test.json` — same operating point, test split.
- `probs-eot.json` — per-frame P(NA)+P(T) on dev (25 Hz grid).
- `probs-int.json` — per-frame 1-P(C) on dev (25 Hz grid).
- `predict.py` — sliding-window inference (downloads checkpoint from HF).
- `README.md` — this file.
