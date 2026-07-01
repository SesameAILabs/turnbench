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

Install ESPnet at the pinned commit:

```bash
git clone https://github.com/espnet/espnet && cd espnet
git checkout 750e3749fc37a09187fe0fc6fb278ccb007181e8   # version 202604
cd tools && make
pip install -e .
```

Additional dependencies:

```bash
pip install numpy soundfile scipy kaldiio
```

### 2. Checkpoint

The trained checkpoint is at:
`exp/universa_turn_taking_only_turn_a40/valid.loss.best.pth`

(ANCHOR TT-only, trained on TurnBench with `conf/train_ar_turn_taking_only.yaml`,
4×A40 DDP, batch size 32, 14 epochs.)

Config: `conf/train_ar_turn_taking_only.yaml`
Token list: `data/token_list/turn_taking_only_tokens/tokens.json` (6 tokens)

### 3. Produce per-frame probabilities

Run sliding-window inference on each speaker channel:

```bash
python pyscripts/run_turn_taking_inference.py \
    --model-dir exp/universa_turn_taking_only_turn_a40 \
    --wavscp    data_turnbench/dev_infer/wav.scp \
    --output    exp/universa_turn_taking_only_turn_a40/decode_dev_chunked/text \
    --device cuda --batch-size 64
```

Then extract P(NA)+P(T) for EOT and 1-P(C) for INT into probs JSON format.

### 4. Get operating point

```bash
uv run python -m eval.sweep baselines/wavlm_large_anchor/probs-eot.json   # → θ_eot = 0.90
uv run python -m eval.sweep baselines/wavlm_large_anchor/probs-int.json   # → θ_int = 0.20
```

### 5. Produce predictions

Apply the sweep's standard commit rule (rising-edge + refractory 2.0 s) at
the operating thresholds.

## Files

- `predictions-dev.json` — committed events at θ_eot=0.90, θ_int=0.20.
- `predictions-test.json` — same operating point, test split.
- `probs-eot.json` — per-frame P(NA)+P(T) on dev (25 Hz grid).
- `probs-int.json` — per-frame 1-P(C) on dev (25 Hz grid).
- `predict.py` — prediction stub.
- `README.md` — this file.
