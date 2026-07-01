# wavlm_large_causal

A lightweight, fully causal turn-taking predictor using a frozen
**WavLM-Large** encoder (~315 M parameters) with causal attention masking.
Trained on Switchboard + TurnBench (train_joint) with hard turn-taking labels.

## Architecture

```
WavLM-Large (frozen, causal-masked)
  → Conv1d subsample (stride 2, 1024d → 256d)
  → 4-layer causal Transformer encoder (256d, 4 heads, FFN 1024)
  → Linear head → 5 classes {C, NA, I, BC, T}
```

Total: ~320 M parameters (4 M trainable, backbone frozen).

The model runs in a **single causal forward pass** per speaker channel — no
sliding windows. Each frame's prediction depends only on audio up to that
frame. Frame rate: 25 Hz (40 ms stride), first prediction at 200 ms.
Declared lookahead: **0 ms**.

## Probability signals

- **EOT:** P(NA) + P(T) — end-of-turn is detected as a transition to silence.
  P(T) alone is too sparse; combining it with P(Silence) captures the signal
  that the speaker is finishing.
- **INT:** P(I) — interruption probability directly.

## Operating point (rule 2: lowest latency at fp_rate ≤ 0.1)

```
θ_eot = 0.85   (from eval.sweep on probs-eot.json)
θ_int = 0.20   (from eval.sweep on probs-int.json)
```

Dev results at operating point:
- EOT: recall=0.451, fp_rate=0.090, latency p50=667ms
- INT: recall=0.625, fp_rate=0.070, latency p50=1032ms

Commitment: online hysteresis detector (tau_low = 0.4 × tau_high, refractory 2.0 s).

## How to reproduce

### 1. Environment

The model is built on ESPnet's `CausalS3prlFrontend` (WavLM with causal
streaming mask). Install ESPnet at the pinned commit:

```bash
git clone https://github.com/espnet/espnet && cd espnet
git checkout 750e3749fc37a09187fe0fc6fb278ccb007181e8   # version 202604
cd tools && make   # installs matching torchaudio + whisper fork
pip install -e .   # provides espnet2
```

Additional dependencies:

```bash
pip install numpy soundfile scipy
```

The predictor model code lives in the ESPnet turn-taking recipe:
`espnet/egs2/universa_unite/turn_taking1/pyscripts/`

- `turn_taking_predictor_model.py` — `CausalTurnTakingPredictor` model
- `train_turn_taking_predictor.py` — training script
- `run_predictor_inference.py` — inference (chunked mode for long audio)

### 2. Checkpoint

The trained checkpoint is at:
`exp/tt_pred_large_turn_swbd_res/valid.loss.best.pth`

(Trained for 200 epochs on TurnBench + Switchboard train_joint, WavLM-Large
frozen, AdamW lr=1e-4, warmup 2000 steps, cosine decay, batch size 64.)

### 3. Produce per-frame probabilities (probs-eot.json / probs-int.json)

Run chunked inference on each speaker channel of the dev set:

```bash
python pyscripts/run_predictor_inference.py \
    --model-dir exp/tt_pred_large_turn_swbd_res \
    --wavscp    data_turnbench/dev_infer/wav.scp \
    --output    exp/tt_pred_large_turn_swbd_res/decode_dev_chunked/text \
    --upstream  wavlm_large \
    --device cuda --batch-size 1 --chunk-seconds 30
```

Then extract per-class probabilities into the probs JSON format (P(T) for EOT,
P(I) for INT) using the frame grid from `eval/durations-dev.json`.

### 4. Get operating point

```bash
uv run python -m eval.sweep baselines/wavlm_large_causal/probs-eot.json   # → θ_eot = 0.85
uv run python -m eval.sweep baselines/wavlm_large_causal/probs-int.json   # → θ_int = 0.20
```

### 5. Produce predictions-dev.json and predictions-test.json

Apply the hysteresis commit detector at the operating thresholds above to the
per-frame P(T) (EOT) and P(I) (INT) probabilities, emitting event times for
each speaker channel.

## Files

- `predictions-dev.json` — committed events at θ_eot=0.85, θ_int=0.20.
- `predictions-test.json` — same operating point, test split.
- `probs-eot.json` — per-frame P(NA)+P(T) on dev (25 Hz grid).
- `probs-int.json` — per-frame P(I) on dev (25 Hz grid).
- `README.md` — this file.
