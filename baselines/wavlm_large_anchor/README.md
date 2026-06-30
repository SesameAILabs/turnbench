# wavlm_large_anchor

A general-purpose speech quality and turn-taking model based on the **ANCHOR**
framework, using a frozen **WavLM-Large** frontend (~315 M parameters) with a
4-layer Transformer audio encoder and a 12-layer autoregressive Transformer
decoder. Trained on Switchboard + TurnBench (train_joint) with hard turn-taking
labels. Inference uses 4-second sliding windows at 40 ms stride.

## Architecture

```
WavLM-Large (frozen)
  → 4-layer Transformer audio encoder
  → 12-layer AR Transformer decoder (token generation)
  → 5-class turn-taking distribution {C, NA, I, BC, T}
```

Total: ~628 M parameters (313 M trainable). Within each 4 s window, attention
is bidirectional, but windows are independent and each ends at the current time,
so no future audio is ever observed. Frame rate: 25 Hz (40 ms stride), first
prediction at 200 ms. Declared lookahead: **0 ms**.

## Operating point (rule 2: lowest latency at fp_rate ≤ 0.1)

```
θ_eot = 0.40   (from eval.sweep on probs-eot.json)
θ_int = 0.20   (from eval.sweep on probs-int.json)
```

Commitment: online hysteresis detector (tau_low = 0.4 × tau_high, refractory 2.0 s).

## How to reproduce

### 1. Environment

The model uses ESPnet's Universa/ANCHOR training and inference pipeline.
Install ESPnet at the pinned commit:

```bash
git clone https://github.com/espnet/espnet && cd espnet
git checkout 750e3749fc37a09187fe0fc6fb278ccb007181e8   # version 202604
cd tools && make   # installs matching torchaudio + whisper fork
pip install -e .   # provides espnet2
```

Additional dependencies:

```bash
pip install numpy soundfile scipy kaldiio
```

The inference script lives in the ESPnet turn-taking recipe:
`espnet/egs2/universa_unite/turn_taking1/pyscripts/`

- `run_turn_taking_inference.py` — sliding-window AR inference
- `eval_turnbench.py` — convert frame probs to Submission JSON + score

### 2. Checkpoint

The trained checkpoint is at:
`exp/tt_pred_large_turn_swbd_res/valid.loss.best.pth`

(WavLM-Large frozen predictor, trained for 200 epochs on TurnBench + Switchboard
train_joint, AdamW lr=1e-4, warmup 2000 steps, cosine decay, batch size 64.)

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
uv run python -m eval.sweep baselines/wavlm_large_anchor/probs-eot.json   # → θ_eot = 0.40
uv run python -m eval.sweep baselines/wavlm_large_anchor/probs-int.json   # → θ_int = 0.20
```

### 5. Produce predictions-dev.json and predictions-test.json

Apply the hysteresis commit detector at the operating thresholds above to the
per-frame P(T) (EOT) and P(I) (INT) probabilities, emitting event times for
each speaker channel.

## Files

- `predictions-dev.json` — committed events at θ_eot=0.40, θ_int=0.20.
- `predictions-test.json` — same operating point, test split.
- `probs-eot.json` — per-frame P(T) on dev (25 Hz grid).
- `probs-int.json` — per-frame P(I) on dev (25 Hz grid).
- `predict.py` — prediction stub.
- `README.md` — this file.
