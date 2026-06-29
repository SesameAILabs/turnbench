# espnet_turntaking

ESPnet Turn-Taking Prediction (Switchboard) — the judge model from
*"Talking Turns"* (Arora et al., ICLR 2025): a frozen Whisper-medium encoder
(~306 M) with a small 5-class head, emitting per-40 ms (25 Hz) probabilities
over five turn-taking states:

| code | class | meaning |
| --- | --- | --- |
| **C** | Continuation | the current speaker keeps holding the floor |
| **NA** | Silence | no speech on the stream |
| **I** | Interruption | a barge-in (one speaker starts over the other) |
| **BC** | Backchannel | a listener cue (e.g. "uh-huh") that does *not* take the floor |
| **T** | Turn-change | the floor changes hands — i.e. a turn ending |

`P_C`, `P_NA`, `P_I`, `P_BC`, `P_T` denote the softmax probability of each
class at a frame.

## Single-stream, mix-trained → per-speaker events

The model is **single-input** and was trained on Switchboard's two-speaker mono
**mix**; it has no per-speaker output. The dataset ships only the two isolated
channels, so we reconstruct the mix as `speaker_1 + speaker_2` (verified
bit-equivalent to the original combined recording), run the model once on it,
and attribute its native signals to a speaker with cheap energy VAD on the two
channels:

```
eot for speaker K          <- P_T (turn-change), attributed to the floor-holder K
interruption for speaker K  <- P_I (native Interruption), attributed to barge-in K
```
where floor-holder / barge-in come from each speaker's recent (~0.5 s) energy
share on its channel. EOT uses **`P_T` only** (NA omitted).

## From per-frame scores to committed event times

The benchmark wants discrete committed times, not a score trace. We turn each
per-frame channel into event times with an **online hysteresis detector**
(`_commit`): a time is emitted at the frame where the score first rises to
`tau_high`; no further event fires until it falls to `tau_low`, and consecutive
commits are ≥ `refractory_s` apart (turn-ends / interruptions have a real
minimum spacing). The emitted time is the acoustic time the model has heard up
to at that frame, so each commit depends only on audio up to that time.

Operating point (tuned on dev against the official scorer; override via env):

| track | `tau_high` | `tau_low` | `refractory_s` | env |
| --- | --- | --- | --- | --- |
| eot | 0.12 | 0.048 | 2.0 | `TT_EOT_TAU_HIGH` / `TT_EOT_TAU_LOW` / `TT_EOT_REFRACTORY_S` |
| interruption | 0.14 | 0.056 | 2.0 | `TT_INT_*` |

## Results (dev, official `eval.score` — 3 s scoring window, 2-of-3 gold)

| task | recall | fp_rate | latency p10/p50/p90 |
| --- | --- | --- | --- |
| EOT | 0.461 | **0.183** | −160 / 52 / 2391 ms |
| INT | 0.772 | **0.159** | 74 / 200 / 1197 ms |

Reference `rms_vad` (energy VAD) on the same gold: EOT 0.595 / **0.547** /
−98 ms, INT 0.994 / **0.390** / 137 ms. The energy baseline fires on every
silence — high recall, high false-positive rate. This model occupies the
opposite corner: **~3× lower fp_rate** at near-zero EOT latency, trading recall
for precision (the regime that wins when ranking gates on a false-positive
budget). The full threshold→(recall, fp_rate, latency) curve is reproducible
with `sweep.py`; lowering `tau_high` buys recall at higher fp_rate. EOT recall
is capped by the `P_T`-only mapping (it can't fire on turn-ends not followed by
an immediate handoff); folding in `P_NA` raises EOT recall at higher fp_rate.

## How to run

The model runs in your own environment (ESPnet + torch); the scorer runs in the
repo's uv env. They share the HF dataset cache.

```bash
# --- produce predictions (model side; your espnet env) ---
export ESPNET_TT_EXP=/abs/path/to/exp/asr_train_asr_whisper_turn_taking_raw_en_word
pip install -r baselines/espnet_turntaking/requirements.txt   # + espnet2 importable
python -m baselines.espnet_turntaking.predict \
    --out baselines/espnet_turntaking/predictions-dev.json

# --- score (official; uv env) ---
uv sync --extra eval --extra dev
uv run python -m eval.score baselines/espnet_turntaking/predictions-dev.json
```

`predict.py` follows the reference baseline shape (`baselines/rms_vad/predict.py`):
`--dataset` (default the gated HF dev set `mundo-ai/turn-benchmark-dev`; or a
local parquet dir), `--out` to write a predictions JSON, else score in-process.
The model is read from `ESPNET_TT_EXP` (HF `espnet/Turn_taking_prediction_SWBD`);
`espnet2` must be importable.

A per-frame score cache (`--cache-dir`, default
`predictions/espnet_turntaking_dev/cache`) lets repeated runs and the threshold
sweep skip the model — rebuilding the JSON at new thresholds is instant.

## Tuning

```bash
# in-memory sweep over cached scores, scored by the official eval.score
python -m baselines.espnet_turntaking.sweep                 # both tracks
python -m baselines.espnet_turntaking.sweep --track eot --grid 0.04:0.20:0.02
```

## Files

- `predict.py` — the baseline (mix inference + energy attribution + commitment).
- `sweep.py` — in-memory threshold sweep on the official scorer.
- `requirements.txt` — model-side deps (scorer deps come from the repo `eval` extra).
- `predictions-dev.json` — committed dev predictions at the operating point above.
  `predictions-test.json` is added when the private test set is released
  (`--dataset <test repo> --out …/predictions-test.json`).

## Notes

- Frame rate **25 Hz**; channels resampled to the model's 16 kHz.
- Inference is the configuration that reproduced the model's Switchboard ROC-AUC;
  steady-state 30 s windows are encoded in batches (bit-exact with the per-frame
  reference). Runtime ~9 min per ~12 min conversation on an H100/GH200-class GPU;
  cached thereafter.
- Dev numbers here were produced with the official `eval.score` over the public
  dev annotations; the eval server runs the same scorer on the private test set.
