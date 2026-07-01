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

The benchmark wants discrete committed times, not a score trace. Each
per-speaker score channel is landed on the canonical 25 Hz grid and committed
with the **central rule** shared by every baseline (`eval.sweep.commit_events`):
one event at each rising edge above a per-task threshold θ, deduped by a 2 s
refractory, timestamped at the frame's end. The emitted time is the acoustic
time the model has heard up to at that frame, so each commit depends only on
audio up to that time. The per-task θ is the dev operating point picked centrally
by `eval.sweep` (highest recall at `fp_rate ≤ 0.1`) — see below.

## Results (dev, official operating point — highest recall at `fp_rate ≤ 0.1`)

Operating point chosen centrally by `eval.sweep` (rule 2 in
[`../README.md`](../README.md)): **θ_eot ≈ 0.000718**, **θ_int = 0.20**, committed
with the standard single-threshold rising-edge rule (2 s refractory). The tiny
θ_eot is expected, not a bug: the EOT score is `P_T × energy-hold` — a
probability *attenuated* by an energy weight, concentrating its mass in
[0, 0.05] (median 0.003) — so its operating point sits at the ~36th percentile
of its own score distribution. `eval.sweep`'s quantile candidates find it; no
fixed uniform grid can.

| task | split | recall | fp_rate |
| --- | --- | --- | --- |
| EOT | dev | 0.836 | 0.074 |
| EOT | test | 0.826 | 0.078 |
| INT | dev | 0.637 | 0.091 |
| INT | test | 0.573 | 0.080 |

Reference `rms_vad` (energy VAD) on the same gold: EOT 0.595 / **0.547** /
−98 ms, INT 0.994 / **0.390** / 137 ms — high recall bought with an fp_rate
far over the budget. This model reaches higher EOT recall *inside* the budget.
The full threshold→(recall, fp_rate, latency) curve is reproducible by sweeping
`probs-eot.json` with `eval.sweep`.

## How to run

`predict.py` (ESPnet + torch) produces the per-frame probability cache; `submit.py`
derives the submission artifacts from that cache with no model re-run; the scorer
runs in the repo's uv env. All share the HF dataset cache.

**Quick reproduce.** `run.sh` shards `predict.py` over all visible GPUs → cache →
`submit.py` probs → `eval.sweep` operating point (highest recall at fp ≤ 0.1, swept
over score-quantile candidates) → `predictions-{split}.json`, then `eval.check`.
TF32 is on by default
(`TT_TF32=1`, ~2.3× on H100 tensor cores at ~5e-3 prob delta; set `TT_TF32=0` for
bit-exact fp32):

```bash
PYTHON=/path/to/espnet-venv/bin/python \
ESPNET_TT_EXP=/abs/.../asr_train_asr_whisper_turn_taking_raw_en_word \
  bash baselines/espnet_turntaking/run.sh dev    # → probs-*.json + predictions-dev.json
# then the test submission, committed at the same dev operating point:
  bash baselines/espnet_turntaking/run.sh test   # → predictions-test.json
```

Or the equivalent manual steps that `run.sh` wraps:

```bash
# --- 1) per-frame probabilities (model side; your espnet env) ---
export ESPNET_TT_EXP=/abs/path/to/exp/asr_train_asr_whisper_turn_taking_raw_en_word
pip install -r baselines/espnet_turntaking/requirements.txt   # + espnet2 importable
python -m baselines.espnet_turntaking.predict \
    --out baselines/espnet_turntaking/predictions-dev.json     # fills predictions/espnet_turntaking_dev/cache
#   test cache: --dataset <test repo> --cache-dir predictions/espnet_turntaking_test/cache --num-shards N --shard i

# --- 2) dev probs → operating point (rule 2: highest recall at fp_rate ≤ 0.1) ---
python -m baselines.espnet_turntaking.submit probs --task eot --out baselines/espnet_turntaking/probs-eot.json
python -m baselines.espnet_turntaking.submit probs --task int --out baselines/espnet_turntaking/probs-int.json
uv run python -m eval.sweep baselines/espnet_turntaking/probs-eot.json   # → θ_eot (≈0.000718)
uv run python -m eval.sweep baselines/espnet_turntaking/probs-int.json   # → θ_int (0.20)

# --- 3) committed predictions at (θ_eot, θ_int) ---
python -m baselines.espnet_turntaking.submit predictions --split dev  --theta-eot 0.0007176333522367602 --theta-int 0.20 \
    --out baselines/espnet_turntaking/predictions-dev.json
python -m baselines.espnet_turntaking.submit predictions --split test --theta-eot 0.0007176333522367602 --theta-int 0.20 \
    --cache-dir predictions/espnet_turntaking_test/cache --out baselines/espnet_turntaking/predictions-test.json

# --- 4) validate every committed file (only way to check the test file) ---
uv run python -m eval.check baselines/espnet_turntaking
```

`predict.py` follows the reference baseline shape (`--dataset` = gated HF dev set
or a local parquet dir; model from `ESPNET_TT_EXP` = HF
`espnet/Turn_taking_prediction_SWBD`; `espnet2` importable). `submit.py` reads the
per-frame cache only: it maps the softmax to per-speaker EOT (`P_T·hold`) /
interruption (`P_I·onset`) scores, lands them on the canonical grid `floor(dur·25)`
by left-padding the 5-frame (0.2 s) model pre-roll, writes `probs-*.json`, and
commits events with the central `eval.sweep.commit_events` rule.

**Environment (reproducible).** The engine is **stock upstream ESPnet**
(`github.com/espnet/espnet`, `master`) — *not a fork*. Pin: commit `750e3749`
(v202604), `pip install -e .` provides `espnet2` (at this version ESPnet ships
`espnet2`/`espnet3` only — there is no `espnet1` package, so import `espnet2`).
A clean install also needs `torchaudio` matched to your `torch`, and ESPnet's
pinned **whisper fork** (`git+https://github.com/espnet/whisper.git`; the latest
pypi `openai-whisper` is incompatible — it removed `whisper.audio.N_MELS`).
`cd tools && make` handles both automatically; see `requirements.txt` for the
manual list. Verified: a from-scratch install at this pin reproduces the
committed `predictions-dev.json` **bit-for-bit**.

## Files

- `predict.py` — the model side (mix inference + energy attribution → per-frame cache).
- `submit.py` — derives the submission artifacts from the cache (probs + predictions).
- `requirements.txt` — model-side deps (scorer deps come from the repo `eval` extra).
- `probs-eot.json` / `probs-int.json` — per-frame dev probabilities for the central sweep.
- `predictions-dev.json` / `predictions-test.json` — committed events at the swept operating point (θ_eot ≈ 0.000718, θ_int = 0.20).

## Notes

- Frame rate **25 Hz**; channels resampled to the model's 16 kHz.
- Inference is the configuration that reproduced the model's Switchboard ROC-AUC;
  steady-state 30 s windows are encoded in batches (bit-exact with the per-frame
  reference). Runtime ~9 min per ~12 min conversation on an H100/GH200-class GPU;
  cached thereafter.
- Dev numbers here were produced with the official `eval.score` over the public
  dev annotations; the eval server runs the same scorer on the private test set.
