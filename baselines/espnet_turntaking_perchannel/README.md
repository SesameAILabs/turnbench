# espnet_turntaking_perchannel

The **individual-channel** variant of the
[`espnet_turntaking`](../espnet_turntaking) baseline. Same model — the "Talking
Turns" frozen Whisper-medium encoder (~306 M) + 5-class head (Arora et al.,
ICLR 2025), emitting per-40 ms (25 Hz) probabilities over {Continuation(C),
Silence(NA), Interruption(I), Backchannel(BC), Turn-change(T)} — but a
different **inference strategy**.

## Strategy: two runs, one per channel (no mixing)

The `espnet_turntaking` baseline runs the model **once on the two-speaker mono
mix** (in-distribution with how the model was trained) and attributes events to
speakers with energy VAD. This baseline instead runs the model **twice — once
on each isolated speaker channel** — and reads each channel's native signals
directly as that speaker's events:

```
eot_speaker_K          = commit(P_T on speaker_K_audio)   # K's turn ending
interruption_speaker_K = commit(P_I on speaker_K_audio)   # K interrupting
```

No mix, no energy attribution: the per-channel run *is* the per-speaker signal.
Per-frame scores are committed to event times with an online hysteresis +
refractory detector (`_commit`); each time depends only on audio up to that
frame.

## Results (dev, official `eval.score`) — head-to-head vs. the mix baseline

Same model, same dev set; only the inference strategy differs.

| Track | Strategy | recall | fp_rate | latency p10/p50/p90 (ms) |
| --- | --- | --- | --- | --- |
| **EOT** | mix + attribution (`espnet_turntaking`) | 0.372 | 0.183 | −187 / −14 / 1585 |
| **EOT** | **individual-channel (this)** | **0.666** | 0.210 | −47 / 428 / 1367 |
| **INT** | mix + attribution (`espnet_turntaking`) | **0.755** | 0.159 | 74 / 197 / 1017 |
| **INT** | **individual-channel (this)** | 0.695 | 0.217 | 65 / 527 / 1600 |

Operating point: EOT `tau_high=0.20`, INT `tau_high=0.15`, refractory 2.0 s
(tuned on dev with `--sweep`; override via `PC_*` env vars).

### Observations

1. **End-of-turn: individual-channel is much better** (recall 0.67 vs 0.37). EOT
   is largely a *single-speaker* event ("this speaker stopped"), which an
   isolated channel captures cleanly.
2. **Interruption: the mix wins** (recall 0.76 vs 0.70 at lower fp). Interruption
   is genuinely *relational* — it needs the other speaker present — and the model
   was trained on the mix, so an isolated channel defines it less well.
3. **Costs:** ~2× higher latency (p50 428 ms vs −14 ms; the mix fires
   early/speculatively) and 2× the inference compute (two passes per
   conversation).
4. **Takeaway:** the best choice is *per-track*. A hybrid — individual-channel
   EOT + mix interruption — would likely beat either alone.

A full threshold sweep (`--sweep`) confirms the trend is stable across operating
points (EOT recall 0.62–0.67 at fp 0.15–0.21; INT peaks ~0.70 at fp ~0.22).

## How to run

```bash
# model side (your espnet env)
export ESPNET_TT_EXP=/abs/path/to/exp/asr_train_asr_whisper_turn_taking_raw_en_word
pip install -r baselines/espnet_turntaking_perchannel/requirements.txt   # + espnet2 importable
python -m baselines.espnet_turntaking_perchannel.predict \
    --out baselines/espnet_turntaking_perchannel/predictions-dev.json

# score (official; uv env)
uv run python -m eval.score baselines/espnet_turntaking_perchannel/predictions-dev.json
```

`predict.py` follows the reference baseline shape: `--dataset` (default the
gated HF dev set; or a local parquet dir), `--out` to write a predictions JSON,
`--shard i --num-shards n` to parallelise the (2× per conversation) inference
across GPUs, and `--sweep` to tune thresholds on the official scorer. A per-frame
cache (`--cache-dir`, default `predictions/espnet_turntaking_perchannel/cache`)
stores each channel's probabilities so re-tuning is instant. The model is read
from `ESPNET_TT_EXP` (HF `espnet/Turn_taking_prediction_SWBD`).

## Files

- `predict.py` — self-contained individual-channel predictor.
- `requirements.txt` — model-side deps (scorer deps come from the repo `eval` extra).
- `predictions-dev.json` — committed dev predictions at the operating point above.
- `predictions-test.json` — committed test predictions (same operating point).
- `run_dev_sharded.sbatch` / `run_test_sharded.sbatch` — sharded inference jobs
  used to produce the caches.

## Notes

- Frame rate **25 Hz**; channels resampled to the model's 16 kHz.
- Runtime is ~2× the mix baseline (two passes per conversation); once cached,
  rebuilding the JSON at new thresholds is instant.
- Caveat the numbers quantify: the model was trained on the two-speaker mix and
  its I/BC/T labels are relational, so an isolated channel is out-of-distribution
  for the relational tracks — which is exactly why interruption degrades while
  the single-speaker EOT improves.
