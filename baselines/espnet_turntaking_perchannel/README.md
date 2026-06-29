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

## Results (dev, official `eval.score`, 3 s scoring window) — head-to-head vs. the mix baseline

Same model, same dev set; only the inference strategy differs.

| Track | Strategy | recall | fp_rate | latency p10/p50/p90 (ms) |
| --- | --- | --- | --- | --- |
| **EOT** | mix + attribution (`espnet_turntaking`) | 0.461 | 0.183 | −160 / 52 / 2391 |
| **EOT** | **individual-channel (this)** | **0.733** | 0.208 | −32 / 499 / 1911 |
| **INT** | mix + attribution (`espnet_turntaking`) | 0.772 | **0.159** | 74 / 200 / 1197 |
| **INT** | **individual-channel (this)** | **0.795** | 0.217 | 69 / 756 / 2150 |

Operating point: EOT `tau_high=0.20`, INT `tau_high=0.15`, refractory 2.0 s
(tuned on dev with `--sweep`; override via `PC_*` env vars).

### Observations

1. **End-of-turn: individual-channel is much better** (recall 0.73 vs 0.46). EOT
   is largely a *single-speaker* event ("this speaker stopped"), which an
   isolated channel captures cleanly.
2. **Interruption: the mix is preferable** (fp 0.16 vs 0.22 and far lower
   latency; recall is close, 0.77 vs 0.80). Interruption is genuinely
   *relational* — it needs the other speaker present — and the model was trained
   on the mix, so an isolated channel defines it less cleanly (more false
   positives, later commits).
3. **Costs:** much higher latency (EOT p50 499 ms vs 52 ms; the mix fires
   early/speculatively) and 2× the inference compute (two passes per
   conversation).
4. **Takeaway:** the best choice is *per-track*. A hybrid — individual-channel
   EOT + mix interruption — would likely beat either alone.

A full threshold sweep (`--sweep`) confirms the trend is stable across operating
points (EOT recall 0.61–0.73 at fp 0.10–0.21; INT peaks ~0.63 at fp ~0.12).

## Decision-threshold trade-off (dev)

`plot_pareto_dev.py` sweeps a single decision threshold θ over the cached
probabilities and plots the two opposing objectives — **EOT median latency** and
**false-interruption rate** — on a dual axis (`pareto_sweep_dev.png`). They move
in opposite directions as θ rises: no single threshold optimizes both, so the
committed per-track operating points (EOT 0.20, INT 0.15) are a Pareto choice
along this curve. The EOT-latency line is masked where EOT recall drops below 5%
(its median is then over too few detections to be meaningful).

```bash
python -m baselines.espnet_turntaking_perchannel.plot_pareto_dev \
    --out baselines/espnet_turntaking_perchannel/pareto_sweep_dev.png
```

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
- `plot_pareto_dev.py` / `pareto_sweep_dev.png` — EOT-latency vs false-interruption-rate trade-off figure.
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
