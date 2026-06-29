# rms_vad

Rule-based energy VAD, no training or parameters: a speaker is speaking wherever
their 20 ms RMS exceeds a fixed threshold (`0.01`). The reference baseline and the
floor of the benchmark — it fires on every silence, so low latency at a high
false-positive rate.

## Input

Each speaker's mono audio channel, scored independently.

## Output

One boolean per 20 ms window: is that window's RMS above the threshold?

## Predictions

Wherever the per-window boolean flips, emit one event: silence→speech (the
speaker started talking) is an interruption, speech→silence (they stopped) is an
EOT. The event time is the end of the flipped window, since the flip is only known
once that window has been heard. Run over both speaker channels of every
conversation:

```bash
# dev (score in-place, or write a JSON)
uv run python -m baselines.rms_vad.predict
uv run python -m baselines.rms_vad.predict --out baselines/rms_vad/predictions-dev.json

# test
uv run python -m baselines.rms_vad.predict \
    --dataset mundo-ai/turn-benchmark-test \
    --out baselines/rms_vad/predictions-test.json
```

## Results (dev)

Latency is `t_pred − t_gold` in ms over true positives (negative = fired early).

| Task | Recall | FP-rate | Latency p10 | Latency p50 | Latency p90 |
| --- | ---: | ---: | ---: | ---: | ---: |
| EOT | 0.595 | 0.547 | -220 | -98 | 2161 |
| INT | 0.994 | 0.390 | 55 | 137 | 602 |
