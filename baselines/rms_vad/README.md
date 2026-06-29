# rms_vad

This is a rule-based Voice Activity Detector (VAD) that makes decisions based on the short-term energy of the audio signal. Specifically, it measures the Root Mean Square (RMS) energy in each 20ms window. If the RMS exceeds a fixed threshold (`0.01`), it considers the speaker to be active (speaking). This threshold-based approach does not require any training or model parameters. This is a very sensitive model, serving as a simple reference baseline. It has a high recall, but also produces many false positives.

## Input

Each speaker's mono audio channel, scored independently.

## Output

One boolean per 20ms window whether that window's RMS is above the threshold.

## Predictions

**Speech onset event:** Boolean flip from False to True.
**Speech offset event:** Boolean flip from True to False.

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


| Task | Recall | FP-rate | Latency p10 | Latency p50 | Latency p90 |
| --- | ---: | ---: | ---: | ---: | ---: |
| EOT | 0.595 | 0.547 | -220 | -98 | 2161 |
| INT | 0.994 | 0.390 | 55 | 137 | 602 |
