# smart_turn_v3

Pipecat Smart Turn v3 — Whisper Tiny encoder + linear classifier for semantic end-of-turn detection. Runs a VAD+accumulate+settling pipeline at 12.5Hz per speaker channel.

**Model:** ONNX (~8M params). Accepts up to 8s of 16kHz mono audio, returns P(turn complete).

**Score direction:** P(turn complete); floor held when score < threshold.

## Setup

```bash
git submodule update --init baselines/smart_turn_v3/smart_turn
pip install -r baselines/smart_turn_v3/requirements.txt
```

## Run

```bash
bash baselines/smart_turn_v3/run.sh          # dev + test + eval.check
bash baselines/smart_turn_v3/run.sh --dev    # dev probs + predictions only
bash baselines/smart_turn_v3/run.sh --test   # sweep existing probs → test predictions
```

## Results (dev, pretrained)

| Task | θ | Recall | FP-rate | Latency p10 | Latency p50 | Latency p90 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| EOT | 0.05 | 0.637 | 0.082 | 701 ms | 1019 ms | 1392 ms |
| INT | 0.95† | 0.579 | 0.356 | 77 ms | 152 ms | 698 ms |

†INT threshold from CHECKPOINT_DEFAULTS (no sweep op within fp_budget=0.10).
