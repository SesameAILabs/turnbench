# smart_turn_v3

Pipecat Smart Turn v3 — Whisper Tiny encoder + linear classifier for semantic end-of-turn detection.

**Model:** ONNX (~8M params). Accepts up to 8s of 16kHz mono audio, returns sigmoid P(turn complete).

## Setup

```bash
git submodule update --init baselines/smart_turn_v3/smart_turn
pip install onnxruntime-gpu transformers torch torchaudio soundfile huggingface_hub
```

ONNX weights auto-downloaded from `pipecat-ai/smart-turn-v3` on first run and symlinked into `smart_turn/`.

## Run

```bash
python3 baselines/smart_turn_v3/predict.py --split eval/splits/dev.txt --run-name smart_turn_v3_dev
```

## Pipeline

Both speaker channels are processed chunk by chunk simultaneously, with independent VAD+accumulate state per channel.

### Per-channel state machine

```
Inactive → (speech onset) → Active → (1s silence) → Settling → (2s) → Inactive
                                          ↑                               |
                                          └── speech: reset & restart ────┘
```

**Inactive:** pre-buffer holds the last 200ms. Score stays 0 until speech onset.

**Active:** chunks accumulated in a rolling 8s buffer — oldest chunks dropped when full. No classification fires during speech. The 8s cap ensures the classifier always sees the most recent context at turn end, not earlier conversation history.

**Settling (2s):** on 1s trailing silence, Smart Turn fires and then re-runs every 32ms chunk for 2 more seconds. As silence grows, P(complete) tends to rise. Speech resumption during settling resets the score and starts a new segment.

Scores are forward-filled at 12.5Hz between classifications.

### Design notes and limitations

Smart Turn v3 was designed for **real-time single-turn classification** — given one user utterance, does it sound complete? It was not designed for multi-turn long-form dialogue, so this harness involves approximations:

- **Rolling 8s window during accumulation:** the original pipeline resets cleanly after each utterance. In long conversations without natural 8s resets, the segment can grow stale. Capping at 8s (dropping oldest chunks) keeps the classifier context anchored to the tail of the current turn, which gives better P(complete) at turn ends than an unbounded accumulation.

- **Settling window:** the original design fires once on silence and locks the score. We extend to a 2s re-classification window so the score can climb as silence accumulates — this helps in cases where the initial prediction is uncertain (e.g. short trailing silence, noisy context).

- **Inherent tension:** a hard 8s flush (as in the upstream `record_and_predict.py`) creates clean short segments at turn ends but also produces spurious mid-speech spikes every 8s. The rolling buffer avoids spikes at the cost of slightly longer-context classifications. Neither approach is ideal for multi-turn eval — this model was simply not designed for it.

- **No interruption head:** `1 - P(turn complete)` is used as a proxy for barge-in likelihood.

## Parameters

| Component | Params |
|---|---|
| Whisper Tiny encoder + linear head | ~8M |
| Silero VAD | ~1M |
| **Total** | **~9M** |

## Score mapping

| Benchmark array | Model output |
|---|---|
| `eot_score_speaker_1` | P(turn complete) for speaker 1 |
| `eot_score_speaker_2` | P(turn complete) for speaker 2 |
| `interruption_score_speaker_1` | `1 - P(turn complete)` for speaker 1 (proxy) |
| `interruption_score_speaker_2` | `1 - P(turn complete)` for speaker 2 (proxy) |

Note: the model outputs P(turn complete) — label 1 = complete, 0 = incomplete (`endpoint_bool` in training data).
There is no explicit interruption head; `1 - P(turn complete)` is used as a directionally-correct proxy for barge-in.
