# smart_turn_v3

Pipecat Smart Turn v3 — Whisper Tiny encoder + linear classifier for semantic end-of-turn detection.

**Model:** ONNX (~8M params). Accepts up to 8s of 16kHz mono audio, returns sigmoid probability of turn completion.

**Inference:** Simulates the real-time pipeline from `smart_turn/record_and_predict.py`:
1. Silero VAD (ONNX, stateful) on 512-sample chunks (32ms each)
2. Speech accumulated with 200ms pre-buffer; no hard segment cap
3. Smart Turn fires on 1s trailing silence, then re-runs every chunk for a 2s settling window — P(complete) can rise as silence accumulates
4. If speech resumes during settling, score resets and a new segment begins
5. Scores forward-filled at 12.5Hz until next speech onset

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
There is no explicit interruption head; `1 - P(turn complete)` ("speaker is mid-utterance") is used as a
directionally-correct proxy for barge-in.
