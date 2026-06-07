# dualturn

DualTurn — Qwen2.5-0.5B + Mimi codec for multi-signal turn-taking prediction.

**Model:** `anyreach-ai/dualturn-qwen2.5-mimi-0.5B` (~0.5B params). Dual-channel 24kHz audio is encoded by the frozen Mimi codec and fed to a Qwen2.5-0.5B backbone with LoRA adapters and 12 per-channel classification heads.

## Setup

```bash
pip install transformers torch torchaudio soundfile
```

Model auto-downloaded from HuggingFace on first run. Pre-download on login node:

```bash
python3 -c "from transformers import AutoModel; AutoModel.from_pretrained('anyreach-ai/dualturn-qwen2.5-mimi-0.5B', trust_remote_code=True)"
```

## Run

```bash
python3 baselines/dualturn/predict.py --split eval/splits/dev.txt --run-name dualturn_dev
```

## Output signals

The model produces 6 per-channel signals. We use:

| Signal | Definition |
|---|---|
| `eot_probs` | High during speech, low at turn end — inverted for EOT score |
| `hold_probs` | Speaker actively holding the floor — proxy for barge-in |

## Score mapping

| Benchmark array | Model output |
|---|---|
| `eot_score_speaker_1` | `1 - eot_probs[:, 0]` |
| `eot_score_speaker_2` | `1 - eot_probs[:, 1]` |
| `interruption_score_speaker_1` | `hold_probs[:, 0]` |
| `interruption_score_speaker_2` | `hold_probs[:, 1]` |

## Parameters

| Component | Params |
|---|---|
| Qwen2.5-0.5B backbone | ~500M |
| LoRA adapters | ~9M trainable |
| Mimi encoder (frozen) | ~40M |
| Classification heads (×12) | ~few M |

## Frame rate

12.5 Hz (80ms frames), matching the Mimi codec output rate.
