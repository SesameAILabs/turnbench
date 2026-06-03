# kyutai_semantic_vad

Kyutai STT-1B semantic VAD for end-of-turn detection.

**Model:** Kyutai STT-1B (`kyutai/stt-1b-en_fr`) — streaming ASR with a VAD head (`vad_heads[2]`) that produces continuous EOT probabilities at 12.5Hz. Each speaker channel is processed independently as the "user" stream.

**Inference:** Streaming — Mimi encodes audio chunk-by-chunk (1920 samples, 80ms), LM steps autoregressively with KV cache via `lm_gen.step_with_extra_heads`.

## Setup

```bash
pip install moshi==0.2.11 julius sphn
```

Model is loaded from `kyutai/stt-1b-en_fr` (cached in `$HF_HOME`). Set `TT_BENCHMARK_DATA` in the repo's `.env`.

## Run

```bash
bash baselines/kyutai_semantic_vad/run.sh

# or directly:
python3 baselines/kyutai_semantic_vad/predict.py --split eval/splits/dev.txt --run-name kyutai_semantic_vad_dev
```

## Parameters

| Component | Params |
|---|---|
| Mimi encoder (params) | 39.39M |
| Mimi encoder (codebook buffers) | 16.84M |
| LM transformer | 989.20M |
| VAD extra heads | ~0M (linear) |
| **Total (used at inference)** | **~1.05B** |

Mimi decoder (~40M) is not used. The full LM runs to maintain KV cache state — text output is discarded.

## Score mapping

| Benchmark array | Model output |
|---|---|
| `eot_score_speaker_1` | VAD prob for speaker 1 channel |
| `eot_score_speaker_2` | VAD prob for speaker 2 channel |
| `interruption_score_speaker_1` | VAD prob for speaker 1 (proxy) |
| `interruption_score_speaker_2` | VAD prob for speaker 2 (proxy) |

Note: this model has no explicit interruption head — VAD probability is used as a proxy for both EOT and interruption scores.
This is based on https://github.com/kyutai-labs/delayed-streams-modeling/blob/main/scripts/stt_from_file_pytorch.py

