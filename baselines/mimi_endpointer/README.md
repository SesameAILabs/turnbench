# mimi_endpointer

Two-stream LSTM over Kyutai Mimi embeddings, trained on SpokenWOZ.

**Model:** Unidirectional 2-layer LSTM (512 hidden, 128 projection) on top of Mimi (24kHz, 8 quantizers, 25Hz after upsample). 5-class output: `[bos, system_end, user_end, system, user]`.

**Inference:** Streaming — Mimi processes audio in 1920-sample chunks (80ms) with KV cache, LSTM steps autoregressively per frame.

## Setup

```bash
pip install -r requirements.txt
```

Set `TT_BENCHMARK_DATA` in the repo's `.env`. Place (or symlink) your checkpoint at:
```
baselines/mimi_endpointer/checkpoint.pt
```

## Run

```bash
# dev set
python3 baselines/mimi_endpointer/predict.py --split eval/splits/dev.txt --run-name mimi_endpointer_dev

# test set
python3 baselines/mimi_endpointer/predict.py --split eval/splits/test.txt --run-name mimi_endpointer_test
```

Writes `predictions/mimi_endpointer_dev/traces/<task_id>.npz` — four float32 score arrays at 25Hz per conversation.

## Parameters

| Component | Params |
|---|---|
| Mimi encoder (SEANet CNN) | 12.63M |
| Mimi encoder transformer | 25.19M |
| Quantizer projections | 0.52M |
| Codebook embeddings (8 × 2048 × 256) | 4.19M |
| Upsample | ~0M |
| **Mimi encoder total** | **42.53M** |
| LSTM projection (512→128) | 0.07M |
| LSTM ×2 streams (2-layer, 512 hidden) | 6.83M |
| Linear head | 0.005M |
| **LSTM endpointer total** | **6.90M** |
| **Full inference stack** | **~49.4M** |

Mimi decoder (~40M) is not used at inference.

## Score mapping

| Benchmark array | Model output |
|---|---|
| `eot_score_speaker_1` | P(user_end) |
| `eot_score_speaker_2` | P(system_end) |
| `interruption_score_speaker_1` | P(user) |
| `interruption_score_speaker_2` | P(system) |
