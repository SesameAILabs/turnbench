# vap

Voice Activity Projection — GPT-like transformer over CPC features for turn-taking prediction.

**Model:** `VapGPT` (~100M params). Takes stereo audio, outputs per-speaker floor-state probabilities at 50Hz. Trained on Switchboard/Fisher.

**Inference:** `step_extraction()` from `run.py` — overlapping windows (context=20s, step=5s) for long audio, avoiding recomputation of context frames.

## Setup

```bash
git submodule update --init baselines/vap/VoiceActivityProjection
cd baselines/vap/VoiceActivityProjection && pip install -e . && cd ../../..
```

Checkpoint is bundled in the submodule at `VoiceActivityProjection/example/VAP_3mmz3t0u_50Hz_ad20s_134-epoch9-val_2.56.pt`.

## Run

```bash
python3 baselines/vap/predict.py --split eval/splits/dev.txt --run-name vap_dev
```

## Parameters

| Component | Params |
|---|---|
| CPC encoder | ~1M |
| GPT transformer (AliBI) | ~100M |
| **Total** | **~100M** |

## Score mapping

| Benchmark array | Model output |
|---|---|
| `eot_score_speaker_1` | `1 - p_future[:, 0]` (low future floor = turn ending) |
| `eot_score_speaker_2` | `1 - p_future[:, 1]` |
| `interruption_score_speaker_1` | `p_now[:, 0]` (floor activity = barge-in proxy) |
| `interruption_score_speaker_2` | `p_now[:, 1]` |
