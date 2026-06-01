# tt-benchmark

Turn-taking events benchmark — corpus of 154 dyadic conversations (~30 hours, 106 unique speakers, 6 conversation types) annotated independently by three annotators for turn-taking phenomena (turns, backchannels, interruptions, overlaps, laughter, etc.).

## Dataset

- **Source (Sesame, IaC-managed):** `gs://sesame-cmu-tt-benchmark-dev/full_delivery_with_metadata/`
- **External mirror (Drive):** `turn-taking-benchmark/` folder on the project owner's Drive
- **Sample layout** (per `task_id/`):
  - `combined_audio.wav`, `speaker_1_audio.wav`, `speaker_2_audio.wav`
  - `speaker_{1,2}_annotation_{a,b,c}.srt` — three independent annotators per speaker
  - `metadata.json` — `task_id`, `speaker_{1,2}_actor_id`, `conversation_type`, `speaker_{1,2}_actor_gender`

The six conversation types: `Argumentative/Deliberative`, `Casual/Spontaneous`, `Collaborative/Problem-Solving`, `Instructional`, `Narrative/Storytelling`, `Task-Oriented/Transactional`.

## Setup

```
cp .env.example .env    # if not present, create it from the keys below
```

`.env` keys:

```
DATA_ROOT=/abs/path/to/local/dataset
BATCH=full_delivery_with_metadata
STATS_DIR=/abs/path/to/tt-benchmark/stats_out
```

Required Python packages: `numpy`, `soundfile`, `matplotlib`.

## Layout

```
data_analysis/       — corpus & per-conversation statistics + figures
  analyze.py             corpus-level totals, label inventory, per-annotator counts
  per_conversation.py    20 metrics per conversation, aggregated by type
  plot_per_type.py       z-score heatmap + small-multiples figures
baselines/           — model baselines (stubs to be implemented)
  silence_threshold.py
  silero_vad.py
  pyannote_segmentation.py
  vap.py
  turngpt.py
```

## Running the analysis

```
python3 data_analysis/analyze.py            # writes stats_out/{summary.json, per_sample.csv, labels.csv}
python3 data_analysis/per_conversation.py   # writes stats_out/{per_conversation.csv, per_type_aggregate.{json,csv}}
python3 data_analysis/plot_per_type.py      # writes stats_out/figures/{heatmap_zscore, metric_bars}.{png,pdf}
```

## Baselines

Each baseline exposes `predict(sample_dir) -> {speaker_id: [(start_s, end_s, label)]}` matching the gold SRT schema, so evaluation is uniform across models.

| Baseline | Modality | Notes |
| --- | --- | --- |
| `silence_threshold` | Audio (energy) | Per-channel energy + gap detection |
| `silero_vad` | Audio (VAD) | Pretrained Silero VAD |
| `pyannote_segmentation` | Audio (segmentation) | `pyannote/segmentation-3.0` |
| `vap` | Audio (joint VAD) | Voice Activity Projection (Ekstedt & Skantze, 2022) |
| `turngpt` | Text | LM `<ts>` probability over ASR transcript |
