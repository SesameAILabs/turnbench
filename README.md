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
eval/                — gold construction + evaluation
  label_map.yaml         fine Mundo labels -> canonical taxonomy (6 classes)
  consensus.py           builds per-sample consensus events from annotators a/b/c
  metrics.py             EOT + Interruption latency / FP-rates / confusion vs gold
baselines/           — model baselines (one folder per baseline, stubs to be implemented)
  gemini/                Gemini (Google) zero-shot prompting
  espnet_turntaking/     ESPnet Turn-Taking Prediction (Switchboard, Arora et al. ICLR 2025)
  mimi_endpointer/       Kyutai Mimi codec + endpointer head
  moshi/                 Kyutai Moshi — full-duplex spoken-language model
  kyutai_semantic_vad/   Kyutai / Unmute STT — semantic end-of-turn classifier
  vap/                   Voice Activity Projection (Ekstedt & Skantze, 2022)
  smart_turn_v3/         Pipecat Smart Turn v3 (Whisper-Tiny + linear head)
```

## Running the analysis

```
python3 data_analysis/analyze.py            # writes stats_out/{summary.json, per_sample.csv, labels.csv}
python3 data_analysis/per_conversation.py   # writes stats_out/{per_conversation.csv, per_type_aggregate.{json,csv}}
python3 data_analysis/plot_per_type.py      # writes stats_out/figures/{heatmap_zscore, metric_bars}.{png,pdf}
```

## Evaluation

**Gold construction (`eval/consensus.py`).** For each sample, the three annotators' fine labels are mapped to the canonical taxonomy (`eval/label_map.yaml`). An event is kept iff all three annotators emit the same canonical label and their start/end times agree within ±200 ms; gold start/end is the median of the three. Regions without consensus are excluded from scoring.

**Canonical taxonomy:** `Turn`, `Interruption`, `Backchannel`, `Overlap`, `Laughter`, `NonContent`. `EOT` is derived from `Turn` event end-times — not a separate label.

**Prediction format.** Each baseline emits one JSONL per `task_id` with point events:
```
{"speaker": 1, "time": 12.345, "label": "EOT"}
{"speaker": 2, "time": 13.802, "label": "Interruption"}
```

**Metrics (`eval/metrics.py`).** EOT and Interruption are scored independently, with a 2 s grace window:
- **TP** — predicted in `(t_gold, t_gold + 2s]`. Latency = `t_pred - t_gold`.
- **FP-premature** — predicted in `(t_gold - 2s, t_gold]` (model cut the speaker off).
- **FP-spurious** — predicted outside any gold event window.
- **FN** — gold event with no matching prediction.

For predicted `Interruption`, a confusion histogram reports what gold event (if any) was actually present — so a model that misclassifies backchannels as interruptions is visible in the `Backchannel` column rather than hidden in spurious-FP totals.

## Baselines

Each baseline lives in its own directory under `baselines/`. The minimum interface is `predict(sample_dir) -> {speaker_id: [(start_s, end_s, label)]}` so evaluation is uniform across models; per-baseline output spaces (continuous probabilities, discrete classes) are documented in each module's docstring along with how EOT and interruption are derived for evaluation.

| Baseline | Modality | Output | Params |
| --- | --- | --- | --- |
| `gemini` | TBD | TBD | TBD |
| `espnet_turntaking` | Audio (Whisper-medium) | 5-class @ 25 Hz (Continuation / Silence / Interruption / Backchannel / Turn-change) | ~307M |
| `mimi_endpointer` | Audio (Mimi codec, 12.5 Hz) | 4-class {user, user-end, system, system-end} | <50M |
| `moshi` | Audio (full-duplex, 12.5 Hz) | Per-frame voice-activity on system stream | ~7B |
| `kyutai_semantic_vad` | Audio + ASR (Kyutai DSM) | Binary EOT (user-only) | >1B |
| `vap` | Audio (two-stream, 50 Hz) | Continuous voice-activity projection per speaker | >100M |
| `smart_turn_v3` | Audio (Whisper-tiny + linear) | Binary per chunk (turn-complete) | ~40M |
