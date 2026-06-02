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
cp .env.example .env
# then edit .env and set the one required variable
```

The only required `.env` variable is **`TT_BENCHMARK_DATA`** — the
absolute path to your local copy of the dataset root (the directory
that contains one subdirectory per `task_id`):

```
TT_BENCHMARK_DATA=/abs/path/to/full_delivery_with_metadata
```

All analysis, consensus, split, and baseline scripts read from this
single variable. Please refer to `TT_BENCHMARK_DATA` by name in any
new script rather than hard-coding paths.

Optional:
```
STATS_DIR=/abs/path/to/tt-benchmark/stats_out   # defaults to ./stats_out
```

Required Python packages: `numpy`, `soundfile`, `matplotlib`, `pyyaml`.

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

**Gold construction (`eval/consensus.py`).** For each sample, the three annotators' fine labels are mapped to the canonical taxonomy (`eval/label_map.yaml`). An event is kept iff all three annotators emit the same canonical label and their start/end times agree within ±200 ms; gold start/end is the median of the three. Any annotator event that does NOT reach 3-way agreement contributes its time span to the `excluded_intervals` list — predictions falling inside these intervals are dropped before scoring (they neither help nor hurt the model).

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

## Splits

The corpus is partitioned into a development set and a held-out test set. **Always use the same `task_id`s the lists below define — do not re-split locally** or numbers stop being comparable across teams.

| File | Description | dev / test | Speaker overlap |
| --- | --- | ---: | ---: |
| `eval/splits/dev.txt` / `test.txt` | **Headline test set.** Speaker-disjoint: every voice actor appears in exactly one partition. Type-balanced within ±1 of 25%. | 38 / 116 | **0** |
| `eval/splits/random_dev.txt` / `random_test.txt` | Random partition baseline. Same target ratio, but speakers leak across dev and test (intentional, for ablation). | 38 / 116 | 58 |

The 25% dev / 75% test target is roughly preserved per conversation type:

| Conversation type | dev | test | total | dev % |
| --- | ---: | ---: | ---: | ---: |
| Argumentative/Deliberative    |  8 |  24 |  32 | 25.0% |
| Casual/Spontaneous            |  7 |  22 |  29 | 24.1% |
| Collaborative/Problem-Solving |  6 |  20 |  26 | 23.1% |
| Instructional                 |  6 |  19 |  25 | 24.0% |
| Narrative/Storytelling        |  5 |  15 |  20 | 25.0% |
| Task-Oriented/Transactional   |  6 |  16 |  22 | 27.3% |
| **All**                       | **38** | **116** | **154** | **24.7%** |

`eval/splits/random_summary.json` records both splits (sizes, per-type counts, speaker overlap). `eval/make_splits.py` regenerates everything from `TT_BENCHMARK_DATA` with a fixed seed.

## Baselines

Each baseline lives in its own directory under `baselines/`. The minimum interface is `predict(sample_dir) -> {speaker_id: [(start_s, end_s, label)]}` so evaluation is uniform across models; per-baseline output spaces (continuous probabilities, discrete classes) are documented in each module's docstring along with how EOT and interruption are derived for evaluation.

**Bidirectional evaluation.** Every conversation is two-channel. Each baseline must run **twice per dialogue** — once treating speaker 1 as the agent (with speaker 2 as the human user) and once treating speaker 2 as the agent. The predicted events for both directions are written into the same JSONL with the appropriate `speaker` field; the eval module scores each direction independently and aggregates. Concretely, a baseline's `predict(sample_dir)` should return predictions for *both* speakers — see the baseline stubs for the boilerplate that reads `TT_BENCHMARK_DATA` and iterates over both directions.

| Baseline | Modality | Output | Params |
| --- | --- | --- | --- |
| `gemini` | TBD | TBD | TBD |
| `espnet_turntaking` | Audio (Whisper-medium) | 5-class @ 25 Hz (Continuation / Silence / Interruption / Backchannel / Turn-change) | ~307M |
| `mimi_endpointer` | Audio (Mimi codec, 12.5 Hz) | 4-class {user, user-end, system, system-end} | <50M |
| `moshi` | Audio (full-duplex, 12.5 Hz) | Per-frame voice-activity on system stream | ~7B |
| `kyutai_semantic_vad` | Audio + ASR (Kyutai DSM) | Binary EOT (user-only) | >1B |
| `vap` | Audio (two-stream, 50 Hz) | Continuous voice-activity projection per speaker | >100M |
| `smart_turn_v3` | Audio (Whisper-tiny + linear) | Binary per chunk (turn-complete) | ~40M |
