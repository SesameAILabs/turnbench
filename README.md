# tt-benchmark

Turn-taking events benchmark — corpus of 154 dyadic conversations (~30 hours, 106 unique speakers, 6 conversation types) annotated independently by three annotators for turn-taking phenomena (turns, backchannels, interruptions, overlaps, laughter, etc.).

## Dataset

- **Source (gated GCS bucket; access granted via IAM per collaborator):** `gs://sesame-cmu-tt-benchmark-dev/full_delivery_with_metadata/`
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

The discrete scorer is packaged separately (`pyproject.toml`) and resolves the
dataset itself (the public dev set on HuggingFace by default, no `.env`
needed):

```
uv sync --extra eval --extra dev
uv run python -m eval.score predictions.json        # score a submission on dev
uv run pytest                                        # scorer tests
```

## Layout

```
data_analysis/       — corpus & per-conversation statistics + figures
  analyze.py             corpus-level totals, label inventory, per-annotator counts
  per_conversation.py    20 metrics per conversation, aggregated by type
  plot_per_type.py       z-score heatmap + small-multiples figures
eval/                — gold construction + evaluation
  gold.py                builds gold event sets on the fly from annotators a/b/c
  score.py               scores a discrete predictions.json (recall / FP-rate / latency)
  submission.py          submission schema + validators (the discrete format)
  data.py                resolves the dataset (HF dev set, or --dataset override)
  label_map.yaml         fine Mundo labels -> canonical taxonomy (reference)
baselines/           — model baselines (one folder per baseline)
  rms_vad/               energy-VAD reference baseline (the simplest, runnable today)
  sesame/                Sesame internal turn-taking system (predictions generated externally)
  gemini/                Gemini full-duplex streaming dialogue (Google) — ASR/VAD-aligned
  moshi/                 Kyutai Moshi — full-duplex spoken-language model
  espnet_turntaking/     ESPnet Turn-Taking Prediction (Switchboard, Arora et al. ICLR 2025)
  mimi_endpointer/       Kyutai Mimi codec + 4-class endpointer head
  kyutai_semantic_vad/   Kyutai / Unmute STT — semantic end-of-turn classifier
  vap/                   Voice Activity Projection (Ekstedt & Skantze, 2022)
  smart_turn_v3/         Pipecat Smart Turn v3 (Whisper-Tiny + linear head)
  wavlm_base_causal/     WavLM-Base Causal Predictor (~98M, fully causal, 25 Hz)
  wavlm_large_anchor/    WavLM-Large ANCHOR Judge (~628M, 4 s sliding windows)
```

## Running the analysis

```
python3 data_analysis/analyze.py            # writes stats_out/{summary.json, per_sample.csv, labels.csv}
python3 data_analysis/per_conversation.py   # writes stats_out/{per_conversation.csv, per_type_aggregate.{json,csv}}
python3 data_analysis/plot_per_type.py      # writes stats_out/figures/{heatmap_zscore, metric_bars}.{png,pdf}
```

## Evaluation

**Gold construction (`eval/gold.py`).** Gold is built on the fly from the three annotators' SRTs, with no stored artifact. Fine labels map to the canonical taxonomy; an event is kept iff all three annotators agree on the canonical label and their endpoints agree within ±200 ms, with the median as the gold boundary. Spans without 3-way agreement become excluded intervals: predictions inside them are neither rewarded nor penalised. EOT positives are the turn-ends where the floor actually passes to the other speaker; INT positives are floor-taking interruption onsets. See the `eval/gold.py` module docstring for the full floor-construction rule.

**Submission + scoring (discrete).** A submission is a single `predictions.json` of committed **event times** per speaker (`eot`, `interruption`), scored by `eval/score.py`:

```
uv run python -m eval.score predictions.json
```

There are no continuous scores or thresholds in a submission: emit an event at the time your system commits to acting on it. Turning a model's continuous scores into committed event times (thresholding, sweeping) is the submitter's responsibility; the scorer exposes an in-memory `score_submission` entry point so you can sweep without writing a file per operating point. Full spec, the causality rule, and the self-sweep pattern: [`docs/SUBMISSION_FORMAT.md`](docs/SUBMISSION_FORMAT.md). `baselines/oracle_annotator/predict.py` is the runnable reference.

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

Each baseline lives in its own directory under `baselines/` and is a standalone predictor: it runs its model over the dataset and emits a discrete `predictions.json` (see `docs/SUBMISSION_FORMAT.md`), which `eval/score.py` scores. A baseline owns the whole path, including its **own continuous→discrete step** (thresholding its model's scores to committed event times, at whatever operating point it picks); the benchmark only ever sees committed events. `baselines/rms_vad/` is the minimal reference: read audio, derive a signal, threshold to events, emit. There is no shared trace format and no central runner.

> **Migration note.** The model baselines below were written against the old continuous-trace pipeline (per-frame scores persisted as NPZ, swept centrally). That pipeline has been removed. Each baseline's `predict.py` needs its author to update it to emit a `predictions.json` directly. Until then they will not run.

| Baseline | Modality | Output | Params | Status |
| --- | --- | --- | --- | --- |
| `rms_vad` | Energy VAD | Speech on/off edges → discrete events | n/a | implemented (runs on dev today) |
| `oracle_annotator` | Sanity check | Replays the gold events | n/a | implemented |
| `sesame` | Internal CD model | Continuous per-frame heads @ 12.5 Hz | (internal) | predictions provided externally |
| `gemini` | Full-duplex streaming dialogue | ASR/VAD-aligned timestamps over output audio | undisclosed | stub |
| `moshi` | Audio (full-duplex, 12.5 Hz) | Per-frame voice-activity on system stream | ~7B | stub |
| `espnet_turntaking` | Audio (frozen Whisper-medium, CMU) | 5-class @ 25 Hz (Continuation / Silence / Interruption / Backchannel / Turn-change), trained on Switchboard (Arora et al., ICLR 2025) | ~307M | stub |
| `mimi_endpointer` | Audio (Mimi codec, 12.5 Hz) | 4-class per frame {user, user-end, system, system-end} | <50M | stub |
| `kyutai_semantic_vad` | Audio + ASR (Kyutai DSM) | Binary EOT per frame (user-only) | >1B | stub |
| `vap` | Audio (two-stream, 50 Hz) | Continuous voice-activity projection per speaker | >100M | stub |
| `smart_turn_v3` | Audio (Whisper-Tiny + linear head) | Binary per 8 s chunk (turn-complete) | ~40M | stub |
| `wavlm_base_causal` | Audio (frozen WavLM-Base-Plus, causal, CMU) | 5-class @ 25 Hz, fully causal single pass, trained on Switchboard | ~98M (3.8M trainable) | stub |
| `wavlm_large_anchor` | Audio (frozen WavLM-Large, 4 s windows, CMU) | 5-class @ 25 Hz, autoregressive decoder, trained on Switchboard | ~628M (313M trainable) | stub |

`rms_vad` and `oracle_annotator` run out of the box today and are the reference
shape for the discrete contract. The model baselines need migration (see the
note above); their intended interfaces are documented in each module's
docstring, and community contributions implementing them are welcome.
