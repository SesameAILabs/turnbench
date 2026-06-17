# TurnBench

A turn-taking benchmark: detecting **end-of-turn (EOT)** and **interruption
(INT)** events in two-channel recorded conversations, scored on recall,
false-positive rate, and detection latency. The corpus is 154 dyadic
conversations (~30 h, 106 unique speakers, 6 conversation types), each
annotated independently by three annotators.

This repo is the benchmark itself — the dataset wiring, the scorer, the
submission spec, and the baselines. The leaderboard website lives in
[cmu-sesame/turnbench](https://github.com/cmu-sesame/turnbench).

## Submit / evaluate

A submission is a single `predictions.json`: for each conversation and speaker,
the times your model commits to an EOT and to an interruption. Run your model
however you like — only the JSON crosses the boundary.

```bash
uv sync --extra eval --extra dev
uv run python -m eval.score predictions.json     # score on the public dev set
```

The dev set downloads from HuggingFace automatically (override with
`--dataset`); there's no data path to manage. A submission carries no
confidence scores or thresholds — emit an event at the time your system commits
to acting on it. Turning a model's continuous scores into committed event times
(thresholding, sweeping) is yours to design; `eval.score.score_submission` lets
you sweep in memory without writing a file per operating point.

Full format, the causality rule, and the sweep pattern:
[`docs/SUBMISSION_FORMAT.md`](docs/SUBMISSION_FORMAT.md). Worked reference:
[`baselines/rms_vad/predict.py`](baselines/rms_vad/predict.py) — run a model,
threshold to events, emit the JSON.

## Gold (`eval/gold.py`)

Gold is built on the fly from the three annotator tracks (no stored artifact),
by **2/3 majority**: an event is kept iff a majority agree on the canonical
label with endpoints within ±200 ms, and the gold boundary is their median. A
dissenting annotator the majority outvotes is settled by the majority, not
treated as uncertain — so only spans with **no majority at all** become
excluded intervals (predictions inside them are neither rewarded nor
penalised). EOT positives are the turn-ends where the floor passes to the other
speaker; INT positives are floor-taking interruption onsets. Full methodology,
the floor-construction rule, and caveats: [`eval/README.md`](eval/README.md).

## Dataset

The scorer pulls the **public dev set** from HuggingFace
(`mundo-ai/turn-benchmark-dev`, revision-pinned, cache-first), so scoring needs
no local data. It is one parquet row per conversation: the two time-aligned
`speaker_{1,2}_audio` channels and the three independent annotator tracks per
speaker (`speaker_{1,2}_annotation_{a,b,c}`, each a `list` of
`{start_s, end_s, label, text}` events parsed losslessly from the source SRTs).
The held-out test set (`mundo-ai/turn-benchmark-test`) is the same schema with
the annotation columns blanked, kept private and scored server-side.

The raw corpus delivery (source of truth) is the gated GCS bucket
`gs://sesame-cmu-tt-benchmark-dev/full_delivery_with_metadata/` — per-`task_id`
dirs with `combined_audio.wav` + the two channels, the six SRTs, and a
`metadata.json` (`task_id`, actor ids, `conversation_type`, genders). The six
types: `Argumentative/Deliberative`, `Casual/Spontaneous`,
`Collaborative/Problem-Solving`, `Instructional`, `Narrative/Storytelling`,
`Task-Oriented/Transactional`.

## Splits

The corpus is partitioned into a development set and a held-out test set.
**Always use the same `task_id`s the lists below define — do not re-split
locally** or numbers stop being comparable across teams.

| File | Description | dev / test | Speaker overlap |
| --- | --- | ---: | ---: |
| `eval/splits/dev.txt` / `test.txt` | **Headline set.** Speaker-disjoint: every voice actor appears in exactly one partition. Type-balanced within ±1 of 25%. | 38 / 116 | **0** |
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

`eval/make_splits.py` regenerates the splits from the raw dataset with a fixed
seed; `eval/splits/random_summary.json` records both.

## Baselines

Each baseline lives in its own directory under `baselines/` and is a standalone
predictor: it runs its model over the dataset and emits a discrete
`predictions.json` (see [`docs/SUBMISSION_FORMAT.md`](docs/SUBMISSION_FORMAT.md)),
which `eval/score.py` scores. A baseline owns the whole path, including its
**own continuous→discrete step** (thresholding its model's scores to committed
event times, at whatever operating point it picks); the benchmark only ever
sees committed events. `baselines/rms_vad/` is the minimal reference: read
audio, derive a signal, threshold to events, emit. There is no shared trace
format and no central runner.

> **Migration note.** The model baselines below were written against an earlier
> continuous-trace pipeline (per-frame scores persisted as NPZ, swept
> centrally). That pipeline has been removed. Each baseline's `predict.py`
> needs its author to update it to emit a `predictions.json` directly (copy
> `rms_vad`'s shape). Until then they will not run.

| Baseline | Modality | Native output | Params | Status |
| --- | --- | --- | --- | --- |
| `rms_vad` | Energy VAD | Speech on/off edges → discrete events | n/a | implemented (runs on dev today) |
| `oracle_annotator` | Sanity check | Replays the gold events | n/a | implemented |
| `sesame` | Internal CD model | Continuous per-frame heads @ 12.5 Hz | (internal) | predictions provided externally |
| `gemini` | Full-duplex streaming dialogue | ASR/VAD-aligned timestamps over output audio | undisclosed | needs migration |
| `moshi` | Audio (full-duplex, 12.5 Hz) | Per-frame voice-activity on system stream | ~7B | needs migration |
| `espnet_turntaking` | Audio (frozen Whisper-medium, CMU) | 5-class @ 25 Hz, trained on Switchboard (Arora et al., ICLR 2025) | ~307M | needs migration |
| `mimi_endpointer` | Audio (Mimi codec, 12.5 Hz) | 4-class per frame {user, user-end, system, system-end} | <50M | needs migration |
| `kyutai_semantic_vad` | Audio + ASR (Kyutai DSM) | Binary EOT per frame (user-only) | >1B | needs migration |
| `vap` | Audio (two-stream, 50 Hz) | Continuous voice-activity projection per speaker | >100M | needs migration |
| `smart_turn_v3` | Audio (Whisper-Tiny + linear head) | Binary per 8 s chunk (turn-complete) | ~40M | needs migration |
| `wavlm_base_causal` | Audio (frozen WavLM-Base-Plus, causal, CMU) | 5-class @ 25 Hz, fully causal, trained on Switchboard | ~98M (3.8M trainable) | needs migration |
| `wavlm_large_anchor` | Audio (frozen WavLM-Large, 4 s windows, CMU) | 5-class @ 25 Hz, autoregressive decoder, trained on Switchboard | ~628M (313M trainable) | needs migration |

`rms_vad` and `oracle_annotator` run out of the box today and are the reference
shape for the discrete contract. The other baselines' native outputs are noted
above; their authors migrate each `predict.py` to emit a `predictions.json`.

## Repo layout

```
eval/                — the benchmark
  gold.py              builds the gold event sets from the SRTs (2/3 majority)
  score.py             scores a predictions.json (recall / FP-rate / latency)
  submission.py        the predictions.json schema + validators
  data.py              resolves the dataset (HF dev set by default, or --dataset)
  parity.py            emits the website's gold + parity bundle (see below)
  make_splits.py       regenerates the speaker-disjoint dev/test splits
  build_eot_validation.py   stratified clips for human EOT validation
baselines/           — one directory per baseline (see above)
data_analysis/       — corpus statistics & figures
```

### Corpus tooling (needs the raw dataset)

`data_analysis/` and `eval/make_splits.py` read a local copy of the raw dataset
through a `.env` file (they also want `numpy`, `soundfile`, `matplotlib`,
`pyyaml`):

```bash
cp .env.example .env
# TT_BENCHMARK_DATA=/abs/path/to/dataset-root   (one subdirectory per task_id)
# STATS_DIR=/abs/path/to/stats_out              (optional; defaults to ./stats_out)

python3 data_analysis/analyze.py            # corpus totals, label inventory
python3 data_analysis/per_conversation.py   # per-conversation metrics
python3 data_analysis/plot_per_type.py      # figures
```

### Website parity (`eval.parity`)

The leaderboard site runs a TypeScript port of this scorer in the browser.
`uv run python -m eval.parity <out>` emits `dev-gold.json` plus parity test
vectors from this canonical scorer; the site vendors them and its vitest
asserts the TS scorer reproduces the scores exactly (`scorer_sha` is the
staleness tripwire).
