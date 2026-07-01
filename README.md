# TurnBench

A turn-taking benchmark: detecting **end-of-turn (EOT)** and **interruption
(INT)** events in two-channel recorded conversations, scored on recall,
false-positive rate, and detection latency. The corpus is 154 dyadic
conversations (~30 h, 106 speakers, 6 types), each triple-annotated.

## Submit / evaluate

A submission is a single `predictions.json`: per conversation and speaker, the
times your model commits to an EOT and an interruption.

```bash
uv sync --extra eval --extra dev
uv run python -m eval.score predictions.json     # score on the public dev set
```

Format and rules on causality are described in [`docs/SUBMISSION_FORMAT.md`](docs/SUBMISSION_FORMAT.md). A worked
reference implementation: [`baselines/rms_vad/predict.py`](baselines/rms_vad/predict.py).

## Gold (`eval/gold.py`)

Built on the fly from the three raw annotator tracks by **2/3 majority** with a ±200 ms endpoint tolerance. EOT positives are turn-ends where
the floor passes to the other speaker. INT positives are floor-taking
interruption onsets. Full methodology: [`eval/README.md`](eval/README.md).

## Dataset

Dev scoring is based on: [mundo-ai/turn-benchmark-dev](https://huggingface.co/datasets/mundo-ai/turn-benchmark-dev). Since annotations are published for dev, the scorer can be run locally against predictions to provide a signal to develop against. Test scoring is based on: [mundo-ai/turn-benchmark-test](https://huggingface.co/datasets/mundo-ai/turn-benchmark-test). Annotations are not provided.

Both HF datasets are gated and require authentication to access — run
`huggingface-cli login`, or put `HF_TOKEN=<token>` in a repo-root `.env` (the
scorer auto-loads it).

Raw corpus (source of truth): the gated GCS bucket
`gs://sesame-cmu-tt-benchmark-dev/full_delivery_with_metadata/` — per-`task_id`
dirs with `combined_audio.wav` + the two channels, the six SRTs, and a
`metadata.json`. The six types: `Argumentative/Deliberative`,
`Casual/Spontaneous`, `Collaborative/Problem-Solving`, `Instructional`,
`Narrative/Storytelling`, `Task-Oriented/Transactional`.

## Splits

**Use the `task_id`s the lists below define — don't re-split locally**, or
numbers stop being comparable across teams.

| File | Description | dev / test | Speaker overlap |
| --- | --- | ---: | ---: |
| `eval/splits/dev.txt` / `test.txt` | **Headline set.** Speaker-disjoint: every voice actor appears in exactly one partition. Type-balanced within ±1 of 25%. | 38 / 116 | **0** |
| `eval/splits/random_dev.txt` / `random_test.txt` | Random partition baseline. Same target ratio, but speakers leak across dev and test (intentional, for ablation). | 38 / 116 | 58 |

The 25% dev / 75% test target holds roughly per conversation type:

| Conversation type | dev | test | total | dev % |
| --- | ---: | ---: | ---: | ---: |
| Argumentative/Deliberative    |  8 |  24 |  32 | 25.0% |
| Casual/Spontaneous            |  7 |  22 |  29 | 24.1% |
| Collaborative/Problem-Solving |  6 |  20 |  26 | 23.1% |
| Instructional                 |  6 |  19 |  25 | 24.0% |
| Narrative/Storytelling        |  5 |  15 |  20 | 25.0% |
| Task-Oriented/Transactional   |  6 |  16 |  22 | 27.3% |
| **All**                       | **38** | **116** | **154** | **24.7%** |

`eval/make_splits.py` regenerates them from the raw dataset with a fixed seed.

## Baselines

Each baseline is a standalone predictor under `baselines/`: it runs its model
over the dataset and emits a `predictions.json`
([format](docs/SUBMISSION_FORMAT.md)), including its own continuous→discrete
thresholding. The benchmark only sees committed events — no shared trace format,
no central runner. `baselines/rms_vad/` is the minimal reference.

**Adding or updating a baseline?** See [`baselines/README.md`](baselines/README.md)
for the author checklist — what to commit, the causal/reproducibility rules, and
the dev threshold-sweep files (`probs-eot.json` / `probs-int.json`).

> `TODO:`-marked baselines are **stubs** — the model code is present but they do
> not yet emit a submission. `oracle_annotator` is a dev-only sanity check (it
> replays the gold, so has no test predictions); every other baseline runs on
> dev + test today.

| Baseline | Modality | Native output |
| --- | --- | --- |
| `rms_vad` | Energy VAD | Speech on/off edges → discrete events |
| `oracle_annotator` | Sanity check | Replays the gold events |
| `openai_server_vad` | OpenAI Realtime `server_vad` (acoustic silence) | `speech_stopped`→EOT, `speech_started`→INT |
| `openai_semantic_vad` | OpenAI Realtime `semantic_vad` (content-aware) | `speech_stopped`→EOT, `speech_started`→INT |
| `espnet_turntaking` | Frozen Whisper-medium, mono two-speaker (CMU, Arora et al. ICLR 2025) | 5-class @ 25 Hz, trained on Switchboard |
| `espnet_turntaking_perchannel` | Same model, per-channel inference | 5-class @ 25 Hz |
| `mimi_endpointer` | Mimi codec, 12.5 Hz | 4-class per frame {user, user-end, system, system-end} |
| `kyutai_semantic_vad` | Kyutai STT-1B VAD head + ASR | Binary EOT per frame |
| `vap` | Two-stream voice-activity projection, 50 Hz (Ekstedt) | Continuous VA projection per speaker |
| `smart_turn_v3` | Whisper-Tiny + linear head (Pipecat) | Binary per 8 s chunk (turn-complete) |
| `wavlm_base_causal` | Frozen WavLM-Base-Plus, fully causal (CMU) | 5-class @ 25 Hz, trained on Switchboard |
| `wavlm_large_anchor` | Frozen WavLM-Large, 4 s windows, AR decoder (CMU) | 5-class @ 25 Hz, trained on Switchboard |
| `wavlm_large_causal` | Frozen WavLM-Large, fully causal (CMU) | 5-class @ 25 Hz, trained on Switchboard + TurnBench |
| `causal_wavlm_predictor` | Frozen WavLM base+large, causal per-channel (CMU) | 5-class @ 25 Hz (base + large variants) |
| TODO: `gemini` | Full-duplex streaming dialogue | ASR/VAD-aligned timestamps over output audio |
| TODO: `moshi` | Audio (full-duplex, 12.5 Hz) | Per-frame voice-activity on system stream |
| TODO: `dualturn` | Qwen2.5-0.5B + Mimi codec (anyreach) | 12 per-channel classification heads |

## Repo layout

```
eval/                — the benchmark
  gold.py              builds the gold event sets from the annotator tracks (2/3 majority)
  score.py             scores a predictions.json (recall / FP-rate / latency)
  submission.py        the predictions.json schema + validators
  data.py              resolves the dataset (HF dev set by default, or --dataset)
  parity.py            emits the website's gold + parity bundle
  make_splits.py       regenerates the speaker-disjoint dev/test splits
baselines/           — one directory per baseline (see above)
data_analysis/       — corpus statistics & figures
```

### Website parity (`eval.parity`)

The leaderboard site runs a TypeScript port of this scorer in the browser.
`uv run python -m eval.parity <out>` emits `dev-gold.json` plus parity test
vectors; the site vendors them and asserts the TS scorer reproduces the scores
exactly (`scorer_sha` is the staleness tripwire).
