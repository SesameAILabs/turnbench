# TurnBench

[![ci](https://github.com/cmu-sesame/turnbench/actions/workflows/ci.yml/badge.svg)](https://github.com/cmu-sesame/turnbench/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

A turn-taking benchmark for **end-of-turn (EOT)** and **interruption (INT)** detection in recorded, two-channel human conversations, scored on recall,
false-positive rate, and detection latency. This corpus contains 154 dyadic
conversations (~30 h, 106 speakers, 6 conversation types), each
triple-annotated.

**Leaderboard and interactive conversation viewer:
[turnbench.sesame.com](https://turnbench.sesame.com)**

## Submission

A submission is a single `predictions.json` containing the timestamps of your model's EOT and INT detections for each conversation and speaker.

```bash
uv sync
uv run python -m turnbench.score predictions.json     # score on the public dev set
```

- Format and rules on causality: [`docs/SUBMISSION_FORMAT.md`](docs/SUBMISSION_FORMAT.md)
- Scoring methodology: [`turnbench/README.md`](turnbench/README.md)
- A worked reference implementation: [`baselines/rms_vad/predict.py`](baselines/rms_vad/predict.py)

## Dataset

**Dev dataset:** [mundo-ai/turn-benchmark-dev](https://huggingface.co/datasets/mundo-ai/turn-benchmark-dev) (audio + labels, scoreable locally)  
**Test dataset:** [mundo-ai/turn-benchmark-test](https://huggingface.co/datasets/mundo-ai/turn-benchmark-test) (audio only, labels withheld)

Both datasets are gated and require a Hugging Face token to access.

## Splits

**Dev split:** [`turnbench/splits/dev.txt`](turnbench/splits/dev.txt)  
**Test split:** [`turnbench/splits/test.txt`](turnbench/splits/test.txt) 

## Baselines

One directory per baseline under `baselines/`. Each baseline runs its model over the
dataset and emits a `predictions.json` at its own operating point.

## Scripts

| Artifact | Command |
| --- | --- |
| Table III (per-type corpus overview) | `uv run python turnbench/analysis/consensus_by_type.py --latex` |
| Table IV / leaderboard (test) | `uv run python turnbench/analysis/results_by_conversation_type.py --dataset mundo-ai/turn-benchmark-test-golden --latex` |
| Agreement stats (Cohen's/Fleiss' kappa, boundary F1) | `uv run python turnbench/analysis/iaa_agreement.py` |
| Timing distributions vs Switchboard (gap/pause/FTO) | `uv run python turnbench/analysis/timing_distributions.py` |
| Fig. 2 threshold sweep + operating points | `uv run python -m turnbench.sweep baselines/<name>/probs-<task>.json` |

## Repo layout

```
turnbench/           — the benchmark (installable package)
  gold.py              builds the gold event sets from the annotator tracks (2/3 majority)
  score.py             scores a predictions.json (recall / FP-rate / latency)
  submission.py        the predictions.json schema + validators
  data.py              resolves the dataset (HF dev set by default, or --dataset)
  sweep.py             threshold sweep + operating-point selection over probs files
  probs.py             fetches the baseline probs files from their pinned HF dataset
  analysis/            corpus statistics, paper tables, and figures
baselines/           — one directory per baseline
tests/               — the scorer/gold/schema test suite
```

## Citation

```bibtex
@misc{jiang2026turnbench,
  title  = {TurnBench: A Multi-Domain Benchmark for Turn-Taking Dynamics in Spoken Dialogue},
  author = {Jiang, Freeman and Sanabria, Ramon and Deshmukh, Soham and Veluri, Bandhav and
            Williams, Simon Michael Vuch and Suen, Elliott K. and Lee, Garreth and
            Choi, Kevin Yoonho and Umeki, Takuya and Kubo, Riku and Udupa, Sathvik and
            Huang, Chien-yu and Kuan, Shih-Yun Shan and Tao, Zhuoyan and Krishna, Satyapriya and
            Eskimez, Sefik Emre and Tsao, Yu and Lee, Hung-yi and Watanabe, Shinji},
  year   = {2026},
  note   = {Under review}
}
```

See [`CITATION.cff`](CITATION.cff) for the complete citation.

## License

The code in this repository is released under the [MIT License](LICENSE). 

The data is hosted on Hugging Face and licensed separately with a non-commercial license that prohibits voice cloning (see the LICENSE file in each dataset repository).