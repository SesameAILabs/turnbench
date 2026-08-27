# TurnBench

[![arXiv](https://img.shields.io/badge/arXiv-2608.25218-b31b1b.svg)](https://arxiv.org/abs/2608.25218)
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
**Training dataset:** [otoearth/otoSpeech-full-duplex-turn-104h](https://huggingface.co/datasets/otoearth/otoSpeech-full-duplex-turn-104h) (~104 h, speaker-disjoint from the benchmark, same annotation protocol; used for the TurnBench-trained baselines)

All datasets are gated and require a Hugging Face token to access.

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
| Table III (raw per-annotator dynamics) | `uv run python turnbench/analysis/per_conversation.py` |
| Agreement stats (Cohen's/Fleiss' kappa, boundary F1) | `uv run python turnbench/analysis/iaa_agreement.py` |
| Human latency reference (floor-transfer offsets, INT yield) | `uv run python turnbench/analysis/human_baseline.py` |
| Timing distributions vs Switchboard (gap/pause/FTO) | `uv run python turnbench/analysis/timing_distributions.py` |
| Fig. 2 threshold sweep + operating points | `uv run python -m turnbench.sweep baselines/<name>/probs-<task>.json` |

## Citation

```bibtex
@misc{jiang2026turnbench,
  title         = {TurnBench: A Multi-Domain Benchmark for Turn-Taking Dynamics in Spoken Dialogue},
  author        = {Jiang, Freeman and Sanabria, Ramon and Deshmukh, Soham and Veluri, Bandhav and
                   Williams, Simon Michael Vuch and Suen, Elliott K. and Lee, Garreth and
                   Choi, Kevin Yoonho and Umeki, Takuya and Kubo, Riku and Udupa, Sathvik and
                   Huang, Chien-yu and Kuan, Shih-Yun Shan and Tao, Zhuoyan and Krishna, Satyapriya and
                   Eskimez, Sefik Emre and Tsao, Yu and Lee, Hung-yi and Watanabe, Shinji},
  year          = {2026},
  eprint        = {2608.25218},
  archivePrefix = {arXiv},
  primaryClass  = {eess.AS},
  url           = {https://arxiv.org/abs/2608.25218}
}
```

See [`CITATION.cff`](CITATION.cff) for the complete citation.

## License

The code in this repository is released under the [MIT License](LICENSE). 

The data is hosted on Hugging Face and licensed separately with a non-commercial license that prohibits voice cloning (see the LICENSE file in each dataset repository).