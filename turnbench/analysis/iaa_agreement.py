#!/usr/bin/env python3
"""Corpus-level inter-annotator agreement — the paper's §IV-A agreement sentence.

Computes, over the full 154-dialogue corpus (dev + private golden test):

  * pairwise Cohen's kappa (a-b, a-c, b-c) and Fleiss' kappa, pooled at the
    frame level: each annotator's track is rasterised into 100 ms frames
    labelled with the canonical collapse the scorer uses (turnbench.gold.CANONICAL;
    unlabelled audio = silence), both speakers' frame sequences are
    concatenated per annotator, and frames are pooled across all
    conversations before computing kappa;
  * boundary F1 per annotator pair: within each conversation, onsets of all
    fine-labelled events (both speakers pooled) matched greedily within
    +/-200 ms, then averaged across conversations.

Prints one line per statistic. These regenerate the sentence "pairwise
Cohen's kappa reaches 0.77-0.80 and Fleiss' kappa is 0.78, and event onsets
agree at a boundary F1 of 0.94-0.96 within +/-200 ms" and the intro bullet's
"Fleiss kappa = 0.78".

Reads the HF releases column-projected, no audio download; needs HF
credentials (HF_TOKEN in the environment or repo-root `.env`).

Usage:
    uv run python turnbench/analysis/iaa_agreement.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


from turnbench.analysis.per_conversation import (  # noqa: E402
    BOUNDARY_TOL_S,
    WINDOW_S,
    boundary_f1,
    cohens_kappa,
    fleiss_kappa,
)
from turnbench.data import (  # noqa: E402
    ANNOTATORS,
    FULL_CORPUS_SOURCES,
    SPEAKERS,
    conversation,
    conversation_ids,
    resolve_dataset,
)
from turnbench.gold import CANONICAL  # noqa: E402

SILENCE = "_"


def canonical_frames(
    events: list[tuple[float, float, str, str]], n_windows: int
) -> np.ndarray:
    """One annotator track -> length-n_windows array of canonical labels.

    events: [(start_s, end_s, fine_label, text), ...] for one (speaker,
    annotator) track. Frames covered by no mapped event are SILENCE; fine
    labels outside turnbench.gold.CANONICAL are ignored, mirroring gold building.
    """
    arr = np.full(n_windows, SILENCE, dtype="<U26")
    for start, end, label, _text in events:
        if label not in CANONICAL:
            continue
        i = max(0, int(start / WINDOW_S))
        j = min(n_windows, int(np.ceil(end / WINDOW_S)))
        arr[i:j] = CANONICAL[label]
    return arr


def main() -> int:
    frame_seqs: dict[str, list[np.ndarray]] = {ann: [] for ann in ANNOTATORS}
    f1_per_conv: dict[tuple[str, str], list[float]] = {}
    pairs = [("a", "b"), ("a", "c"), ("b", "c")]

    for source in FULL_CORPUS_SOURCES:
        dataset = resolve_dataset(source=source, skip_audio=True)
        ids = conversation_ids(dataset)
        print(f"{source}: {len(ids)} conversations...", file=sys.stderr)
        for cid in ids:
            conv = conversation(dataset, cid)
            n_win = int(np.ceil(conv.duration_s / WINDOW_S))
            onsets = {}
            for ann in ANNOTATORS:
                frame_seqs[ann].append(
                    np.concatenate(
                        [
                            canonical_frames(conv.annotations[(sp, ann)], n_win)
                            for sp in SPEAKERS
                        ]
                    )
                )
                onsets[ann] = [
                    start
                    for sp in SPEAKERS
                    for start, _end, label, _text in conv.annotations[(sp, ann)]
                    if label
                ]
            for x, y in pairs:
                f1_per_conv.setdefault((x, y), []).append(
                    boundary_f1(onsets[x], onsets[y])
                )

    pooled = {ann: np.concatenate(frame_seqs[ann]) for ann in ANNOTATORS}
    print(f"pooled frames per annotator: {len(pooled['a'])} ({WINDOW_S*1000:.0f} ms)")
    kappas = [cohens_kappa(pooled[x], pooled[y]) for x, y in pairs]
    for (x, y), k in zip(pairs, kappas):
        print(f"cohen_kappa_{x}{y}: {k:.4f}")
    print(f"cohen_kappa_range: {min(kappas):.2f}-{max(kappas):.2f}")
    print(f"fleiss_kappa: {fleiss_kappa([pooled[a] for a in ANNOTATORS]):.4f}")
    f1_means = [float(np.mean(f1_per_conv[p])) for p in pairs]
    for (x, y), f1 in zip(pairs, f1_means):
        print(f"boundary_f1_{x}{y}_mean: {f1:.4f}  (+/-{BOUNDARY_TOL_S*1000:.0f} ms)")
    print(f"boundary_f1_range: {min(f1_means):.2f}-{max(f1_means):.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
