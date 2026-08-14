# Switchboard reference statistics

Per-conversation turn-taking statistics for Switchboard-1 (LDC97S62) in the
metric schema of `data_analysis/per_conversation.py`: the Switchboard reference
row in the paper's per-type overview table, plus the Switchboard side of the
timing-distribution comparison (`plot_timing_distributions.py`). Vendored from
[`cyhuang-tw/swbd-statistics`](https://github.com/cyhuang-tw/swbd-statistics);
this copy is canonical.

Switchboard has no hand-labeled turn-taking events, so events are derived from
the MS-State word alignments, the VAP/Ekstedt backchannel list, and optionally
SWDA dialog-act tags. How comparable each metric is to the hand-annotated
benchmark numbers:

- **exact** (`word_rate_wpm`, `words_sp*`, `wpm_sp*`, `ttr_sp*`) — identical
  formula and input; same word regex as `per_conversation.py`.
- **analog** (silence, turns, FTO, speaker balance, overlap, laughter,
  non-content, backchannels) — same phenomenon, but events are reconstructed
  from word timings (same-speaker IPUs merged across the listener's
  backchannels) or read from the VAP detector list. Overlap counts run higher
  than annotator-marked overlaps; turn lengths are coarser.
- **approx** (`int_*`, `question_rate_per_min`) — heuristics for judgments SWBD
  never recorded: interruption is an overlap-with-floor-change proxy (no TRP
  information, likely over-counts); questions come from SWDA dialog acts on the
  1,155-conversation subset it covers.
- **blank** — backchannel/interruption subtypes (no SWBD information) and all
  inter-annotator-agreement columns (single annotator).

Per-metric detail: the script's docstrings, or the original writeup in the
upstream repo.

## Outputs (committed under `results/`)

| File | Content |
| --- | --- |
| `swbd_per_type_aggregate.csv` | the single Switchboard comparison row |
| `swbd_per_conversation.csv` | one row per conversation (2,438) |
| `swbd_per_type_aggregate.json` | mean/median/std/min/max/n per metric |
| `swbd_timing_distributions.json` | full FTO / pause distributions (gap = the positive FTOs) |

## Run

```bash
# Inputs (public downloads, not committed):
./data_analysis/swbd/fetch_backchannels.sh data_analysis/swbd/backchannels.csv
curl -LO https://isip.piconepress.com/projects/switchboard/releases/switchboard_word_alignments.tar.gz
tar xzf switchboard_word_alignments.tar.gz

uv run python data_analysis/swbd/per_conversation_swbd.py \
    --trans-root swb_ms98_transcriptions \
    --backchannels data_analysis/swbd/backchannels.csv \
    --swda-root '' \
    --out-dir data_analysis/swbd/results
# ~4 min, CPU only. Add --swda-root <path-to-cgpotts/swda> for question rate.
```

Knobs (defaults): `--ipu-gap 0.2` (merge a speaker's own words into an IPU
across ≤ this gap, seconds — also the floor of the pause distribution) and
`--trp-tol 1.0` (interruption onset window, seconds).
