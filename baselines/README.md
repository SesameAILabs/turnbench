# Baselines — author guide

Each baseline is a standalone predictor under `baselines/<name>/` that emits a
`predictions.json` scored by `eval.score`
([format](../docs/SUBMISSION_FORMAT.md)). `rms_vad/` is the minimal reference.

## What to commit

In `baselines/<name>/`:

| File | What |
| --- | --- |
| `predictions-dev.json` | committed events at your declared operating point (dev) |
| `predictions-test.json` | same, test split (audio-only — scored by the eval server, not locally) |
| `probs-eot.json` / `probs-int.json` | per-frame EOT / interruption probabilities for the dev sweep — only if your model has a continuous score (see below); prob-less models (Gemini, binary SmartTurn) skip them and appear as a single point |
| `README.md` | machine requirements + exact commands to reproduce all of the above on a fresh clone.|

## Rules

1. **Causal** — output at `t` may use audio only up to `t`; fold lookahead into the timestamp ([details](../docs/SUBMISSION_FORMAT.md)).
2. **Operating point tuned on dev**.
3. **Reproducible** from your `README.md` alone.

## Dev threshold sweep

To illustrate the threshold trade-off on recall and false positive, the paper sweeps a threshold over the raw per-frame probabilities for the EOT/INT task. If your model produces continuous probabilities, please commit a `probs-dev.json` file with the probabilities for the dev set in order to be included in this analysis.

```json
{
  "schema_version": 1,
  "task": "eot",
  "frame_rate_hz": 25.0,
  "probs": [
    { "conversation_id": "20",
      "speaker_1": { "prob": [0.01, 0.04, 0.93] },
      "speaker_2": { "prob": [0.00, 0.02, 0.10] } }
  ]
}
```

## Validate before submitting

One command validates every file present in your baseline directory — coverage,
in-audio event times, and (for probs) the frame grid:

```bash
uv run python -m eval.check baselines/<name>
```

It prints a line per file and exits non-zero if any fail. This is also the only
way to validate `predictions-test.json` — it can't be scored locally, since test
labels are withheld.
