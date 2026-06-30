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
2. **Operating point** — tuned on dev. For EOT and interruption **independently**, pick the threshold that gives the **lowest latency at `fp_rate ≤ 0.1`** on dev, and commit your `predictions-{dev,test}.json` at that point. This is so operating points are comparable across models.
3. **Reproducible** from your `README.md` alone.

## Dev threshold sweep

To illustrate the latency vs false-interruption trade-off, the paper sweeps a threshold over the raw per-frame probabilities for the EOT / interruption task. If your model produces continuous probabilities, commit `probs-eot.json` and/or `probs-int.json` with the per-frame dev-set probabilities to be included in this analysis — the same sweep then picks your operating point (rule 2) and is scored centrally.

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
