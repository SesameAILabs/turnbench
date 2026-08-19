# oracle_annotator

Replays the gold events as the prediction (no training or parameters): reads the
three annotator tracks and emits the 2/3-majority consensus events, the same gold
the scorer builds via `turnbench.gold`. Perfect by construction (recall 1.0, fp-rate
0.0) — the ceiling of the benchmark and a sanity check on the scorer.

## Input

The three per-speaker annotator tracks (not audio).

## Output

The gold EOT and interruption event times.

## Predictions

No massaging — the gold events are already discrete. Run over every conversation:

```bash
# dev (score in-place — prints recall=1.0, fp=0.0 — or write a JSON)
uv run python -m baselines.oracle_annotator.predict
uv run python -m baselines.oracle_annotator.predict --out baselines/oracle_annotator/predictions-dev.json
```

Needs HF access to the gated dev repo. **Dev only:** the test set's gold is
withheld (annotation columns are empty), so there is nothing to replay.

Perfect scores are the design property, not a finding: the predictor replays
the gold, so recall is 1, fp-rate is 0, and latency is 0 by construction.
