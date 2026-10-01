# Results

Final scored artifacts, regenerated whenever baselines' committed
predictions or operating points change.

- `leaderboard-test.json`: the official test leaderboard. Every baseline's
  committed `predictions-test.json` is scored against the held-out gold.
  Regenerate it in two steps, in this order:

  ```
  uv run python turnbench/analysis/leaderboard.py \
      --dataset mundo-ai/turn-benchmark-test-golden \
      --json results/leaderboard-test.json
  uv run python turnbench/analysis/results_by_conversation_type.py \
      --dataset mundo-ai/turn-benchmark-test-golden \
      --json results/leaderboard-test.json
  ```

  The second step merges the per-conversation-type breakdown (`by_type`,
  `types`) into the same file. The leaderboard site serves a copy of this
  file; re-sync it after regenerating.
