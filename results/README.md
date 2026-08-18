# Results

Final scored artifacts, regenerated whenever baselines' committed
predictions or operating points change.

- `leaderboard-test.json` — the official test leaderboard: every baseline's
  committed `predictions-test.json` scored against the held-out gold at the
  benchmark's fp budget. Regenerate with:

  ```
  uv run python turnbench/analysis/leaderboard.py \
      --dataset mundo-ai/turn-benchmark-test-golden \
      --json results/leaderboard-test.json
  ```

  The website (`turnbench` repo) serves a manual copy at
  `site/src/lib/leaderboard.json` — re-copy it after regenerating here.
