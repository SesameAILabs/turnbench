# Baselines — author guide

Each baseline is a standalone predictor under `baselines/<name>/` that emits a
`predictions.json` scored by `turnbench.score`
([format](../docs/SUBMISSION_FORMAT.md)). `rms_vad/` is the minimal reference.

## What to commit

In `baselines/<name>/`:

| File | What |
| --- | --- |
| `predictions-dev.json` | committed events at your declared operating point (dev) |
| `predictions-test.json` | same, test split (audio-only — scored by the eval server, not locally) |
| `probs-eot.json` / `probs-int.json` | per-frame EOT / interruption probabilities for the dev sweep — only if your model has a continuous score (see below); prob-less models (Gemini, binary SmartTurn) skip them and appear as a single point. Not committed to git: `uv run python -m turnbench.probs` fetches the pinned set, and new files are uploaded to the [probs dataset](https://huggingface.co/datasets/freemanjiang/turnbench-baseline-probs) (`turnbench/probs.py`) |
| `README.md` | machine requirements + exact commands to reproduce all of the above on a fresh clone.|

## Rules

1. **Causal** — output at `t` may use audio only up to `t`; fold lookahead into the timestamp ([details](../docs/SUBMISSION_FORMAT.md)).
2. **Operating point** — tuned on dev. Pick **independently for EOT and interruption**: for each task, the threshold giving the **highest recall at `fp_rate ≤ 0.1`** on dev (`turnbench.sweep` on that task's probs file prints it) — i.e. operate as aggressively as the false-positive budget allows. Candidate thresholds are the **quantiles of your model's own score distribution**, not a fixed uniform grid — the search is scale-invariant, so it finds the true optimum whether your scores concentrate near 0, near 1, or anywhere between. This gives two thresholds, θ_eot and θ_int; your single `predictions-{dev,test}.json` then carries its `eot` times committed at θ_eot and its `interruption` times at θ_int. The dev budget is where selection happens; on test your fp_rate must additionally stay within **1.5× the budget (0.15)** or the submission is rejected as miscalibrated — dev is publicly labeled, so the test-side bound is what keeps the gate honest. Honestly dev-calibrated operating points clear it comfortably (worst observed drift among the committed baselines: 0.135).
3. **Reproducible** from your `README.md` alone.

## Dev threshold sweep

To illustrate the latency vs false-interruption trade-off, the paper sweeps a decision threshold over the raw per-frame probabilities for the EOT / interruption task (the figure plots a uniform threshold grid; the *marked operating point* comes from the full quantile candidate set of rule 2, so it always reflects the committed submission). If your model produces continuous probabilities, add `probs-eot.json` and/or `probs-int.json` with the per-frame dev-set probabilities to be included in this analysis; the same sweep then picks your operating point (rule 2 above) and is scored centrally. Probs files live in the [probs dataset](https://huggingface.co/datasets/freemanjiang/turnbench-baseline-probs), not in git: upload yours there and bump `REVISION` in `turnbench/probs.py` (instructions in its docstring).

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

**Get your operating point (rule 2).** Run the sweep on each task's probs file; it sweeps θ on dev and prints the chosen threshold (highest recall at `fp_rate ≤ 0.1`):

```bash
uv run python -m turnbench.sweep baselines/<name>/probs-eot.json   # → θ_eot
uv run python -m turnbench.sweep baselines/<name>/probs-int.json   # → θ_int
```

Then commit `predictions-{dev,test}.json` with your `eot` times generated at θ_eot and `interruption` times at θ_int.

**Render the figure** (all baselines that have probs, scored in memory — no intermediate file):

```bash
uv run --extra plot python turnbench/analysis/plot_sweep.py \
    baselines/*/probs-eot.json --out sweep-eot.png
```

## Validate before submitting

One command validates every file present in your baseline directory — coverage,
in-audio event times, and (for probs) the frame grid:

```bash
uv run python -m turnbench.check baselines/<name>
```

It prints a line per file and exits non-zero if any fail. This is also the only
way to validate `predictions-test.json` — it can't be scored locally, since test
labels are withheld.
