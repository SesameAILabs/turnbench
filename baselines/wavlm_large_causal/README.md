# wavlm_large_causal

A lightweight, fully causal turn-taking predictor using a frozen
**WavLM-Large** encoder (~315 M parameters) with causal attention masking.
Trained on Switchboard + TurnBench (train_joint) with hard turn-taking labels.

## Architecture

```
WavLM-Large (frozen, causal-masked)
  → Conv1d subsample (stride 2, 1024d → 256d)
  → 4-layer causal Transformer encoder (256d, 4 heads, FFN 1024)
  → Linear head → 5 classes {C, NA, I, BC, T}
```

Total: ~320 M parameters (4 M trainable, backbone frozen).

The model runs in a **single causal forward pass** per speaker channel — no
sliding windows. Each frame's prediction depends only on audio up to that
frame. Frame rate: 25 Hz (40 ms stride), first prediction at 200 ms.
Declared lookahead: **0 ms**.

## Probability signals

- **EOT:** P(NA) + P(T) — end-of-turn is detected as a transition to silence.
  P(T) alone is too sparse; combining it with P(Silence) captures the signal
  that the speaker is finishing.
- **INT:** P(I) — interruption probability directly.

## Operating point (rule 2: highest recall at fp_rate ≤ 0.1)

```
θ_eot ≈ 0.8323   (turnbench.sweep on probs-eot.json — score-quantile candidates)
θ_int ≈ 0.1844   (turnbench.sweep on probs-int.json)
```

Scores: [leaderboard](https://turnbench.sesame.com) · `results/leaderboard-test.json`.

Commitment: central rising-edge detector (`turnbench.sweep.commit_events`, refractory 2.0 s).
`probs-{eot,int}.json` (dev) and `probs-test-{eot,int}.json` are emitted by this
`predict.py` (`--probs-out-dir`); predictions are committed centrally from those
files (`turnbench/analysis/finalize_ops.py`), so artifacts and code stay self-consistent.

## How to reproduce

### 1. Environment

```bash
pip install espnet soundfile numpy huggingface_hub
# s3prl: use espnet's pinned fork; every public s3prl asserts attn_mask is None on
# WavLM's fast attention path, which the causal streaming mask needs. Route masked
# calls to the reference (slow) path — a 2-line, semantics-preserving guard change:
pip install "git+https://github.com/espnet/s3prl.git@6553a49"
python - <<'EOF'
import pathlib, s3prl
f = pathlib.Path(s3prl.__path__[0]) / "upstream/wavlm/modules.py"
t = f.read_text()
t = t.replace("            and self.q_head_dim == self.head_dim\n        ):\n            assert key is not None and value is not None\n            assert attn_mask is None",
              "            and self.q_head_dim == self.head_dim\n            and attn_mask is None\n        ):\n            assert key is not None and value is not None")
f.write_text(t)
EOF
```

`CausalS3prlFrontend` is **not** in stock ESPnet — install it from the HF repo:

```bash
wget -P $(python -c "import espnet2; print(espnet2.__path__[0])")/asr/frontend/ \
    https://huggingface.co/ZhuoyanTao/causal-wavlm-turn-taking/resolve/main/espnet2/asr/frontend/causal_s3prl.py
```

### 2. Checkpoint

Available on HuggingFace: [`ZhuoyanTao/causal-wavlm-turn-taking`](https://huggingface.co/ZhuoyanTao/causal-wavlm-turn-taking)
(`tt_pred_large_turn_swbd_res/valid.loss.best.pth`).

The checkpoint is self-contained — model config is embedded; no config.yaml needed.
WavLM-Large is auto-downloaded by s3prl on first run.

### 3. Run predict.py

```bash
python -m baselines.wavlm_large_causal.predict                   # score on dev
python -m baselines.wavlm_large_causal.predict --out preds.json  # write predictions
```

### 4. Get operating point

```bash
uv run python -m turnbench.sweep baselines/wavlm_large_causal/probs-eot.json   # → θ_eot = 0.85
uv run python -m turnbench.sweep baselines/wavlm_large_causal/probs-int.json   # → θ_int = 0.20
```

## Files

- `predictions-dev.json` — committed events at θ_eot=0.85, θ_int=0.20.
- `predictions-test.json` — same operating point, test split.
- `probs-eot.json` — per-frame P(NA)+P(T) on dev (25 Hz grid).
- `probs-int.json` — per-frame P(I) on dev (25 Hz grid).
- `README.md` — this file.
