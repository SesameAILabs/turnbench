# wavlm_large_anchor

A turn-taking model based on the **ANCHOR** framework (AR Universa), using a
frozen **WavLM-Large** frontend (~315 M parameters) with a 4-layer Transformer
audio encoder and a 12-layer autoregressive Transformer decoder. This is the
**TT-only** variant — the decoder vocabulary contains only 6 turn-taking tokens
(5 classes + meta), no quality metrics. Trained on TurnBench with hard
turn-taking labels. Inference uses 4-second sliding windows at 40 ms stride.

## Architecture

```
WavLM-Large (frozen)
  → 4-layer Transformer audio encoder
  → 12-layer AR Transformer decoder (6-token TT-only vocabulary)
  → 5-class turn-taking distribution {C, NA, I, BC, T}
```

Total: ~628 M parameters (313 M trainable). Within each 4 s window, attention
is bidirectional, but windows are independent and each ends at the current time,
so no future audio is ever observed. Frame rate: 25 Hz (40 ms stride), first
prediction at 200 ms. Declared lookahead: **0 ms**.

## Probability signals

- **EOT:** P(NA) + P(T) — end-of-turn detected as a transition to silence.
- **INT:** 1 - P(C) — interruption detected as a drop in continuation
  probability. P(I) alone is too sparse for this model; 1-P(C) captures the
  same event more reliably.

## Operating point (rule 2: highest recall at fp_rate ≤ 0.1)

```
θ_eot ≈ 0.89     (turnbench.sweep on probs-eot.json — score-quantile candidates)
θ_int ≈ 0.1214   (turnbench.sweep on probs-int.json)
```

Scores: [leaderboard](https://turnbench.sesame.com) · `results/leaderboard-test.json`.

`probs-{eot,int}.json` (dev) and `probs-test-{eot,int}.json` are emitted by
`predict.py --probs-out-dir` (sharded via `--shard K N`, merged with
`turnbench/analysis/merge_prob_shards.py`); predictions are committed centrally from
those files (`turnbench/analysis/finalize_ops.py`).

## How to reproduce

### 1. Environment

The ANCHOR model is built on ESPnet's **Universa** framework, which lives on an
open upstream PR (espnet/espnet#5959) plus author-fork modules published in the
HF repo. The working recipe (verified end-to-end):

```bash
pip install espnet soundfile numpy huggingface_hub h5py g2p_en jamo espnet_tts_frontend
# s3prl: espnet's pinned fork + route masked WavLM attention to the reference slow
# path (every public s3prl asserts attn_mask is None on the fast path; see the
# wavlm_base_causal README for the exact 2-line guard change)
pip install "git+https://github.com/espnet/s3prl.git@6553a49"

# universa framework (the branch behind espnet/espnet#5959)
git clone --depth 1 -b uni_versa.sh https://github.com/ftshijt/espnet.git espnet-universa

# overlay the author's fork modules from the HF repo (ar_universa model class,
# task registry with causal_s3prl + defer_full_meta, tokenizer, base classes) —
# copy DEREFERENCED (the HF cache stores symlinks):
python - <<'EOF'
from huggingface_hub import snapshot_download
import shutil, subprocess
p = snapshot_download("ZhuoyanTao/causal-wavlm-turn-taking", allow_patterns=["espnet2/**"])
subprocess.run(f"cd {p} && find espnet2 -type l -o -type f | while read f; do "
               "mkdir -p espnet-universa/$(dirname $f); cp -L $f espnet-universa/$f; done",
               shell=True, check=True, cwd=".")
EOF

# two fork-only classes are imported by the task registry but never instantiated
# by this checkpoint — satisfy the imports with raise-if-instantiated stubs:
#   espnet2/universa/base_flexible_type.py   (UniversaBaseFlexibleType)
#   espnet/nets/pytorch_backend/transformer/embedding.py  (append RoPEPositionalEncoding stub)
# (see PR #59 discussion for the stub bodies)

# run everything with the universa tree shadowing the installed espnet:
export PYTHONPATH=/abs/path/to/espnet-universa
```

`TT_TF32=1` enables TF32 (~1.5× on H100; gated numerics check showed max prob
delta 0.0034 on this checkpoint — used for the committed test probs; dev is fp32).

### 2. Checkpoint

Available on HuggingFace: [`ZhuoyanTao/causal-wavlm-turn-taking`](https://huggingface.co/ZhuoyanTao/causal-wavlm-turn-taking)
(`universa_turn_taking_only_turn_a40/{config.yaml, valid.loss.best.pth}`).

Loaded via ESPnet's `UniversaTask.build_model_from_file`. Config.yaml, tokenizer
data, and WavLM-Large are downloaded automatically on first run.

### 3. Run predict.py

```bash
python -m baselines.wavlm_large_anchor.predict                   # score on dev
python -m baselines.wavlm_large_anchor.predict --out preds.json  # write predictions
```

### 4. Get operating point

```bash
uv run python -m turnbench.sweep baselines/wavlm_large_anchor/probs-eot.json   # → θ_eot = 0.90
uv run python -m turnbench.sweep baselines/wavlm_large_anchor/probs-int.json   # → θ_int = 0.20
```

## Files

- `predictions-dev.json` — committed events at θ_eot=0.90, θ_int=0.20.
- `predictions-test.json` — same operating point, test split.
- `probs-eot.json` — per-frame P(NA)+P(T) on dev (25 Hz grid).
- `probs-int.json` — per-frame 1-P(C) on dev (25 Hz grid).
- `predict.py` — sliding-window inference (downloads checkpoint from HF).
- `README.md` — this file.
