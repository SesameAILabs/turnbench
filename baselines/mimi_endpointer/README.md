# mimi_endpointer

Two-stream LSTM over Kyutai Mimi embeddings, trained on SpokenWOZ with label-delayed supervision. Each speaker channel is encoded through Mimi (24 kHz, 8 quantizers, 25 Hz after upsample) and processed by a shared unidirectional 2-layer LSTM (512 hidden, 128 projection) with a 5-class head: `[bos, system_end, user_end, system, user]`.

**Model:** [`viks66/mimi-endpointer`](https://huggingface.co/viks66/mimi-endpointer) — oto_d1f checkpoint is the main submission.

**Score direction:**
- EOT: `1 - P(user)` (high = turn ending); fires when score rises above θ_eot.
- INT: `P(user)` (high = taking floor); fires when score rises above θ_int.

## Setup

```bash
pip install -r baselines/mimi_endpointer/requirements.txt
```

Place (or symlink) the oto_d1f checkpoint at `baselines/mimi_endpointer/checkpoint.pt`, or omit `--checkpoint` to download from HuggingFace.

## Run

```bash
# dev: probs + auto-threshold predictions + score + sweep tables
bash baselines/mimi_endpointer/run.sh --dev --oto

# test: sweep existing dev probs → pick threshold → test predictions
bash baselines/mimi_endpointer/run.sh --test --oto

# both in one shot
bash baselines/mimi_endpointer/run.sh --oto

# other checkpoints
bash baselines/mimi_endpointer/run.sh --dev --swbd
bash baselines/mimi_endpointer/run.sh --dev --swbd-oto
bash baselines/mimi_endpointer/run.sh --dev          # pretrained (prefixed outputs)
```

## Results (oto_d1f, swept operating point @ fp ≤ 0.1)

| Task | θ | dev recall/fp | test recall/fp |
| --- | ---: | ---: | ---: |
| EOT | 0.92 | 0.759 / 0.047 | 0.782 / 0.078 |
| INT | 0.9224 | 0.876 / 0.100 | 0.899 / 0.106 |
