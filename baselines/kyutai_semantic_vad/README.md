# kyutai_semantic_vad

Kyutai STT-1B semantic VAD for end-of-turn detection. Uses the VAD head (`vad_heads[2]`) of the Kyutai STT-1B model to produce continuous EOT probabilities at 12.5Hz. Both speaker channels are batched together (batch_size=2) through a single streaming pass — one forward pass per frame for both speakers.

**Model:** [`kyutai/stt-1b-en_fr-candle`](https://huggingface.co/kyutai/stt-1b-en_fr-candle) — streaming ASR with a VAD extra head.

**Score direction:** P(turn ending); `probs-eot = score`, `probs-int = 1 − score`.

## Setup

```bash
pip install -r baselines/kyutai_semantic_vad/requirements.txt
```

Model weights are loaded from `kyutai/stt-1b-en_fr-candle` (cached in `$HF_HOME`).

## Run

```bash
bash baselines/kyutai_semantic_vad/run.sh          # dev + test + turnbench.check
bash baselines/kyutai_semantic_vad/run.sh --dev    # dev probs + predictions only
bash baselines/kyutai_semantic_vad/run.sh --test   # sweep existing probs → test predictions
```

## Results (pretrained, swept operating point @ fp ≤ 0.1)

| Task | θ | dev recall/fp | test recall/fp |
| --- | ---: | ---: | ---: |
| EOT | 0.8125 | 0.803 / 0.100 | 0.773 / 0.059 |
| INT | 0.9734 | 0.934 / 0.085 | 0.898 / 0.081 |
