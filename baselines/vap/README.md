# vap

Voice Activity Projection (VAP): a two-stream GPT-like transformer over CPC features that produces continuous floor-holding probabilities at 50Hz. Trained to predict future voice activity of both speakers simultaneously, `p_now[:, spk]` gives the probability that speaker `spk` holds the floor in the next 0–0.4s. Inference uses overlapping 25s windows (20s context + 5s step) to simulate bounded real-time context.

**Model:** [VapGPT](https://github.com/ErikEkstedt/VoiceActivityProjection) — transformer over 50Hz CPC features.

## Setup

Clone the VoiceActivityProjection repo and install it as a package:

```bash
git clone https://github.com/ErikEkstedt/VoiceActivityProjection baselines/vap/VoiceActivityProjection
pip install -e baselines/vap/VoiceActivityProjection
pip install -r baselines/vap/requirements.txt
```

The pretrained checkpoint is included in the cloned repo at `VoiceActivityProjection/example/VAP_3mmz3t0u_50Hz_ad20s_134-epoch9-val_2.56.pt`. Fine-tuned checkpoints (oto, swbd, swbd_oto) are downloaded automatically from `viks66/VAP_checkpoints` on first run.

## Run

```bash
bash baselines/vap/run.sh               # default: dev + test (oto checkpoint)
bash baselines/vap/run.sh --dev         # dev only
bash baselines/vap/run.sh --dev --pretrained   # dev, pretrained checkpoint
bash baselines/vap/run.sh --test        # test (needs prior --dev run for probs)
```

## Results (dev, oto checkpoint, operating point θ=0.90)

| Task | Recall | FP-rate | Latency p10 | Latency p50 | Latency p90 |
| --- | ---: | ---: | ---: | ---: | ---: |
| EOT | 0.833 | 0.065 | −47 ms | 419 ms | 1648 ms |
| INT | 0.922 | 0.064 | 516 ms | 1087 ms | 2286 ms |
