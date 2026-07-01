# openai_semantic_vad

EOT/interruption baseline using the **OpenAI Realtime API's `semantic_vad`**
(content-aware endpointing — decides when a speaker has *finished what they were
saying*, not just paused). Each speaker's channel is streamed into its own Realtime
session; `speech_stopped` → EOT, `speech_started` → interruption. No responses are
generated (`create_response: false`). Runs at the API's **stock defaults**
(`eagerness auto`) — a single fixed operating point (no threshold sweep). Shares its
engine with `openai_server_vad` (`baselines/openai_realtime.py`); the semantic
counterpart to the acoustic one.

## Setup

```bash
uv pip install -r baselines/openai_semantic_vad/requirements.txt
echo "OPENAI_API_KEY=sk-..." >> .env   # repo-root, shared by both openai_* baselines; gitignored
```

## Run

```bash
python -m baselines.openai_semantic_vad.predict                      # score on dev
python -m baselines.openai_semantic_vad.predict --out preds.json     # or write predictions
python -m baselines.openai_semantic_vad.predict --dataset mundo-ai/turn-benchmark-test --out preds.json
```

Each channel is resampled to 24 kHz PCM16 and streamed in real time (20 ms chunks);
conversations fan out in parallel (`--concurrency`), and `--resume` skips those
already in `--out`. Commit time = audio position heard when the event arrived
(causal; folds in the model's variable decision delay).

## Results (dev)

| Task | Recall | FP-rate | Latency p10/50/90 (ms) |
| --- | ---: | ---: | ---: |
| EOT | 0.310 | 0.037 | 437 / 763 / 2311 |
| INT | 0.553 | 0.320 | 118 / 206 / 695 |

## Caveats

- Paid external API, network-only, not bit-reproducible (server behavior drifts); the
  exact model (`gpt-realtime`) is recorded in `baselines/openai_realtime.py`.
- **Reported latency includes client↔server round-trip.**
- Conservative by design: low fp, but low recall and slower than the acoustic VAD.
