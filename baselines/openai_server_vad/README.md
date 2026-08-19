# openai_server_vad

EOT/interruption baseline using the **OpenAI Realtime API's `server_vad`** (acoustic
silence VAD). Each speaker's channel is streamed into its own Realtime session;
`speech_stopped` → EOT, `speech_started` → interruption. No responses are generated
(`create_response: false`). Runs at the API's **stock defaults** — a single fixed
operating point (no threshold sweep). Shares its engine with `openai_semantic_vad`
(`baselines/openai_realtime.py`); the acoustic counterpart to the semantic one.

## Setup

```bash
uv pip install -r baselines/openai_server_vad/requirements.txt
echo "OPENAI_API_KEY=sk-..." >> .env   # repo-root, shared by both openai_* baselines; gitignored
```

## Run

```bash
python -m baselines.openai_server_vad.predict                      # score on dev
python -m baselines.openai_server_vad.predict --out preds.json     # or write predictions
python -m baselines.openai_server_vad.predict --dataset mundo-ai/turn-benchmark-test --out preds.json
```

Each channel is resampled to 24 kHz PCM16 and streamed in real time (20 ms chunks);
conversations fan out in parallel (`--concurrency`), and `--resume` skips those
already in `--out`. Commit time = audio position heard when the event arrived
(causal; folds in the decision delay).

## Results

Scores: [leaderboard](https://turnbench.sesame.com) · `results/leaderboard-test.json`.

## Caveats

- Paid external API, network-only, not bit-reproducible (server behavior drifts); the
  exact model (`gpt-realtime`) is recorded in `baselines/openai_realtime.py`.
- **Reported latency includes client↔server round-trip.**
- Acoustic onset over-fires on backchannels, so INT fp is high.
