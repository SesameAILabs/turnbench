# gpt_live_1

VAD-based readout of **OpenAI GPT-Live-1** (`gpt-live-1`, released 2026-09-10),
OpenAI's full-duplex voice model, evaluated with the generative protocol used
for [`gemini_vad`](../gemini_vad) and [`moshi_vad`](../moshi_vad): the model
converses in real time with each dataset speaker, its output audio is recorded
aligned with the input, and its turn-taking decisions are read off that audio.

GPT-Live exposes no detector interface. Unlike the Realtime API used by
[`openai_server_vad`](../openai_server_vad) and
[`openai_semantic_vad`](../openai_semantic_vad), the Live API emits no
speech-start/stop or turn events, only transcript deltas (documented as "not
definitive turn boundaries") and output audio. The audio readout is therefore
the only measurement available.

## Pipeline

### 1. Record — `record.py`

Per conversation x direction K, `speaker_K_audio` (resampled to 24 kHz mono
PCM16, peak-normalised to -3 dBFS like the Gemini runs) is streamed into a
Live session in 20 ms chunks paced at wall-clock real time, as the API
requires. Session config: `system_prompt.txt` as `instructions`, voice
`marin`, client delegation with no backend. The prompt states that no backend
exists; if the model delegates anyway, the recorder answers the delegation with
a `thinking.append` saying so, so the conversation does not stall (the dev
pilot produced no delegations).

Output audio arrives as `session.output_audio.delta` events at real-time pace
with no playback timing fields. The recorder simulates a jitter-free client
player: each delta is placed at max(arrival offset on the input clock, end of
the previous delta). This is what a listener hears, and it is the same
arrival-anchored convention `openai_semantic_vad` uses for commit times.

Layout, matching the Gemini recorder:

```
recordings/<task_id>/speaker_K/output.flac    # agent audio, 24 kHz mono, input-aligned
recordings/<task_id>/speaker_K/events.jsonl   # transcripts, delegations, timing, usage
recordings/<task_id>/speaker_K/.done          # written only after a complete send
```

Resume-safe per direction: rerun the same command after any interruption.

```bash
# needs openai>=3.13.0 (first release with Live support, 2026-09-10); pass
# --exclude-newer while it is inside the repo's 7-day uv cooldown
uv run --exclude-newer 2026-12-31 --with 'openai[realtime]==3.13.0' \
    --with scipy --with python-dotenv \
    python baselines/gpt_live_1/record.py --split dev --parallel 20

uv run --exclude-newer 2026-12-31 --with 'openai[realtime]==3.13.0' \
    --with scipy --with python-dotenv \
    python baselines/gpt_live_1/record.py --split test --parallel 20
```

`OPENAI_API_KEY` comes from the environment or the repo-root `.env`. Cost is
session time: $0.05/min, about 60 h of sessions for both splits (two
directions per conversation), so roughly $180 total. Wall-clock is bounded by
real-time pacing divided by `--parallel` (rate limits are counted in concurrent
sessions per usage tier).

### 2. Readout — `predict.py`

Imports the pyannote-VAD readout from `baselines/gemini_vad` unchanged and
points it at these recordings:

```
eot_speaker_K = agent VAD onsets in direction K while user_K is VAD-inactive,
                committed at the onset.
```

**EOT only. Interruption lists are committed empty**, as for `gemini_vad` and
`moshi_vad`: the benchmark has no INT methodology for full-duplex models (see
the `gemini_vad` README for the reasoning).

```bash
uv run --with 'torch==2.8.0' --with 'torchaudio==2.8.0' \
    --with 'pyannote.audio==3.3.2' --with omegaconf \
    python baselines/gpt_live_1/predict.py --out baselines/gpt_live_1/predictions-dev.json

uv run --with 'torch==2.8.0' --with 'torchaudio==2.8.0' \
    --with 'pyannote.audio==3.3.2' --with omegaconf \
    python baselines/gpt_live_1/predict.py --dataset mundo-ai/turn-benchmark-test \
    --out baselines/gpt_live_1/predictions-test.json
```

`pyannote/segmentation` is a gated HF model (accept its terms; `HF_TOKEN`).

## Results

Scores: [leaderboard](https://turnbench.sesame.com) · `results/leaderboard-test.json`.

## Caveats

- Paid external API, network-only, not bit-reproducible (server behaviour
  drifts; the model has no dated snapshot beyond `gpt-live-1`).
- Reported latency includes the model's response onset plus client-server
  round-trip, read through a non-causal VAD (pyannote, ~2 s of context); see the
  `gemini_vad` README for how that affects latency comparability.
- The model's speaking behaviour depends on the prompt; this run uses the same
  minimal prompt as the Gemini baseline plus the no-backend delegation policy.
- Recorded after the paper (SLT 2026); not one of the paper's 14 baselines.
