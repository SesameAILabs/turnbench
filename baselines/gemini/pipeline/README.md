# Gemini Live Inference (LiveKit)

Batch full-duplex inference for **Google Gemini 2.5 Live** and **Gemini 3.1 Flash Live (preview)** over the `dev_release_flac` dataset — no browser or web UI required.

Built on [LiveKit Agents](https://docs.livekit.io/agents/): an agent process hosts the realtime model, and a headless client streams each speaker's audio into a room while recording the model's spoken response, time-aligned with the input.

## Supported Models

| Provider     | Model                                  | Required Keys    |
| ------------ | -------------------------------------- | ---------------- |
| `gemini-2.5` | Google Gemini 2.5 Live API             | `GOOGLE_API_KEY` |
| `gemini-3.1` | Google Gemini 3.1 Flash Live (preview) | `GOOGLE_API_KEY` |

The agent resolves the provider from the **room-name prefix** (e.g. room `gemini-3.1-batch-0-t1` → Gemini 3.1), so a single agent process can serve both models at once. `LK_PROVIDER` is only the fallback default.

## Dataset & Output Layout

Input dataset (`/bathrooms/kcire/sesame/data/dev_release_flac`):

```
dev_release_flac/
├── 105/
│   ├── speaker_1_audio.flac    ← inference input 1
│   ├── speaker_2_audio.flac    ← inference input 2
│   └── … (annotations, combined_audio.flac, metadata.json — unused)
├── 106/
└── …
```

Each speaker file is run through the model in a **separate inference**. The output mirrors the dataset hierarchy:

```
results/<provider>/
├── system_prompt.txt           ← prompt snapshot for the run
├── 105/
│   ├── speaker_1/
│   │   ├── speaker_1_audio.flac   ← copy of the input
│   │   ├── output.wav             ← model response (24 kHz mono PCM-16)
│   │   └── output.flac            ← same response, FLAC
│   └── speaker_2/
│       ├── speaker_2_audio.flac
│       ├── output.wav
│       └── output.flac
├── 106/
└── …
```

The response is **time-aligned** with the input: same duration, with the model's audio placed at the offset it arrived during streaming.

## Files

```
scripts/
├── lk_agent.py          # LiveKit voice agent (gemini-2.5 / gemini-3.1)
├── lk_audio_client.py   # Headless client: streams input, records response
├── run_batch.sh         # Batch orchestrator (discovers tasks, runs workers)
├── run_worker.sh        # Per-subset worker (called by run_batch.sh)
├── system_prompt.txt    # System prompt given to the agent
├── .env.local.example   # Environment variable template
└── .env.local           # Your credentials (do NOT commit)
```

## Setup

```bash
conda create -n livekit python=3.12 -y
conda activate livekit

pip install "livekit-agents[google]~=1.3" \
            "livekit[crypto]~=1.0" \
            python-dotenv numpy soundfile scipy
```

Then configure credentials:

```bash
cp .env.local.example .env.local
# fill in LIVEKIT_URL, LIVEKIT_API_KEY, LIVEKIT_API_SECRET, GOOGLE_API_KEY
```

## Batch Inference

Run the whole dataset (38 folders × 2 speakers = 76 inferences; each file is ~10 min and streams in real time, so parallelism matters):

```bash
conda activate livekit

# Gemini 2.5 Live
bash run_batch.sh --provider gemini-2.5

# Gemini 3.1 Flash Live (preview)
bash run_batch.sh --provider gemini-3.1
```

`run_batch.sh` automatically:

1. Discovers all `speaker_{1,2}_audio.flac` tasks in the dataset
2. Skips tasks whose `output.flac` already exists (safe to re-run / resume)
3. Splits the remaining tasks into manifests of `--subset-size` each
4. Starts `lk_agent.py dev` (unless `--no-agent`)
5. Launches up to `--max-parallel` workers, each in its own LiveKit room
6. Aggregates per-worker results at the end

Both providers can run **in parallel** (separate terminals): start one with the agent, the second with `--no-agent` to reuse it — the room-name prefix routes each task to the right model:

```bash
# Terminal 1
bash run_batch.sh --provider gemini-2.5

# Terminal 2 (reuses the agent from terminal 1)
bash run_batch.sh --provider gemini-3.1 --no-agent
```

### Options

```
run_batch.sh --provider NAME [options]

  --provider NAME       gemini-2.5 | gemini-3.1   (required)
  --data-dir DIR        Input dataset directory
                        (default: /bathrooms/kcire/sesame/data/dev_release_flac)
  --output-dir DIR      Output directory (default: results/<provider>)
  --output-base DIR     Base for the auto output dir
  --silence SEC         Trailing silence appended to each input (default: 0)
  --max-parallel N      Max concurrent workers (default: 10)
  --subset-size N       Max tasks per worker (default: 10)
  --system-prompt PATH  System prompt file (default: scripts/system_prompt.txt)
  --no-agent            Don't start lk_agent.py (agent already running)
```

Logs live under `results/<provider>/.batch_work/` (`agent.log`, `worker_*.log`).

## Single-File Inference

**Terminal 1** — start the agent:

```bash
python lk_agent.py dev
```

**Terminal 2** — stream one file:

```bash
python lk_audio_client.py \
    -i /bathrooms/kcire/sesame/data/dev_release_flac/105/speaker_1_audio.flac \
    -o output.wav \
    --room gemini-2.5-test
```

Writes `output.wav` **and** `output.flac`. Use a `gemini-3.1-…` room name to target Gemini 3.1 instead.

## Output Format

- **Sample rate:** 24 kHz, mono, 16-bit PCM (both `.wav` and `.flac`)
- **Duration:** exactly the same as the input (time-aligned)

## Troubleshooting

| Issue | Solution |
| --- | --- |
| Agent doesn't join the room | Make sure `lk_agent.py dev` is running and shows "ready". The agent auto-dispatches when a participant joins. |
| Output is all silence | The model may not have spoken within the input window. Check `agent.log`; verify `GOOGLE_API_KEY`. |
| `LIVEKIT_URL` / key errors | Verify `.env.local`. Run `lk cloud auth && lk app env -w` to refresh credentials. |
| Resuming an interrupted run | Just re-run the same `run_batch.sh` command — completed tasks (with `output.flac`) are skipped. |
