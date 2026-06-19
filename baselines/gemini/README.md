# Gemini baseline

**Google Gemini 3.1 Live** (`gemini-3.1-flash-live-preview`) as a
representative commercial full-duplex streaming voice agent, evaluated
with the generative-model protocol: the model converses with each dataset
speaker, and its turn-taking decisions are read out of the audio it
produces.

## Pipeline

Three stages; stages 1–2 are expensive (API streaming in real time) and
cached on disk, stage 3 is what `predict.py` runs.

### 1. Inference — `pipeline/` (LiveKit client)

For every task and each direction (speaker 1 as user, then speaker 2):
`lk_audio_client.py` streams `speaker_K_audio` into a Gemini Live session
through LiveKit (`lk_agent.py` hosts the realtime model; `run_batch.sh` /
`run_worker.sh` drive the batch). The agent's output audio is recorded on
a wall-clock timeline and saved **sample-aligned with the input** (output
duration == input duration). System prompt: `pipeline/system_prompt.txt`.
Credentials via `pipeline/.env.local.example` → `.env.local`.

Produces `<AUDIO_OUT>/gemini/<model>/<task_id>/speaker_K/{speaker_K_audio.flac,output.wav,output.flac}`.

### 2. ASR — `pipeline/asr_generic.py` (driver: `pipeline/asr_batch.sh`)

Word-level timestamps with `nvidia/parakeet-tdt-0.6b-v2` for both the
model output (`output.wav`) and the user input (`speaker_K_audio.flac`).
Long-audio settings: local attention + 300 s chunked transcription with
timestamp offsetting.

```
python pipeline/asr_generic.py \
    --input_dir <AUDIO_OUT>/gemini/gemini-3.1 --output_dir <GEMINI_ASR_DIR> \
    --filenames output.wav speaker_1_audio.flac speaker_2_audio.flac \
    --local_attention --chunk_sec 300 --skip_existing
```

Produces `<GEMINI_ASR_DIR>/<task_id>/speaker_K/{output.json,speaker_K_audio.json}`.

### 3. Traces — `predict.py` (shared logic in `../asr_floor.py`)

Identical to the moshi baseline (see `baselines/moshi/README.md` for the
full description): 0.5 s word-gap merging, single-frame pulses at agent
speech onsets at 12.5 Hz — user silent → `eot_score_speaker_K`, user
speaking → `interruption_score_speaker_J` (agent's seat) — trimmed to the
conversation duration.

```
# .env: TT_BENCHMARK_DATA=...  GEMINI_ASR_DIR=...
python baselines/gemini/predict.py     # → predictions/gemini/
```

## Environments

Each stage runs in its own environment (versions are what produced the
handed-in predictions):

- **Stage 1 — inference** (conda env `livekit`, Python 3.12): LiveKit
  Agents with the Google realtime plugin; credentials per
  `pipeline/.env.local.example`.

  ```
  conda create -n livekit python=3.12
  pip install livekit-agents==1.4.4 livekit-plugins-google==1.4.6 \
      google-genai==1.66.0 python-dotenv==1.2.2 \
      soundfile==0.13.1 numpy==2.4.2
  ```

- **Stage 2 — ASR** (conda env `nemo`, Python 3.10): NVIDIA NeMo with the
  ASR collection; needs a CUDA GPU (≥20 GB works with the provided
  `--local_attention --chunk_sec 300` flags).

  ```
  conda create -n nemo python=3.10
  pip install "nemo-toolkit[asr]==2.7.0rc0" soundfile==0.13.1 tqdm==4.67.1
  ```

- **Stage 3 — traces** (`predict.py`): any Python ≥3.10 with `numpy` and
  `soundfile` (the repo's base requirements already cover it).

## Notes

- Scores are binary pulses, so the eval's threshold sweep is flat for
  this baseline — intrinsic to VAD/ASR-derived readouts of a generative
  model, which expose decisions, not confidences.
- ASR-as-VAD is "semantic": non-lexical vocalizations in the output audio
  do not produce words and therefore do not count as the agent holding
  the floor.
