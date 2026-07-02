# Moshi baseline

Kyutai **Moshi** (Defossez et al., 2024) — a full-duplex spoken-language
model — evaluated with the generative-model protocol: the model converses
with each dataset speaker, and its turn-taking decisions are read out of
the audio it produces.

## Pipeline

Three stages; stages 1–2 are expensive (GPU + real-time streaming) and
cached on disk, stage 3 is what `predict.py` runs.

### 1. Inference — `pipeline/inference_moshi_dev_release.py`

Server: the official [kyutai-labs/moshi](https://github.com/kyutai-labs/moshi)
repo, installed per its guide and run with `python -m moshi.server`
(default checkpoint `kyutai/moshiko-pytorch-bf16`).

For every task and each direction (speaker 1 as user, then speaker 2):
stream `speaker_K_audio` in real time over websocket into the server
and record the model's output stream, time-aligned with the input
(output sample i ≈ input sample i; ~80 ms server skip-frame padding at
the start). Input is streamed continuously — the model's own speech never
pauses the input. After the input ends, recording continues until the
model has been silent for 10 s (this tail is trimmed again in stage 3).

```
python pipeline/inference_moshi_dev_release.py \
    --server_ip <moshi-server:port> \
    --input  $TT_BENCHMARK_DATA \
    --output <AUDIO_OUT>/moshi
```

Produces `<AUDIO_OUT>/moshi/<task_id>/speaker_K/{speaker_K_audio.flac,output.wav,output.flac}`.

### 2. ASR — `pipeline/asr_generic.py` (driver: `pipeline/asr_batch.sh`)

Word-level timestamps with `nvidia/parakeet-tdt-0.6b-v2` for both the
model output (`output.wav`) and the user input (`speaker_K_audio.flac`).
Long-audio settings: local attention + 300 s chunked transcription with
timestamp offsetting (the dataset's 10–15 min files OOM full attention
on small GPUs).

```
python pipeline/asr_generic.py \
    --input_dir <AUDIO_OUT>/moshi --output_dir <MOSHI_ASR_DIR> \
    --filenames output.wav speaker_1_audio.flac speaker_2_audio.flac \
    --local_attention --chunk_sec 300 --skip_existing
```

Produces `<MOSHI_ASR_DIR>/<task_id>/speaker_K/{output.json,speaker_K_audio.json}`
(`{"text", "chunks": [{"text", "timestamp": [start_s, end_s]}]}`).

### 3. Traces — `predict.py` (shared logic in `../asr_floor.py`)

Converts the cached ASR JSONs into the unified submission format:

- word intervals separated by < **0.5 s** are merged into speech regions;
- each **agent speech onset** (silent → speaking transition) becomes a
  single-frame pulse of 1.0 at **12.5 Hz**:
  - user silent at the onset → `eot_score_speaker_K` (the agent judged
    the user's turn over),
  - user speaking at the onset → `interruption_score_speaker_J` where J
    is the seat the agent occupies (the agent barges in);
- arrays are trimmed to the conversation duration (drops the post-input
  response tail), so all four arrays share one length per task.

Direction K fills `eot_score_speaker_K` and `interruption_score_speaker_J`;
the two directions together fill all four arrays exactly once.

```
# .env: TT_BENCHMARK_DATA=...  MOSHI_ASR_DIR=...
python baselines/moshi/predict.py     # → predictions/moshi/
```

## Environments

Each stage runs in its own environment (versions are what produced the
handed-in predictions):

- **Stage 1 — inference** (conda env `moshi`, Python 3.12): the official
  [kyutai-labs/moshi](https://github.com/kyutai-labs/moshi) package
  provides the server and most client deps; the streaming client also
  needs scipy and websockets.

  ```
  conda create -n moshi python=3.12
  pip install moshi==0.2.12 scipy==1.17.0 websockets==16.0 soundfile==0.13.1
  python -m moshi.server          # serves kyutai/moshiko-pytorch-bf16 on :8998
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
- ASR-as-VAD is "semantic": non-lexical vocalizations (backchannel
  hums, laughter) in the output audio do not produce words and therefore
  do not count as the agent holding the floor. For a VAD-based readout
  of Moshi recordings — which catches non-lexical output — see
  [`../moshi_vad`](../moshi_vad/README.md).
