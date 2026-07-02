# gemini_vad

VAD-based readout of **Google Gemini 3.1 Live** (`gemini-3.1-flash-live-preview`)
as a representative commercial full-duplex streaming voice agent. Same
recorded conversations as [`baselines/gemini`](../gemini) — the model
talks to each dataset speaker in real time and its output audio is saved
sample-aligned with the input — but a different **readout**.

## Strategy: pyannote VAD on the agent output (no ASR)

The `gemini` baseline runs word-level ASR on the model output and reads
turn-taking events off word timings. That misses non-lexical vocalisations
(agent hums, laughs, throat-clears): if no word lands, the ASR-VAD says
"agent silent" even when the agent is clearly holding the floor.

This baseline instead runs **pyannote/segmentation VAD** on the agent
output and reads events off the region boundaries directly. Per direction
K ∈ {1, 2} (Gemini playing the role opposite speaker K):

```
eot_speaker_K          = agent VAD onsets while user_K is VAD-inactive.
                         Committed at the onset.
```

`MERGE_GAP_S=0.5` merges VAD regions separated by ≤500 ms, so the
predictor operates on contiguous "agent speaking" spans, not per-frame
flicker.

**EOT only — the interruption lists are committed empty.** The natural
VAD-side INT readout (agent VAD *offset* while the user is active nearby,
i.e. the agent yielding the floor) measures the agent's yield decision,
not the barge-in onset. Its commit times are offset-anchored, so its
latencies are not comparable with the onset-anchored INT convention every
other baseline uses (user speech onset; cf. `baselines/openai_realtime.py`),
and it also fires at ordinary turn exchanges (agent stops, user replies
within the window). The ASR readout in [`baselines/gemini`](../gemini)
remains the INT source for Gemini.

## Results (official `eval.score` of the committed predictions)

| Split | Track | Readout | recall | fp_rate | latency p10/p50/p90 (ms) |
| --- | --- | --- | --- | --- | --- |
| dev | **EOT** | **pyannote VAD (this)** | **0.665** | **0.047** | 747 / 1197 / 1937 |
| **test** | **EOT** | **pyannote VAD (this)** | **0.657** | **0.022** | 847 / 1233 / 1951 |
| dev | EOT | ASR words (`gemini`)* | 0.554 | 0.087 | 428 / 725 / 1189 |

Test row scored against `mundo-ai/turn-benchmark-test-golden`; the
dev→test recall delta (−0.008) is noise-level.

\* The ASR row is from the same recordings run through the
`baselines/gemini` word-timestamp readout; those predictions are not yet
committed, so treat it as indicative until they are.

Operating point: `onset=0.5, offset=0.363` (fixed; not swept). It sits
at fp_rate 0.047 on dev, under the 0.1 budget, so no threshold sweep was
run.

### Observations

1. **VAD beats ASR-VAD on EOT recall.** Non-lexical output (backchannels,
   filler vocalisations) is real turn-holding and gets picked up by
   acoustic VAD but not by ASR.
2. **The cost is latency.** pyannote is non-causal (~2 s of bidirectional
   context inside the segmentation model), so p50 latency is ~450 ms
   higher than the ASR readout. This is inherent to the VAD, not to the
   readout rules — swapping in a causal VAD would reclaim it.

## How to run

Stage 1 (record) and Stage 2 (nothing here — no ASR) use the shared
Gemini pipeline; see [`../gemini/README.md`](../gemini/README.md) for the
LiveKit and direct-API recorders (`pipeline/run_split.py` is the
resumable driver that recorded the committed test split). Stage 3 is
this predictor.

Environment: the repo's `eval` extra plus pyannote. The versions below
are load-bearing — pyannote 4.x requires torchcodec (CUDA-13 builds fail
on older drivers), and pyannote 3.x needs torchaudio ≤ 2.8:

```bash
uv pip install 'torch==2.8.0' 'torchaudio==2.8.0' \
    'pyannote.audio==3.3.2' 'omegaconf'
```

`pyannote/segmentation` is a gated HF model: accept its conditions on
your HF account and make sure no narrower `HF_TOKEN` env var shadows
your CLI login when running.

```bash
# score dev in place
uv run --extra eval python -m baselines.gemini_vad.predict \
    --sample-runs baselines/gemini/sample_runs

# write dev predictions JSON
uv run --extra eval python -m baselines.gemini_vad.predict \
    --sample-runs baselines/gemini/sample_runs \
    --out baselines/gemini_vad/predictions-dev.json

# same, test split (needs test-split Gemini recordings)
uv run --extra eval python -m baselines.gemini_vad.predict \
    --dataset mundo-ai/turn-benchmark-test \
    --sample-runs baselines/gemini/sample_runs \
    --out baselines/gemini_vad/predictions-test.json

# validate
uv run python -m eval.check baselines/gemini_vad
```

`predict.py` runs pyannote VAD once per audio file and caches regions to
`--cache-dir` (default `baselines/gemini/.vad_cache/`). The cache is
keyed by file only, not by VAD parameters — after changing `VAD_PARAMS`
or `MERGE_GAP_S`, delete the cache dir or stale regions will be served.

## Files

- `predict.py` — self-contained predictor (pyannote VAD + boundary
  readout).
- `predictions-dev.json` — committed dev predictions at the operating
  point above.
- `predictions-test.json` — committed test predictions, same operating
  point, from the test-split recordings made by
  `../gemini/pipeline/run_split.py` (116 conversations × 2 directions).

## Notes

- **Non-causal VAD.** pyannote/segmentation reads ~2 s of bidirectional
  context per frame; the readout rules add no lookahead of their own.
  Latencies above are honest — they include the model's inherent
  lookahead — but a strictly causal VAD would improve p50 by ~400–500 ms.
- **Binary output.** No confidence scores; there's no threshold sweep and
  no `probs-{eot,int}.json`. The pyannote `onset`/`offset` are already
  tuned defaults.
- **Test predictions provenance.** The test-split recordings (116
  conversations, ~92 h of audio) were made with
  `../gemini/pipeline/run_split.py` (resumable, `.done`-marker gated) and
  read out with this predictor unchanged (`--dataset
  mundo-ai/turn-benchmark-test`). All 232 directions recorded cleanly;
  every conversation contributes events (per-direction minimum 26 EOTs).
