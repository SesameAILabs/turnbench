# moshi_vad

VAD-based readout of **Kyutai Moshi** (`kyutai/moshiko-pytorch-bf16`,
Défossez et al. 2024) — an open full-duplex spoken-language model —
evaluated with the generative protocol: the model converses in real time
with each dataset speaker, its output audio is recorded sample-aligned
with the input, and its turn-taking decisions are read off that audio.

Same readout strategy as [`baselines/gemini_vad`](../gemini_vad); the
paired [`baselines/moshi`](../moshi) baseline reads a word-level ASR
transcript instead and so misses non-lexical vocalisations (hums,
backchannels, laughs) — a large share of Moshi's floor-holding.

## Strategy: pyannote VAD on the agent output (no ASR)

Per direction K ∈ {1, 2} (Moshi conversing with dataset speaker K):

```
eot_speaker_K          = agent VAD onsets while user_K is VAD-inactive.
                         Committed at the onset.
```

**EOT only — the interruption lists are committed empty**, as in
`gemini_vad`. An onset-anchored "user onset inside an agent VAD region"
readout is possible (it mirrors `baselines/moshi`'s ASR-side rule), but
for a generative model it measures *passive floor-overlap*, not
interruption **detection**: the model's only contribution to each event
is "was I speaking there". The detection signal a full-duplex model
could actually exhibit — yielding after a barge-in — is measurable, and
Moshi shows none:

| After a user barge-in… | median time-to-stop | stops within 1 s |
| --- | --- | --- |
| Moshi (1,927 test barge-ins) | 1.38 s | 37% |
| Moshi counterfactual (random moment, same region) | 1.44 s | 39% |
| Gemini (7,294 test barge-ins) | **0.46 s** | **71%** |
| Gemini counterfactual | 1.43 s | 40% |

Gemini's pipeline demonstrably self-cancels on barge-ins; Moshi's speech
ends when it was going to end anyway. Since the yield itself cannot be
scored under the benchmark's causal rules (offset-anchored timestamps,
or lookahead at commit time), the INT track is out of scope for VAD
readouts of generative models.

## Operating point (swept on dev, per the repo protocol)

Moshi holds the floor only ~4–8% of frames when fed one side of a
human-human conversation, so the stock pyannote thresholds sit far from
its optimum. The VAD threshold is this prob-less baseline's θ: we swept
onset × offset × merge-gap on dev (segmentation scores extracted once
per file; thresholds applied offline) and froze the setting with the
highest recall at fp_rate ≤ 0.1:

| Task | onset | offset | merge gap |
| --- | --- | --- | --- |
| EOT | 0.88 | 0.862 | 0.15 s |

The optimum is high-onset / narrow-hysteresis / small-gap — it fragments
Moshi's brief, quiet floor-taking into many crisp onsets. Recall falls
off above onset ≈ 0.9 (regions start vanishing) and monotonically below
it (regions merge; the user-side gate also over-triggers).

## Results (official `turnbench.score` of the committed predictions)

| Split | Track | recall | fp_rate | latency p10/p50/p90 (ms) |
| --- | --- | --- | --- | --- |
| dev | EOT | 0.212 | 0.066 | −125 / 771 / 2371 |
| **test** | EOT | **0.233** | **0.044** | −132 / 702 / 2489 |

Test row scored against `mundo-ai/turn-benchmark-test-golden` at the
dev-frozen operating point; the dev→test transfer is clean (EOT recall
+0.021 at lower fp).

For reference, the ASR readout in PR #58 scored dev EOT 0.243 at
fp_rate 0.119 — **over the 0.1 budget**, hence not a valid operating
point.

The honest headline: even with a generous acoustic readout at its swept
optimum, Moshi rarely takes the floor at the right moments in
human-human material. An oracle readout that fires on *every* agent
speech onset reaches only 0.298 test EOT recall (at fp 0.132) on these
recordings — the same oracle over the Gemini recordings reaches 0.725 at
fp 0.032 — so the gap is the model's behaviour, not the readout.

Two mechanisms, separable in the recordings:

1. **Long-session collapse dominates.** Moshi's mean floor share by
   conversation quarter is 21.5% → 10.3% → 7.2% → 1.0% (Gemini:
   24.9/25.0/25.0/15.0). It engages at Gemini-like levels for the first
   ~3 minutes — about its 5-minute training-sequence horizon — then
   slides into silence as an absorbing state (its own recent stream
   history is silence, and in its training data silence predicts
   silence). The committed score therefore partly measures long-session
   robustness, not just turn-timing.
2. **Distribution shift on the rest.** Even while engaged, its onsets
   place poorly against human turn ends: the instruct tuning (20k h of
   synthetic assistant-user dialogues) teaches it to speak when
   addressed, and one side of a natural human-human conversation mostly
   isn't. Unlike the engineered commercial pipelines (which force a
   response whenever VAD detects the user stopped), Moshi's turn-taking
   is a pure learned policy — nothing external compels a reply.

## Pipeline

### 1. Inference — `../moshi/pipeline/run_fleet.py`

Records Moshi's output for every (conversation, speaker): a fleet of
`moshi.server` instances (one session each; 3/GPU sustains ≥1× real-time
on H100 — 4/GPU measured 0.92× and truncates tails) fed from a shared
job queue by `inference_moshi_dev_release.py`. Resumable; ~50 min for
dev and ~2.5 h for test on 8×H100 (streaming is hard real-time paced).

```bash
# flac layout for the client (no local delivery needed)
uv run --extra eval python baselines/moshi/pipeline/export_flac_dataset.py \
    --split dev --out <audio>/dev
# record (moshi venv: moshi==0.2.12 scipy websockets soundfile)
python baselines/moshi/pipeline/run_fleet.py \
    --input <audio>/dev --output <moshi_out>/dev --num-servers 24 --num-gpus 8
```

### 2. Readout — `predict.py`

```bash
python -m baselines.moshi_vad.predict --sample-runs <moshi_out>/dev \
    --vad-workers 48 --out baselines/moshi_vad/predictions-dev.json
python -m baselines.moshi_vad.predict \
    --dataset mundo-ai/turn-benchmark-test --sample-runs <moshi_out>/test \
    --vad-workers 48 --out baselines/moshi_vad/predictions-test.json
uv run python -m turnbench.check baselines/moshi_vad
```

`--vad-workers` shards the segmentation passes round-robin across GPUs
(or CPU cores with `CUDA_VISIBLE_DEVICES=`). The cache
(`.vad_cache/*.npz`, gitignored) stores per-frame segmentation *scores*,
not regions — thresholds and merge gaps are applied at read time, so
changing `PARAMS` never serves stale results and needs no recompute.

Environment: the repo's `eval` extra plus pyannote — same pinned stack
as [`baselines/gemini_vad`](../gemini_vad/README.md) (versions are
load-bearing):

```bash
uv pip install 'torch==2.8.0' 'torchaudio==2.8.0' \
    'pyannote.audio==3.3.2' 'omegaconf'
```

`pyannote/segmentation` is a gated HF model: accept its conditions on
your HF account and make sure no narrower `HF_TOKEN` env var shadows
your CLI login.

## Files

- `predict.py` — self-contained predictor (pyannote segmentation scores
  + hysteresis threshold + boundary readout).
- `predictions-dev.json` — committed dev predictions at the swept
  operating point above.
- `predictions-test.json` — committed test predictions at the same
  frozen operating point (116 conversations × 2 directions, recorded by
  the fleet in one pass, 232/232 complete).

## Notes

- **Non-causal VAD.** As in `gemini_vad`: pyannote/segmentation reads
  ~2 s of bidirectional context per frame; the readout rules add no
  lookahead of their own. Latencies include the model's inherent
  lookahead.
- **Prob-less baseline.** The committed events are threshold crossings of
  pyannote's score on *recorded audio*, not a per-frame turn-taking
  probability from Moshi itself, so no `probs-{eot,int}.json` is shipped;
  the dev sweep above plays the role of the threshold sweep.
