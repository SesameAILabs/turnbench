# espnet_turntaking_perchannel

The **individual-channel** variant of the
[`espnet_turntaking`](../espnet_turntaking) baseline. Same model — the "Talking
Turns" frozen Whisper-medium encoder (~306 M) + 5-class head (Arora et al.,
ICLR 2025), emitting per-40 ms (25 Hz) probabilities over {Continuation(C),
Silence(NA), Interruption(I), Backchannel(BC), Turn-change(T)} — but a
different **inference strategy**.

## Strategy: two runs, one per channel (no mixing)

The `espnet_turntaking` baseline runs the model **once on the two-speaker mono
mix** (in-distribution with how the model was trained) and attributes events to
speakers with energy VAD. This baseline instead runs the model **twice — once
on each isolated speaker channel** — and reads each channel's native signals
directly as that speaker's events:

```
eot_speaker_K          = commit(P_T on speaker_K_audio)   # K's turn ending
interruption_speaker_K = commit(P_I on speaker_K_audio)   # K interrupting
```

No mix, no energy attribution: the per-channel run *is* the per-speaker signal.
Each channel's per-frame `P_T` (EOT) / `P_I` (interruption) is the continuous
score; the committed submission (`submit.py`) thresholds it with the central
single-threshold rising-edge rule (2 s refractory), per `eval.sweep`.

## Submission (dev operating point — highest recall at `fp_rate ≤ 0.1`)

Operating point chosen centrally by `eval.sweep` (rule 2 in
[`../README.md`](../README.md)): **θ_eot ≈ 0.309**, **θ_int ≈ 0.217**. Reproduce
via [How to run](#how-to-run).

| task | split | recall | fp_rate |
| --- | --- | --- | --- |
| EOT | dev | 0.640 | 0.100 |
| EOT | test | 0.711 | 0.081 |
| INT | dev | 0.585 | 0.096 |
| INT | test | 0.611 | 0.135 |

## Mix vs. individual-channel (each at its `fp_rate ≤ 0.1` operating point)

Same model, same dev set; only the inference strategy differs. Each baseline is
scored at its own swept operating point (`recall / fp_rate`, dev):

| task | mix + attribution (`espnet_turntaking`) | individual-channel (this) |
| --- | --- | --- |
| EOT | **0.836 / 0.074** | 0.640 / 0.100 |
| INT | **0.637 / 0.091** | 0.585 / 0.096 |

At the swept operating points **the mix wins both tracks** — a reversal of the
comparison at earlier, coarser operating points, driven entirely by the mix's
EOT score (`P_T × energy-hold`): its optimum lives at θ ≈ 0.0007, which coarse
threshold grids could not reach (it was previously stuck at 0.347 recall).
Per-channel inference remains the *architecturally* cleaner single-speaker EOT
signal, but the energy-weighted mix score turns out to rank turn-ends better —
and per-channel costs 2× the inference compute (two passes per conversation).

## How to run

`predict.py` (ESPnet + torch) fills the per-frame probability cache; `submit.py`
derives the submission artifacts from that cache with no model re-run.

**Quick reproduce.** `run.sh` shards `predict.py` over all visible GPUs → cache →
`submit.py` probs → `eval.sweep` operating point (highest recall at fp ≤ 0.1, swept
over score-quantile candidates) → `predictions-{split}.json` + `eval.check`. TF32 on by default (`TT_TF32=1`,
~2.3× on H100; `TT_TF32=0` for bit-exact fp32). Runs the model twice per
conversation (one pass per channel):

```bash
PYTHON=/path/to/espnet-venv/bin/python \
ESPNET_TT_EXP=/abs/.../asr_train_asr_whisper_turn_taking_raw_en_word \
  bash baselines/espnet_turntaking_perchannel/run.sh dev    # → probs-*.json + predictions-dev.json
  bash baselines/espnet_turntaking_perchannel/run.sh test   # → predictions-test.json (dev op)
```

Or the equivalent manual steps that `run.sh` wraps:

```bash
# --- 1) per-frame probabilities (model side; your espnet env) ---
export ESPNET_TT_EXP=/abs/path/to/exp/asr_train_asr_whisper_turn_taking_raw_en_word
pip install -r baselines/espnet_turntaking_perchannel/requirements.txt   # + espnet2 importable
python -m baselines.espnet_turntaking_perchannel.predict \
    --out baselines/espnet_turntaking_perchannel/predictions-dev.json    # fills predictions/espnet_turntaking_perchannel/cache
#   for test: --dataset <test repo> --shard i --num-shards n  (2× per conversation; same cache dir)

# --- 2) dev probs → operating point (rule 2: highest recall at fp_rate ≤ 0.1) ---
python -m baselines.espnet_turntaking_perchannel.submit probs --task eot --out baselines/espnet_turntaking_perchannel/probs-eot.json
python -m baselines.espnet_turntaking_perchannel.submit probs --task int --out baselines/espnet_turntaking_perchannel/probs-int.json
uv run python -m eval.sweep baselines/espnet_turntaking_perchannel/probs-eot.json   # → θ_eot (≈0.309)
uv run python -m eval.sweep baselines/espnet_turntaking_perchannel/probs-int.json   # → θ_int (≈0.217)

# --- 3) committed predictions at (θ_eot, θ_int) ---
python -m baselines.espnet_turntaking_perchannel.submit predictions --split dev  --theta-eot 0.3091361972333867 --theta-int 0.21735300794389617 \
    --out baselines/espnet_turntaking_perchannel/predictions-dev.json
python -m baselines.espnet_turntaking_perchannel.submit predictions --split test --theta-eot 0.3091361972333867 --theta-int 0.21735300794389617 \
    --out baselines/espnet_turntaking_perchannel/predictions-test.json

# --- 4) validate every committed file (only way to check the test file) ---
uv run python -m eval.check baselines/espnet_turntaking_perchannel
```

`predict.py` runs the model twice per conversation (once per isolated channel);
`--shard i --num-shards n` parallelises across GPUs, and a per-frame cache
(`--cache-dir`, default `predictions/espnet_turntaking_perchannel/cache`, holding
dev + test) makes `submit.py` instant. `submit.py` reads each channel's `P_T`/`P_I`,
lands them on the canonical grid `floor(dur·25)` by left-padding the 5-frame (0.2 s)
model pre-roll, and commits with the central `eval.sweep.commit_events` rule.

**Environment (reproducible).** The engine is **stock upstream ESPnet**
(`github.com/espnet/espnet`, `master`) — *not a fork*. Pin: commit `750e3749`
(v202604), `pip install -e .` provides `espnet2` (at this version ESPnet ships
`espnet2`/`espnet3` only — no `espnet1` package, so import `espnet2`). A clean
install also needs `torchaudio` matched to your `torch`, and ESPnet's pinned
**whisper fork** (`git+https://github.com/espnet/whisper.git`; the latest pypi
`openai-whisper` is incompatible — it removed `whisper.audio.N_MELS`).
`cd tools && make` handles both automatically; see `requirements.txt` for the
manual list. Verified: a from-scratch install at this pin reproduces the
committed per-frame probabilities **bit-for-bit**.

## Files

- `predict.py` — individual-channel model side (→ per-frame cache).
- `submit.py` — derives the submission artifacts from the cache (probs + predictions).
- `requirements.txt` — model-side deps (scorer deps come from the repo `eval` extra).
- `probs-eot.json` / `probs-int.json` — per-frame dev probabilities for the central sweep.
- `predictions-dev.json` / `predictions-test.json` — committed events at the swept operating point (θ_eot ≈ 0.309, θ_int ≈ 0.217).
- `run_dev_sharded.sbatch` / `run_test_sharded.sbatch` — sharded inference jobs
  used to produce the caches.

## Notes

- Frame rate **25 Hz**; channels resampled to the model's 16 kHz.
- Runtime is ~2× the mix baseline (two passes per conversation); once cached,
  rebuilding the JSON at new thresholds is instant.
- Caveat the numbers quantify: the model was trained on the two-speaker mix and
  its I/BC/T labels are relational, so an isolated channel is out-of-distribution
  for the relational tracks — which is exactly why interruption degrades while
  the single-speaker EOT improves.
