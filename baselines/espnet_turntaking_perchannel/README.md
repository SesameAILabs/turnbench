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
[`../README.md`](../README.md)): **θ_eot = 0.35**, **θ_int = 0.25**. Reproduce via
[How to run](#how-to-run).

| task | recall | fp_rate | latency p10/p50/p90 |
| --- | --- | --- | --- |
| EOT | 0.550 | 0.080 | 97 / 914 / 2281 ms |
| INT | 0.470 | 0.074 | 169 / 1233 / 2511 ms |

## Mix vs. individual-channel (each at its `fp_rate ≤ 0.1` operating point)

Same model, same dev set; only the inference strategy differs. Each baseline is
scored at its own committed operating point (`recall / fp_rate`):

| task | mix + attribution (`espnet_turntaking`) | individual-channel (this) |
| --- | --- | --- |
| EOT | 0.347 / 0.094 | **0.550 / 0.080** |
| INT | **0.637 / 0.091** | 0.470 / 0.074 |

The better strategy is per-track:

- **End-of-turn favours individual-channel** (recall 0.55 vs 0.35). EOT is
  largely a *single-speaker* event ("this speaker stopped"), which an isolated
  channel captures cleanly.
- **Interruption favours the mix** (recall 0.64 vs 0.47). Interruption is
  *relational* — it needs the other speaker present — and the model was trained
  on the mix, so an isolated channel is out-of-distribution for it.
- **Cost:** individual-channel is 2× the inference compute (two passes per
  conversation) and commits later. A hybrid — individual-channel EOT + mix
  interruption — would likely beat either alone.

## How to run

`predict.py` (ESPnet + torch) fills the per-frame probability cache; `submit.py`
derives the submission artifacts from that cache with no model re-run.

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
uv run python -m eval.sweep baselines/espnet_turntaking_perchannel/probs-eot.json   # → θ_eot (0.35)
uv run python -m eval.sweep baselines/espnet_turntaking_perchannel/probs-int.json   # → θ_int (0.25)

# --- 3) committed predictions at (θ_eot, θ_int) ---
python -m baselines.espnet_turntaking_perchannel.submit predictions --split dev  --theta-eot 0.35 --theta-int 0.25 \
    --out baselines/espnet_turntaking_perchannel/predictions-dev.json
python -m baselines.espnet_turntaking_perchannel.submit predictions --split test --theta-eot 0.35 --theta-int 0.25 \
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
- `predictions-dev.json` / `predictions-test.json` — committed events at θ_eot=0.35 / θ_int=0.25.
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
