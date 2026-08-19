# smart_turn_v3

Pipecat Smart Turn v3 — Whisper Tiny encoder + linear classifier for semantic end-of-turn detection. Runs a VAD+accumulate+settling pipeline at 12.5Hz per speaker channel.

**Model:** ONNX (~8M params). Accepts up to 8s of 16kHz mono audio, returns P(turn complete).

**Score direction:** P(turn complete); floor held when score < threshold.

## Setup

The pinned `smart_turn` submodule commit was force-removed upstream, so
`git submodule update --init` fails ("not our ref") — clone `pipecat-ai/smart-turn`
at HEAD instead. The v3.1 ONNX weights aren't vendored; download them from
`pipecat-ai/smart-turn-v3` on HF. Note the naming: `inference.py` builds a session
at import against the un-suffixed `smart-turn-v3.1.onnx`, while `predict.py`
overrides it with the `-gpu` variant on CUDA — so fetch both and alias the
un-suffixed name:

```bash
git clone https://github.com/pipecat-ai/smart-turn baselines/smart_turn_v3/smart_turn
D=baselines/smart_turn_v3/smart_turn
huggingface-cli download pipecat-ai/smart-turn-v3 smart-turn-v3.1-gpu.onnx --local-dir "$D"
huggingface-cli download pipecat-ai/smart-turn-v3 smart-turn-v3.1-cpu.onnx --local-dir "$D"
cp "$D/smart-turn-v3.1-cpu.onnx" "$D/smart-turn-v3.1.onnx"   # inference.py's import-time model
pip install -r baselines/smart_turn_v3/requirements.txt
```

## Run

```bash
bash baselines/smart_turn_v3/run.sh          # dev + test + turnbench.check
bash baselines/smart_turn_v3/run.sh --dev    # dev probs + predictions only
bash baselines/smart_turn_v3/run.sh --test   # sweep existing probs → test predictions
```

## Operating point (swept @ fp ≤ 0.1)

θ_eot = 0.0082, θ_int = 0.0155 (`turnbench.sweep`).

Scores: [leaderboard](https://turnbench.sesame.com) · `results/leaderboard-test.json`.

Test tracks dev closely. INT recall is real but weak: this is primarily an
end-of-turn detector, and the small in-budget interruption signal only becomes
visible with quantile-resolution threshold candidates (a fixed grid found none).
