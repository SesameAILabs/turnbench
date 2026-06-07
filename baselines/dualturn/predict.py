#!/usr/bin/env python3
"""DualTurn — Qwen2.5-0.5B + Mimi codec for multi-signal turn-taking prediction.

Dual-channel audio is encoded by the Mimi codec (12.5Hz, 24kHz) and fed to a
Qwen2.5-0.5B backbone with LoRA adapters and 12 per-channel classification heads.

Signals used (per channel):
  eot_probs  — P(speech offset where other speaker takes floor within 4s)
  bot_probs  — P(speech onset following the other speaker)

Score mapping:
  eot_score_speaker_{1,2}          ← eot_probs[:, spk]
  interruption_score_speaker_{1,2} ← bot_probs[:, spk]

Model: anyreach-ai/dualturn-qwen2.5-mimi-0.5B (HuggingFace)
Frame rate: 12.5Hz
Input: 24kHz mono per channel, encoded via kyutai/mimi

Setup:
    git submodule update --init baselines/dualturn/dualturn
    cd baselines/dualturn/dualturn && pip install -e . && cd ../../..
    pip install transformers torch torchaudio soundfile huggingface_hub
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio
from huggingface_hub import hf_hub_download

_HERE    = Path(__file__).resolve().parent
_DT_REPO = _HERE / "dualturn"
sys.path.insert(0, str(_DT_REPO))

sys.path.insert(0, str(_HERE.parent))
from runner import run  # noqa: E402

MIMI_SR      = 24000
FRAME_RATE   = 12.5
CHUNK_FRAMES = 375          # 30s at 12.5Hz
HF_REPO      = "anyreach-ai/dualturn-qwen2.5-mimi-0.5B"
CKPT_FILE    = "best.pt"

_mimi_model  = None
_dt_model    = None
_input_mode  = None


def _load_models(device: str):
    global _mimi_model, _dt_model, _input_mode
    if _dt_model is not None:
        return

    print("Loading Mimi encoder...", flush=True)
    from transformers import MimiModel
    _mimi_model = MimiModel.from_pretrained("kyutai/mimi").to(device).eval()

    print("Downloading DualTurn checkpoint...", flush=True)
    ckpt_path = hf_hub_download(HF_REPO, CKPT_FILE)

    print("Loading DualTurn model...", flush=True)
    from evaluation.agent_action_eval import load_model
    _dt_model, _input_mode = load_model(ckpt_path, device)
    print(f"Models loaded. input_mode={_input_mode}", flush=True)


@torch.no_grad()
def _mimi_encode(audio_24k: torch.Tensor, device: str):
    """Encode mono 24kHz audio through Mimi.

    Returns:
      codes: [T, 8] int64  (discrete mode)
      feats: [T, 512] float16  (continuous mode)
    """
    audio = audio_24k.unsqueeze(0).unsqueeze(0).to(device)  # [1, 1, samples]

    # continuous features (encoder → transformer → downsample)
    enc = _mimi_model.encoder(audio)
    enc_t = _mimi_model.encoder_transformer(enc.transpose(1, 2))
    if hasattr(enc_t, "last_hidden_state"):
        enc_t = enc_t.last_hidden_state
    feats = _mimi_model.downsample(enc_t.transpose(1, 2)).squeeze(0).T  # [T, 512]

    # discrete codes (quantizer)
    codes = _mimi_model.encode(audio).audio_codes.squeeze(0).T  # [T, 8]

    return codes.cpu(), feats.cpu().to(torch.float16)


@torch.no_grad()
def _run_inference(codes_ch0, codes_ch1, feats_ch0, feats_ch1, device):
    T = codes_ch0.shape[0]
    eot  = np.zeros((T, 2), dtype=np.float32)
    bot  = np.zeros((T, 2), dtype=np.float32)

    for i in range(0, T, CHUNK_FRAMES):
        j = min(i + CHUNK_FRAMES, T)
        if j - i < 10:
            continue

        ch0 = codes_ch0[i:j].unsqueeze(0).to(device)
        ch1 = codes_ch1[i:j].unsqueeze(0).to(device)
        kwargs = {}
        if _input_mode == "continuous":
            kwargs["mimi_feat_ch0"] = feats_ch0[i:j].unsqueeze(0).to(device)
            kwargs["mimi_feat_ch1"] = feats_ch1[i:j].unsqueeze(0).to(device)

        out = _dt_model(ch0, ch1, mode="inference", **kwargs)
        eot[i:j] = out["eot_probs"].squeeze(0).float().cpu().numpy()
        bot[i:j] = out["bot_probs"].squeeze(0).float().cpu().numpy()

    return eot, bot


def predict_scores(sample_dir: Path) -> dict:
    t0 = time.time()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    _load_models(device)

    def load_wav(path):
        wav, sr = sf.read(path, dtype="float32")
        t = torch.from_numpy(wav)
        if sr != MIMI_SR:
            t = torchaudio.functional.resample(t, sr, MIMI_SR)
        return t  # [samples]

    wav1 = load_wav(sample_dir / "speaker_1_audio.wav")
    wav2 = load_wav(sample_dir / "speaker_2_audio.wav")

    codes1, feats1 = _mimi_encode(wav1, device)
    codes2, feats2 = _mimi_encode(wav2, device)

    T = min(codes1.shape[0], codes2.shape[0])
    codes1, codes2 = codes1[:T], codes2[:T]
    feats1, feats2 = feats1[:T], feats2[:T]

    eot, bot = _run_inference(codes1, codes2, feats1, feats2, device)

    duration = min(len(wav1), len(wav2)) / MIMI_SR
    elapsed  = time.time() - t0
    print(f"{sample_dir.name}: {duration:.1f}s audio → {elapsed:.1f}s ({duration/elapsed:.1f}x RT)", flush=True)

    return {
        "frame_rate_hz":                FRAME_RATE,
        "eot_score_speaker_1":          eot[:, 0].astype(np.float32),
        "eot_score_speaker_2":          eot[:, 1].astype(np.float32),
        "interruption_score_speaker_1": bot[:, 0].astype(np.float32),
        "interruption_score_speaker_2": bot[:, 1].astype(np.float32),
    }


def _inspect(sample_dir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    scores   = predict_scores(sample_dir)
    hz       = scores["frame_rate_hz"]
    eot1     = scores["eot_score_speaker_1"]
    eot2     = scores["eot_score_speaker_2"]
    int1     = scores["interruption_score_speaker_1"]
    int2     = scores["interruption_score_speaker_2"]

    T         = min(len(eot1), len(eot2))
    t         = np.arange(T) / hz
    duration  = t[-1]
    vlines    = np.arange(1, int(duration) + 1)
    tick_locs = np.arange(0, int(duration) + 1, 5)

    arrays = [eot1[:T], eot2[:T], int1[:T], int2[:T]]
    labels = ["EOT spk1", "EOT spk2", "INT spk1", "INT spk2"]
    colors = ["#2196F3", "#F44336", "#4CAF50", "#FF9800"]

    fig, axes = plt.subplots(6, 1, figsize=(max(duration / 10, 20), 10),
                             gridspec_kw={"hspace": 0.05})

    for spk_idx, ax in enumerate(axes[:2]):
        p = sample_dir / f"speaker_{spk_idx + 1}_audio.wav"
        wav, sr = sf.read(p, dtype="float32")
        ax.plot(np.arange(len(wav)) / sr, wav, lw=0.3, color="#555", alpha=0.7)
        peak = float(np.abs(wav).max()) or 1.0
        ax.set_ylim(-peak * 1.1, peak * 1.1)
        for vl in vlines: ax.axvline(vl, color="#ccc", lw=0.4, zorder=0)
        ax.set_ylabel(f"Spk {spk_idx + 1}\nwav", fontsize=7)
        ax.set_xlim(0, duration); ax.set_xticks(tick_locs); ax.tick_params(labelbottom=False)

    for i, (arr, label, color, ax) in enumerate(zip(arrays, labels, colors, axes[2:])):
        ax.plot(t, arr, lw=0.6, color=color)
        ax.fill_between(t, arr, alpha=0.15, color=color)
        ax.set_ylim(-0.05, 1.05)
        for vl in vlines: ax.axvline(vl, color="#ccc", lw=0.4, zorder=0)
        ax.set_ylabel(label, fontsize=7)
        ax.set_xlim(0, duration); ax.set_xticks(tick_locs)
        if i < 3: ax.tick_params(labelbottom=False)

    axes[-1].set_xlabel("Time (s)", fontsize=9)
    fig.suptitle(f"dualturn  task={sample_dir.name}  {hz}Hz", fontsize=10)
    out = _HERE / f"inspect_{sample_dir.name}.png"
    fig.savefig(out, dpi=100, bbox_inches="tight")
    print(f"Saved → {out}")
    for label, arr in zip(labels, arrays):
        print(f"  {label:<20s}  min={arr.min():.4f}  max={arr.max():.4f}  mean={arr.mean():.4f}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--split",    default=None)
    parser.add_argument("--run-name", default="dualturn")
    parser.add_argument("--inspect",  default=None, help="Inspect single sample dir and plot")
    args = parser.parse_args()

    if args.inspect:
        _inspect(Path(args.inspect))
    else:
        split_file = Path(args.split) if args.split else None
        sys.exit(run("dualturn", predict_scores, run_name=args.run_name, split_file=split_file))
