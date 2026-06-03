import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download
from moshi.models import loaders as moshi_loaders

# Config (from ms_lstm_mimi-25hz-nq8_delay1f training run)
MODEL_DEFAULTS = {
    "input_size": 512,    # Mimi feat_size
    "hidden_size": 512,
    "num_layers": 2,
    "output_size": 5,
    "dropout": 0.1,
    "bidirectional": False,
    "project": 128,       # projects 512 → 128 before LSTMs
    "num_project": 1,
}


AUDIO_DEFAULTS = {
    "sr": 24000,           # Mimi native sample rate
    "num_quantizers": 8,
    "frame_rate_hz": 25.0, # after ×2 upsample from Mimi's native 12.5Hz
}


class MimiLSTM(nn.Module):
    """Two-stream LSTM over Mimi embeddings.

    Input x: (batch, 2, feat_size, T) — speaker_1 at index 0, speaker_2 at index 1.
    Output:   (batch, T, output_size)
    """
    def __init__(self, input_size, hidden_size, num_layers, output_size,
                 dropout, bidirectional, project, **_):
        super().__init__()
        self.mel_embed = nn.Sequential(
            nn.Linear(input_size, project),
            nn.ReLU(),
        )
        lstm_kwargs = dict(
            input_size=project,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=bidirectional,
        )
        self.model1 = nn.LSTM(**lstm_kwargs)
        self.model2 = nn.LSTM(**lstm_kwargs)
        self.linear = nn.Linear(hidden_size * 2, output_size)

    def init_hidden(self, batch_size, device):
        h = torch.zeros(self.model1.num_layers, batch_size, self.model1.hidden_size).to(device)
        c = torch.zeros(self.model1.num_layers, batch_size, self.model1.hidden_size).to(device)
        return h, c

    def infer(self, x):
        """Full-sequence inference. x: (1, 2, feat, T)."""
        x = x.permute(0, 1, 3, 2)          # (1, 2, T, feat)
        x = self.mel_embed(x)               # (1, 2, T, project)
        h, c = self.init_hidden(x.size(0), x.device)
        x1, _ = self.model1(x[:, 0], (h, c))
        x2, _ = self.model2(x[:, 1], (h, c))
        return self.linear(torch.cat([x1, x2], dim=-1))  # (1, T, output_size)

    def infer_ar_step(self, feat1, feat2, h1, c1, h2, c2):
        """Single-frame AR step.

        feat1, feat2: (1, feat_size) — one frame per speaker.
        Returns: logits (1, output_size), updated (h1, c1, h2, c2).
        """
        f1 = self.mel_embed(feat1).unsqueeze(1)   # (1, 1, project)
        f2 = self.mel_embed(feat2).unsqueeze(1)
        out1, (h1, c1) = self.model1(f1, (h1, c1))  # (1, 1, hidden)
        out2, (h2, c2) = self.model2(f2, (h2, c2))
        logits = self.linear(torch.cat([out1, out2], dim=-1)).squeeze(1)  # (1, output_size)
        return logits, h1, c1, h2, c2

    def infer_ar(self, x):
        """Frame-by-frame AR inference. x: (1, 2, feat, T).
        Equivalent to infer() for a unidirectional LSTM but simulates real-time decoding.
        """
        x = x.permute(0, 1, 3, 2)          # (1, 2, T, feat)
        x = self.mel_embed(x)               # (1, 2, T, project)
        T = x.size(2)
        h1, c1 = self.init_hidden(x.size(0), x.device)
        h2, c2 = self.init_hidden(x.size(0), x.device)
        outputs = []
        for t in range(T):
            f1 = x[:, 0, t:t+1, :]         # (1, 1, project)
            f2 = x[:, 1, t:t+1, :]
            out1, (h1, c1) = self.model1(f1, (h1, c1))
            out2, (h2, c2) = self.model2(f2, (h2, c2))
            outputs.append(self.linear(torch.cat([out1, out2], dim=-1)))
        return torch.cat(outputs, dim=1)    # (1, T, output_size)


class AudioFeatureExtractor:
    """Extracts Mimi embeddings from raw waveform using the moshi library."""

    def __init__(self, sr, num_quantizers, device="cuda", **_):
        mimi_weight = hf_hub_download(moshi_loaders.DEFAULT_REPO, moshi_loaders.MIMI_NAME)
        self.mimi = moshi_loaders.get_mimi(mimi_weight, device=device, num_codebooks=num_quantizers)
        self.mimi.eval()
        self.sr = sr
        self.device = device

    def _to_tensor(self, wav):
        """1D numpy array → (1, 1, T) tensor on device."""
        return torch.from_numpy(wav).float().to(self.device).unsqueeze(0).unsqueeze(0)

    @torch.no_grad()
    def __call__(self, wav):
        """Full-sequence. wav: 1D numpy array. Returns (1, feat_size, T) at 25Hz."""
        x = self._to_tensor(wav)
        emb = self.mimi.encode_to_latent(x, quantize=True)  # (1, feat, T) at 12.5Hz
        return self.mimi.upsample(emb)                       # (1, feat, T) at 25Hz

    @torch.no_grad()
    def stream(self, wav1, wav2):
        """Streaming both speakers batched together, one Mimi frame (1920 samples) at a time.
        Yields (1, feat_size, frames) for speaker1 and speaker2 per chunk.
        Single streaming context so KV cache is shared and there's no double-enter conflict.
        """
        chunk_samples = self.mimi.frame_size  # 1920

        def pad(wav):
            t = torch.from_numpy(wav).float().to(self.device)
            r = len(t) % chunk_samples
            return torch.nn.functional.pad(t, (0, chunk_samples - r if r else 0))

        w1, w2 = pad(wav1), pad(wav2)
        n_chunks = len(w1) // chunk_samples

        with self.mimi.streaming(batch_size=2):
            for i in range(n_chunks):
                s = i * chunk_samples
                chunk = torch.stack([w1[s:s+chunk_samples], w2[s:s+chunk_samples]]).unsqueeze(1)  # (2, 1, 1920)
                emb = self.mimi.encode_to_latent(chunk, quantize=True)  # (2, feat, 1)
                emb = self.mimi.upsample(emb)                            # (2, feat, 2)
                yield emb[0:1], emb[1:2]                                 # (1, feat, 2) each


def load_model(checkpoint_path, device="cpu"):
    model = MimiLSTM(**MODEL_DEFAULTS)
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model.to(device)


CLASS_NAMES = ["silence", "system", "system_end", "user", "user_end"]


if __name__ == "__main__":
    import argparse
    import numpy as np
    import soundfile as sf
    import matplotlib.pyplot as plt

    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--sample_dir", required=True)
    parser.add_argument("--ar", action="store_true", help="Streaming Mimi (chunk-by-chunk) + AR LSTM step-by-step.")
    parser.add_argument("--out", default="eval_output.png")
    args = parser.parse_args()

    import torchaudio
    import numpy as np
    import soundfile as sf
    import matplotlib.pyplot as plt
    from pathlib import Path

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.checkpoint:
        model = load_model(args.checkpoint, device=device)
        print(f"Loaded: {args.checkpoint}")
    else:
        model = MimiLSTM(**MODEL_DEFAULTS).to(device).eval()
        print("No checkpoint — using random weights.")

    extractor = AudioFeatureExtractor(**AUDIO_DEFAULTS, device=device)
    target_sr = AUDIO_DEFAULTS["sr"]
    d = Path(args.sample_dir)

    wav1, sr1 = sf.read(d / "speaker_1_audio.wav", dtype="float32")
    wav2, sr2 = sf.read(d / "speaker_2_audio.wav", dtype="float32")
    if sr1 != target_sr:
        wav1 = torchaudio.functional.resample(torch.from_numpy(wav1), sr1, target_sr).numpy()
    if sr2 != target_sr:
        wav2 = torchaudio.functional.resample(torch.from_numpy(wav2), sr2, target_sr).numpy()

    if args.ar:
        # streaming Mimi chunk-by-chunk + AR LSTM step-by-step
        print("Running streaming Mimi + AR LSTM...")
        from tqdm import tqdm
        h1, c1 = model.init_hidden(1, device)
        h2, c2 = model.init_hidden(1, device)
        all_logits = []
        n_chunks = (len(wav1) + 1919) // 1920
        with torch.no_grad():
            for feat1_chunk, feat2_chunk in tqdm(
                extractor.stream(wav1, wav2),
                total=n_chunks, unit="chunk", desc="Streaming"
            ):
                feat1_chunk, feat2_chunk = feat1_chunk.to(device), feat2_chunk.to(device)
                for t in range(feat1_chunk.shape[-1]):
                    logits, h1, c1, h2, c2 = model.infer_ar_step(
                        feat1_chunk[:, :, t], feat2_chunk[:, :, t], h1, c1, h2, c2
                    )
                    all_logits.append(logits)
        out = torch.stack(all_logits, dim=1)  # (1, T, output_size)
    else:
        feat1 = extractor(wav1)  # (1, feat, T)
        feat2 = extractor(wav2)
        T = min(feat1.shape[-1], feat2.shape[-1])
        x = torch.cat([feat1[:, :, :T], feat2[:, :, :T]], dim=0).unsqueeze(0)  # (1, 2, feat, T)
        with torch.no_grad():
            out = model.infer(x.to(device))

    T = out.shape[1]
    print(f"Audio: {len(wav1)/target_sr:.1f}s, {T} frames @ {AUDIO_DEFAULTS['frame_rate_hz']} Hz")
    probs = torch.softmax(out[0], dim=-1).cpu().numpy()   # (T, 5)

    frame_times = np.arange(T) / AUDIO_DEFAULTS["frame_rate_hz"]
    wav_times = np.arange(len(wav1)) / target_sr

    duration = len(wav1) / target_sr
    fig_width = max(28, int(duration * 0.2))  # ~0.2 inches per second
    fig, (ax_wav, ax_pred) = plt.subplots(
        2, 1, figsize=(fig_width, 6),
        gridspec_kw={"hspace": 0.08, "height_ratios": [1, 1]},
    )

    ax_wav.plot(wav_times, wav1, linewidth=0.3, color="steelblue", alpha=0.7, label="Speaker 1")
    ax_wav.plot(wav_times, wav2, linewidth=0.3, color="darkorange", alpha=0.7, label="Speaker 2")
    ax_wav.set_ylabel("Amplitude")
    ax_wav.set_xlim(wav_times[0], wav_times[-1])
    ax_wav.legend(loc="upper right", fontsize=8)
    ax_wav.set_xticklabels([])

    for i, name in enumerate(CLASS_NAMES):
        ax_pred.plot(frame_times, probs[:, i], label=name, linewidth=1.0)
    ax_pred.set_ylabel("Softmax probability")
    ax_pred.set_xlabel("Time (s)")
    ax_pred.set_xlim(frame_times[0], frame_times[-1])
    ax_pred.set_ylim(0, 1)
    ax_pred.legend(loc="upper right", fontsize=8)

    plt.savefig(args.out, dpi=150, bbox_inches="tight")
    print(f"Saved: {args.out}")
