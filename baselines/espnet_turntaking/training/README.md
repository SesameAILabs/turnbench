# Training the `espnet_turntaking` model

This folder reproduces the turn-taking model that the
[`espnet_turntaking`](../) and
[`espnet_turntaking_perchannel`](../../espnet_turntaking_perchannel) baselines
wrap — a frozen Whisper-medium encoder (~306 M) with a small 5-class head
(*"Talking Turns"*, Arora et al., ICLR 2025), emitting per-40 ms (25 Hz)
probabilities over {Continuation(C), Silence(NA), Interruption(I),
Backchannel(BC), Turn-change(T)}.

The training **engine is ESPnet** (the same `espnet2` the inference baseline
already imports to load the checkpoint). This folder holds only the **portable,
corpus-specific layer** — data preparation, the training config, the fixed token
list, and a driver — and **pins ESPnet as a dependency** rather than vendoring
it.

> ### This is the MIX setup (single-stream, two-speaker mono mix)
>
> The model is **single-stream**, trained on the **two-speaker mono mix** — the
> data prep builds `wav.scp` as `sox … remix 1,2` (the mean downmix of the two
> channels, i.e. the same `ch1 + ch2` signal the inference baseline
> reconstructs). It is **not** trained per-channel.
>
> There is exactly **one** model, and **both** baselines wrap it:
> `espnet_turntaking` runs it on the mix; `espnet_turntaking_perchannel` runs the
> *same mix-trained* model on each isolated channel — that is purely an
> **inference-time** choice, with no separate training. Mix training is required
> because the labels are relational (I = both speakers active; T = the floor
> hands to the *other* speaker), so they can only be learned from a signal that
> contains both speakers.

## 1. Dependency: ESPnet (pinned)

Training runs `python -m espnet2.bin.slu_train` through ESPnet's shared
`egs2/TEMPLATE/slu1/slu.sh` pipeline. Install ESPnet at the pinned revision:

```bash
git clone https://github.com/espnet/espnet
cd espnet && git checkout 750e3749fc37a09187fe0fc6fb278ccb007181e8   # version 202604
cd tools && make            # sets up the Python env + torch + espnet (editable)
```

This is the **same ESPnet** the `espnet_turntaking` baseline needs at inference
time (`from espnet2.tasks.slu import SLUTask`), so it is not a new dependency —
just pinned for reproducibility. Other deps: see `requirements.txt`.

## 2. Model

- Encoder: **Whisper-medium, frozen** (`freeze_param: [encoder]`), 1024-dim.
- Head: `Tanh → Linear(1024→1024) → Linear(1024→5)` (`superb_setup: true`).
- Target: the **single 5-class label of the chunk a 30 s window ends on**
  (`use_only_last_correct: true`), trained with cross-entropy + label smoothing.
- Token list (`tokens.txt`): `<blank> <unk> C NA I BC T <sos/eos>` — the
  class→id mapping **must** match the published checkpoint; do not reorder.

Full hyper-parameters are in `conf/train_turn_taking.yaml`.

## 3. Data preparation (TURN, in-distribution)

The TURN training corpus is
[`otoearth/otoSpeech-full-duplex-turn-104h`](https://huggingface.co/datasets/otoearth/otoSpeech-full-duplex-turn-104h)
(gated — request access, then `huggingface-cli login`). Each conversation is a
folder `<id>/` with `combined_audio.wav` (stereo: ch0=spk1, ch1=spk2) and
`speaker_{1,2}_annotation_a.srt`.

```bash
# download the audio + SRT annotations
python -c "from huggingface_hub import snapshot_download; \
  snapshot_download('otoearth/otoSpeech-full-duplex-turn-104h', repo_type='dataset', \
    allow_patterns=['*/combined_audio.wav','*/speaker_*_annotation_a.srt','*/metadata.json'])"

# SRT -> per-40 ms 5-class labels -> ESPnet data dirs + token list
python prepare_turn_data.py \
    --root  <snapshot_dir> \
    --out-train data/turn_train \
    --out-valid data/turn_valid \
    --token-list data/en_token_list/word/tokens.txt
```

What `prepare_turn_data.py` does (self-contained, no ESPnet imports):
- Maps SRT fine labels → canonical (Turn/Interruption→floor "IPU",
  Backchannel→BC, Laughter/AwkwardSilence/NonContent→NA).
- Collapses the two channels to the mono 5-class event stream {C,NA,I,BC,T} with
  the Switchboard recipe's exact rule (I = both speakers active; T = floor hands
  to the other; C = same speaker continues).
- Class-balances (keeps all T, subsamples the rest ∝ T).
- Writes `wav.scp` with `sox … remix 1,2` — the **mean downmix** of the two
  channels, which is the same mono mix the inference baseline reconstructs as
  `ch1 + ch2`, so **train and eval see the same signal**.

> **Data hygiene:** the TURN-104h training corpus is *speaker-disjoint* from the
> benchmark dev/test sets (verified: 0 shared actor ids), so it is not leakage —
> but never add benchmark dev/test conversations to the training pool.

Then make the data dirs ESPnet-valid (inside your recipe dir):

```bash
utils/fix_data_dir.sh data/turn_train
utils/fix_data_dir.sh data/turn_valid
```

## 4. Train

Point the driver at an ESPnet egs2 SLU recipe (e.g. `egs2/swbd/slu1`, which has
the `slu.sh`/`utils`/`steps` symlinks) whose `data/` holds the dirs from step 3:

```bash
RECIPE_DIR=/path/to/espnet/egs2/swbd/slu1 \
NGPU=2 EXP=exp/tt_turn STATS=exp/slu_stats_turn \
  bash run_training.sh
```

This runs `slu.sh` stages **3 → 11** (format/dump → length filter → collect-stats
→ train); stage 5 is skipped because we supply the fixed `tokens.txt`. Notes:
- **Memory:** `batch_size 1500` fits at `NGPU=2` on a 95 GB GPU; use
  `NGPU=1` with `batch_size 750` (edit the config) for a single GPU. The
  original published model used 8 GPUs at `batch_size 4000`.
- **`valid_batch_size: 64`** keeps validation from OOMing (it otherwise groups
  long segments into one batch).
- **Resume:** ESPnet trains with `--resume true`; if a job times out, just
  re-run `run_training.sh` — it continues from the last checkpoint.
- **Runtime:** ~50 min/epoch × 32 epochs on a GH200-class GPU.
- **Output:** `${RECIPE_DIR}/${EXP}/valid.loss.ave.pth` (+ `config.yaml`).

## 5. Use the trained checkpoint

```bash
export ESPNET_TT_EXP=/path/to/espnet/egs2/swbd/slu1/exp/tt_turn
python -m baselines.espnet_turntaking.predict \
    --out baselines/espnet_turntaking/predictions-dev.json
```

The baseline's `predict.py` loads `${ESPNET_TT_EXP}/{config.yaml,
valid.loss.ave.pth}` via `SLUTask` — closing the train → eval loop.

## 6. Switchboard / mixed (out-of-distribution) — documented, not packaged

The "trained-on-Switchboard" (OOD) and "TURN+SWBD" (mixed) variants need the
**licensed** Switchboard corpus, which cannot be redistributed here:

- **LDC97S62** (Switchboard-1 audio) + **MS-State** word alignments + a
  backchannel CSV.
- The ESPnet `egs2/swbd/slu1` recipe's
  `local/create_switchboard_data_2channels{,_mono}.py` +
  `subsample_2channel_switchboard_mono.py` produce the SWBD data dir the same
  way (those scripts live in the ESPnet repo — Apache-2.0).
- The published OOD checkpoint is on the Hub:
  `espnet/Turn_taking_prediction_SWBD`.
- The **mixed** model = pool the TURN and SWBD data dirs with
  `utils/combine_data.sh data/mix_train data/turn_train data/train`, then train
  with the same config/driver (`TRAIN_SET=mix_train`).

## Files

- `prepare_turn_data.py` — SRT → ESPnet data dirs (self-contained).
- `conf/train_turn_taking.yaml` — training config.
- `tokens.txt` — fixed 5-class token list.
- `run_training.sh` — portable ESPnet training driver.
- `requirements.txt` — training-side deps.
