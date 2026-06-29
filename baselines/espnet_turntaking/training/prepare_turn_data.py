#!/usr/bin/env python3
"""Build ESPnet SLU data directories (train / valid) for the TURN corpus, so the
ESPnet recipe's training stages (3->11) run unchanged.

Input: a directory of per-conversation folders, each
    <root>/<id>/
        combined_audio.wav            # stereo: ch0=spk1, ch1=spk2
        speaker_1_annotation_a.srt    # SRT with "[Fine Label] text" cues
        speaker_2_annotation_a.srt
(this is the layout of `otoearth/otoSpeech-full-duplex-turn-104h` on the
HuggingFace Hub — see README for the download command).

Pipeline, per conversation:
  1. SRT cues -> canonical labels -> per-speaker per-40 ms activity:
       BC  if a Backchannel span is active,
       IPU if a Turn or Interruption span is active,
       NA  otherwise (Laughter / AwkwardSilence / NonContent / gaps).
  2. Two-channel {IPU/BC/NA} -> mono 5-class event {C, NA, I, BC, T} + floor
     speaker, via the Switchboard recipe's exact mono-collapse.
  3. Class-balance subsample (keep all T; subsample others ~ proportional to T).
  4. Write data/<set>/{wav.scp, segments, text, utt2spk}:
       wav.scp : "<id> sox <combined.wav> -r 16000 -t wav - remix 1,2 |"
                 remix 1,2 = mean downmix of the two channels (== the SWBD
                 recipe's `sox ... channels 1`, and the same mono mix the
                 espnet_turntaking inference reconstructs as ch1+ch2).
       segments: a 30 s window ending at each labeled chunk (use_only_last_correct).
       text    : "<seg_id> <label>"  (single 5-class token per segment).

Output is the standard Kaldi/ESPnet data-dir format; run the recipe's
`utils/fix_data_dir.sh` afterwards (the README shows the full flow).

Self-contained: no ESPnet / recipe imports, no hardcoded paths.

    python prepare_turn_data.py --root <conv_root> \
        --out-train data/turn_train --out-valid data/turn_valid \
        --token-list data/en_token_list/word/tokens.txt
"""
from __future__ import annotations

import argparse
import glob
import os
import random
import re
import sys
from collections import Counter

CHUNK = 0.04
MIN_START = 0.2
SEG_WINDOW = 30.0

# 5-class token list (the class->id mapping the model is trained with; MUST match
# the published checkpoint's config.yaml token_list).
TOKEN_LIST = ["<blank>", "<unk>", "C", "NA", "I", "BC", "T", "<sos/eos>"]

# Fine SRT label -> canonical bucket, then bucket -> per-speaker activity role.
ACTIVE_CANON = {"Turn", "Interruption"}     # floor-relevant speech -> IPU
BC_CANON = {"Backchannel"}                  # listener cue -> BC
# Laughter / AwkwardSilence / NonContent -> NA (not floor-holding)
FINE_TO_CANON = {
    "Normal Turn": "Turn", "Regular Turn": "Turn", "Strong Floor Hold": "Turn",
    "Bounded Response": "Turn", "Filler": "Turn", "Overlap": "Turn",
    "Laughter": "Laughter",
    "Awkward Silence": "AwkwardSilence",
    "Floor-taking Competitive Interruption": "Interruption",
    "Floor-taking Cooperative Interruption": "Interruption",
    "Non-floor Taking Competitive Interruption": "Interruption",
    "Non-floor Taking Cooperative Interruption": "Interruption",
    "Acknowledgement Backchannel": "Backchannel",
    "Continuer Backchannel": "Backchannel",
    "Reaction Backchannel": "Backchannel",
    "Non-Speech Noise": "NonContent", "Channel Bleed": "NonContent",
    "Speech, Non-Linguistic": "NonContent",
}

_TS = re.compile(r"(\d\d):(\d\d):(\d\d),(\d\d\d)\s*-->\s*(\d\d):(\d\d):(\d\d),(\d\d\d)")
_LABEL = re.compile(r"^\[([^\]]+)\]")


# --------------------------------------------------------------------------- #
# SRT -> per-40 ms two-channel labels -> mono 5-class collapse                 #
# --------------------------------------------------------------------------- #
def _secs(h, m, s, ms):
    return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000.0


def parse_srt(path):
    """-> list of (start_s, end_s, canonical_label) in file order."""
    out = []
    if not os.path.exists(path):
        return out
    block = []
    for ln in open(path, encoding="utf-8", errors="replace").read().splitlines() + [""]:
        if ln.strip() == "":
            if block:
                start = end = fine = None
                for b in block:
                    mt = _TS.search(b)
                    if mt:
                        start = _secs(*mt.group(1, 2, 3, 4))
                        end = _secs(*mt.group(5, 6, 7, 8))
                        continue
                    ml = _LABEL.match(b.strip())
                    if ml and fine is None:
                        fine = ml.group(1).strip()
                if start is not None and fine is not None:
                    out.append((start, end, FINE_TO_CANON.get(fine, "NonContent")))
                block = []
        else:
            block.append(ln)
    return out


def _spans(segments, canon_set):
    return sorted((s, e) for s, e, c in segments if c in canon_set and e > s)


def _active(cs, ce, spans, idx):
    """Stateful sweep: chunk [cs,ce] is active if a span overlaps it (one-chunk
    lookahead). Returns (bool, idx)."""
    n = len(spans)
    while idx < n and spans[idx][1] <= cs:
        idx += 1
    if idx < n and spans[idx][0] <= ce + CHUNK and spans[idx][1] > cs:
        return True, idx
    return False, idx


def chunk_labels(seg_a, seg_b, end_time):
    bcA, bcB = _spans(seg_a, BC_CANON), _spans(seg_b, BC_CANON)
    ipA, ipB = _spans(seg_a, ACTIVE_CANON), _spans(seg_b, ACTIVE_CANON)
    n = int((end_time - MIN_START) / CHUNK)
    ia = ib = ja = jb = 0
    rows = []
    for i in range(n):
        cs = MIN_START + i * CHUNK
        ce = cs + CHUNK
        a_bc, ia = _active(cs, ce, bcA, ia)
        a_ip, ja = _active(cs, ce, ipA, ja)
        b_bc, ib = _active(cs, ce, bcB, ib)
        b_ip, jb = _active(cs, ce, ipB, jb)
        lA = "BC" if a_bc else ("IPU" if a_ip else "NA")
        lB = "BC" if b_bc else ("IPU" if b_ip else "NA")
        rows.append((round(cs, 2), round(ce, 2), lA, lB))
    return rows


def collapse(rows):
    """Two-channel {IPU/BC/NA} -> mono event {C,NA,I,BC,T,BC_1,BC_2} + floor."""
    events, prev_arr, prev = [], [], "NA"
    for _, _, a, b in rows:
        prev_arr.append(prev)
        if a == "NA" and b == "NA":
            events.append("NA")
        elif a == "IPU" and b == "NA":
            if prev == "A":
                events.append("C")
            else:
                events.append("C" if prev == "AB" else "T")
                prev = "A"
        elif a == "NA" and b == "IPU":
            if prev == "B":
                events.append("C")
            else:
                events.append("C" if prev == "BA" else "T")
                prev = "B"
        elif a == "IPU" and b == "IPU":
            events.append("I")
            if prev not in ("AB", "BA"):
                prev = "AB" if prev == "A" else "BA"
        elif a == "BC" and b == "BC":
            events.append("BC_2")
        elif a == "BC":
            events.append("BC" if prev in ("B", "BA") else "BC_1")
        elif b == "BC":
            events.append("BC" if prev in ("A", "AB") else "BC_1")
        else:
            events.append("NA")
    return events, prev_arr


def conv_rows(conv_dir, cid):
    """-> list of [file_id, chunk_start, chunk_end, label, prev_speaker] (str)."""
    sa = parse_srt(os.path.join(conv_dir, "speaker_1_annotation_a.srt"))
    sb = parse_srt(os.path.join(conv_dir, "speaker_2_annotation_a.srt"))
    ends = [e for _, e, _ in sa] + [e for _, e, _ in sb]
    rows = chunk_labels(sa, sb, max(ends) if ends else 0.0)
    events, prev_arr = collapse(rows)
    return [[cid, f"{cs}", f"{ce}", ev, pv]
            for (cs, ce, _a, _b), ev, pv in zip(rows, events, prev_arr)]


# --------------------------------------------------------------------------- #
# class-balance subsample (keep all T; subsample others ~ proportional to T)   #
# --------------------------------------------------------------------------- #
def subsample(all_rows, seed=0):
    rng = random.Random(seed)
    turn_dict, groups, kept = {}, {}, []
    prev, cur_key = None, None
    for r in all_rows:
        lab = r[-2]
        if lab in ("BC_1", "BC_2"):
            continue
        if lab == "BC" and r[-1] not in ("A", "B"):
            continue
        turn_dict.setdefault(lab, [])
        groups.setdefault(lab, {})
        key = r[0] + "_" + r[1]
        if prev is None or prev[0] != r[0] or lab != prev[-2]:
            groups[lab][key] = []
            cur_key = key
        if lab == "T":
            kept.append(r)
        turn_dict[lab].append(r)
        groups[lab][cur_key].append(r)
        prev = r
    if not turn_dict.get("T"):
        raise RuntimeError("no T (turn-change) events found; cannot set subsample ratio")
    count_t = len(turn_dict["T"])
    for k in groups:
        if k == "T":
            continue
        ratio = len(turn_dict[k]) / count_t
        for grp in groups[k].values():
            a1 = max(1, len(grp) / ratio + 1)
            if k in ("BC", "I"):
                kept.append(grp[0])
                if int(a1) > 1:
                    kept.extend(rng.sample(grp[1:], int(a1) - 1))
            else:
                kept.extend(rng.sample(grp, int(a1)))
    return kept


# --------------------------------------------------------------------------- #
# ESPnet data dir                                                             #
# --------------------------------------------------------------------------- #
def _hexid(t):
    return "{:.6f}".format(float(t) / 10**4).split(".")[-1]


def _rid(cid):
    """Fixed-width 'turnNNNN' recording/speaker id — fixed width + non-numeric
    prefix avoids Kaldi's underscore-vs-digit collation bug."""
    return f"turn{int(cid):04d}" if str(cid).isdigit() else f"turn_{cid}"


def write_data_dir(out_dir, rows, audio_path_of):
    os.makedirs(out_dir, exist_ok=True)
    convs = sorted({r[0] for r in rows}, key=_rid)
    with open(os.path.join(out_dir, "wav.scp"), "w") as f:
        for cid in convs:
            f.write(f"{_rid(cid)} sox \"{audio_path_of(cid)}\" -r 16000 -t wav - "
                    f"remix 1,2 |\n")
    seg = open(os.path.join(out_dir, "segments"), "w")
    txt = open(os.path.join(out_dir, "text"), "w")
    u2s = open(os.path.join(out_dir, "utt2spk"), "w")
    by_conv = {}
    for r in rows:
        by_conv.setdefault(r[0], []).append(r)
    n = 0
    for cid in convs:
        for r in sorted(by_conv[cid], key=lambda x: float(x[1])):
            cs, ce, lab = r[1], r[2], r[3]
            end = float(ce)
            start = 0.0 if end < SEG_WINDOW else end - SEG_WINDOW
            seg_id = f"{_rid(cid)}_{_hexid(cs)}_{_hexid(ce)}"
            seg.write(f"{seg_id} {_rid(cid)} {start:.2f} {end:.2f}\n")
            txt.write(f"{seg_id} {lab}\n")
            u2s.write(f"{seg_id} {_rid(cid)}\n")
            n += 1
    seg.close(); txt.close(); u2s.close()
    return len(convs), n


def write_token_list(path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        f.write("\n".join(TOKEN_LIST) + "\n")


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True,
                    help="directory of per-conversation folders (<id>/combined_audio.wav + *.srt)")
    ap.add_argument("--out-train", default="data/turn_train")
    ap.add_argument("--out-valid", default="data/turn_valid")
    ap.add_argument("--token-list", default=None,
                    help="also write the fixed 5-class token list here")
    ap.add_argument("--valid-every", type=int, default=14,
                    help="every Nth conv id (sorted) goes to valid (default 14)")
    ap.add_argument("--ids", help="optional comma-separated subset (for a quick dry run)")
    args = ap.parse_args()

    conv_dirs = {
        os.path.basename(d): d
        for d in glob.glob(os.path.join(args.root, "*"))
        if os.path.isdir(d) and os.path.exists(os.path.join(d, "combined_audio.wav"))
    }
    if not conv_dirs:
        ap.error(f"no <id>/combined_audio.wav conversation folders under {args.root}")
    ids = sorted(conv_dirs, key=lambda c: int(c) if c.isdigit() else c)
    if args.ids:
        want = set(args.ids.split(","))
        ids = [i for i in ids if i in want]
    valid_ids = set(ids[:: args.valid_every]) if args.valid_every > 0 else set()
    train_ids = [i for i in ids if i not in valid_ids]
    print(f"TURN convs: {len(ids)} total -> {len(train_ids)} train / "
          f"{len(valid_ids)} valid", file=sys.stderr)

    def build(split_ids, out_dir):
        if not split_ids:
            return
        rows = []
        for cid in split_ids:
            rows.extend(conv_rows(conv_dirs[cid], cid))
        kept = subsample(rows)
        nc, ns = write_data_dir(
            out_dir, kept,
            lambda c: os.path.join(conv_dirs[c], "combined_audio.wav"))
        print(f"{out_dir}: {nc} convs, {ns} segments, "
              f"labels={dict(Counter(r[3] for r in kept))}", file=sys.stderr)

    build(train_ids, args.out_train)
    build(sorted(valid_ids, key=lambda c: int(c) if c.isdigit() else c), args.out_valid)
    if args.token_list:
        write_token_list(args.token_list)
        print(f"wrote token list -> {args.token_list}", file=sys.stderr)


if __name__ == "__main__":
    main()
