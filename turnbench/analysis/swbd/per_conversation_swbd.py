#!/usr/bin/env python3
"""Per-conversation turn-taking statistics for Switchboard-1 (LDC97S62).

Vendored from cyhuang-tw/swbd-statistics; this copy is canonical.

Produces the same per-conversation metric schema as
`turnbench/analysis/per_conversation.py`, but computed from Switchboard instead of
the benchmark's 3-annotator tracks, so SWB-1 can be added as a single
comparison row to `stats_out/per_type_aggregate.csv` — plus the full
FTO / pause timing distributions (swbd_timing_distributions.json).

Switchboard has no FLOOR-style turn-taking annotation, so every event type is a
*derived analog* built from data SWBD does have:

  * MS-State word/utterance alignments  -> words, voiced/silence, turns, FTO,
                                            overlap, laughter, non-content
  * VAP/Ekstedt backchannels.csv        -> backchannel events  (2436 sessions)
  * Switchboard Dialog Act corpus (SWDA)-> question rate        (1155 sessions)

Multi-annotator IAA columns (iaa_kappa_*, iaa_fleiss_kappa, event_count_cv_abc,
boundary_f1_*) are intentionally omitted: SWBD is single-annotator.

Columns SWBD cannot support are written empty (left for the reader to drop):
  bc_ack/cont/react_per_min, int_competitive/cooperative_per_min
`int_*` and `question_rate_per_min` are APPROXIMATE -- see the per-metric notes
in the docstrings below and in the README the companion writeup points to.

Usage (point --trans-root / --swda-root at your local SWBD + SWDA copies):
  python per_conversation_swbd.py \
      --trans-root /path/to/swb_ms98_transcriptions \
      --backchannels backchannels.csv \
      --swda-root /path/to/swda \
      --out-dir results
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import re
import sys
from collections import defaultdict

import numpy as np

# FLOOR uses this exact word regex; reuse it so word counts/TTR are comparable.
WORD = re.compile(r"[A-Za-z][A-Za-z'\-]*")
NONCONTENT_TOKENS = {"[noise]", "[vocalized-noise]"}

# ----- thresholds (exposed as CLI args, defaults documented in --help) -----
DEF_IPU_GAP = 0.2     # s; merge a speaker's own words into an IPU across <= this gap
DEF_TRP_TOL = 1.0     # s; "interruption" onset must precede the incumbent turn end
#                       by more than this (a barge-in well before the floor was free)


# --------------------------------------------------------------------------- #
# transcript parsing
# --------------------------------------------------------------------------- #
def classify_token(tok: str):
    """Map one MS-State word-token to (kind, word).

    kind in {silence, noise, laughter, word}; `word` is the cleaned lexical
    form (lower-cased, FLOOR-regex-extracted) or None.

      [silence]                 -> ("silence", None)
      [noise] / [vocalized-noise] -> ("noise", None)        (non-content)
      [laughter]                -> ("laughter", None)       (laughter event)
      [laughter-all]            -> ("word", "all")          (word said laughing)
      an[y]- , r[ight]- , word  -> ("word", "<clean>")      (incl. partials)
    """
    t = tok.strip()
    if not t:
        return ("silence", None)
    if t == "[silence]":
        return ("silence", None)
    if t in NONCONTENT_TOKENS:
        return ("noise", None)
    if t == "[laughter]":
        return ("laughter", None)
    if t.startswith("[laughter-"):
        inner = t[len("[laughter-"):].rstrip("]")
        m = WORD.search(inner)
        return ("word", m.group(0).lower() if m else None)
    # ordinary (possibly partial) word: drop bracket chars, extract lexical core
    m = WORD.search(t.replace("[", "").replace("]", ""))
    return ("word", m.group(0).lower() if m else None)


def parse_word_file(path: str):
    """Return list of (start, end, kind, word) for a *-word.text file."""
    out = []
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            p = line.split()
            if len(p) < 4:
                continue
            try:
                s, e = float(p[1]), float(p[2])
            except ValueError:
                continue
            tok = " ".join(p[3:])
            kind, word = classify_token(tok)
            out.append((s, e, kind, word))
    return out


# --------------------------------------------------------------------------- #
# interval helpers
# --------------------------------------------------------------------------- #
def merge_intervals(ivs, gap=0.0):
    """Union of intervals, coalescing neighbours within `gap`."""
    ivs = sorted(ivs)
    if not ivs:
        return []
    out = [list(ivs[0])]
    for s, e in ivs[1:]:
        if s <= out[-1][1] + gap:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e])
    return [(s, e) for s, e in out]


def union_dur(ivs):
    return sum(e - s for s, e in merge_intervals(ivs))


def intersect(a, b):
    """Intersection segments of two interval lists (each already a union)."""
    a, b = merge_intervals(a), merge_intervals(b)
    out, i, j = [], 0, 0
    while i < len(a) and j < len(b):
        lo = max(a[i][0], b[j][0])
        hi = min(a[i][1], b[j][1])
        if lo < hi:
            out.append((lo, hi))
        if a[i][1] < b[j][1]:
            i += 1
        else:
            j += 1
    return out


def in_any(t, ivs):
    return any(s <= t < e for s, e in ivs)


# --------------------------------------------------------------------------- #
# side inputs: backchannels (VAP) and questions (SWDA)
# --------------------------------------------------------------------------- #
def load_backchannels(path):
    """sess(str) -> {'A':[(s,e),...], 'B':[...]} from VAP/Ekstedt backchannels.csv."""
    bc = defaultdict(lambda: {"A": [], "B": []})
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            try:
                s, e = float(row["start"]), float(row["end"])
            except (ValueError, KeyError):
                continue
            sess = str(row["session"]).strip()
            spk = row["speaker"].strip()
            if spk in ("A", "B"):
                bc[sess][spk].append((s, e))
    return bc


def load_swda_questions(swda_root):
    """sess(str) -> n_question_utterances, from SWDA act_tags.

    DAMSL question tags: qy (yes-no), qw (wh), qo (open), qh (rhetorical),
    qrr (or-clause) and their ^d/^t/... variants, plus ^g (tag-question).
    The *primary* dialog act is the part before the first '^' modifier, so
    'qy^d' -> 'qy' counts, but '^q' (Quotation) and '^h' (Hold) do NOT.
    """
    if not swda_root or not os.path.isdir(swda_root):
        return {}
    q = defaultdict(int)
    seen = set()
    for path in glob.glob(os.path.join(swda_root, "*", "*.utt.csv")):
        with open(path, newline="", encoding="utf-8", errors="replace") as f:
            for row in csv.DictReader(f):
                sess = str(row.get("conversation_no", "")).strip()
                if not sess:
                    continue
                seen.add(sess)
                tag = (row.get("act_tag") or "").strip().lower()
                comps = [c for c in re.split(r"[ ;,]+", tag) if c]
                is_q = any(c.split("^", 1)[0].startswith("q") for c in comps) \
                    or "^g" in tag
                if is_q:
                    q[sess] += 1
    # ensure every SWDA session is a key (0 questions still means "covered")
    return {s: q.get(s, 0) for s in seen}


# --------------------------------------------------------------------------- #
# per-conversation analysis
# --------------------------------------------------------------------------- #
def build_turns(ipus):
    """ipus: list of (start, end, speaker). Merge consecutive same-speaker IPUs
    (sorted by onset) into floor-holding turns; the floor changes only when the
    *other* speaker produces a (non-backchannel) IPU. Returns (turns, pauses):
    turns as (start, end, speaker), pauses as the positive within-turn silences
    each same-speaker merge bridges (the speaker resumes, floor unchanged).
    Pauses have a floor of --ipu-gap: shorter within-speaker silences were
    already absorbed when the words were merged into IPUs."""
    ipus = sorted(ipus)
    turns = []
    pauses = []
    for s, e, sp in ipus:
        if turns and turns[-1][2] == sp:
            if s - turns[-1][1] > 0:
                pauses.append(s - turns[-1][1])
            turns[-1] = (turns[-1][0], max(turns[-1][1], e), sp)
        else:
            turns.append((s, e, sp))
    return turns, pauses


def analyze(sess, words_a, words_b, bc, q_count, ipu_gap, trp_tol):
    dur_s = max([e for _, e, _, _ in words_a] + [e for _, e, _, _ in words_b] + [0.0])
    if dur_s <= 0:
        return None
    dur_min = dur_s / 60.0

    bc_a, bc_b = bc.get("A", []), bc.get("B", [])
    bc_a, bc_b = merge_intervals(bc_a), merge_intervals(bc_b)

    def sided(words, bc_iv):
        # categorise this speaker's word tokens
        voiced = [(s, e) for s, e, k, _ in words if k in ("word", "laughter")]
        activity = [(s, e) for s, e, k, _ in words if k != "silence"]
        noncontent = sum(e - s for s, e, k, _ in words if k == "noise")
        laughter = sum(1 for *_x, k, _ in [(s, e, k, w) for s, e, k, w in words]
                       if k == "laughter")
        lex = [w for _s, _e, k, w in words if k == "word" and w]
        # "floor" words = lexical words that are NOT inside a backchannel span
        floor_words = [(s, e) for s, e, k, w in words
                       if k == "word" and w and not in_any((s + e) / 2.0, bc_iv)]
        floor_ipus = merge_intervals(floor_words, gap=ipu_gap)
        return dict(voiced=voiced, activity=activity, noncontent=noncontent,
                    laughter=laughter, lex=lex, floor_ipus=floor_ipus)

    A, B = sided(words_a, bc_a), sided(words_b, bc_b)

    # ---- words / WPM / TTR -------------------------------------------------
    w1, w2 = A["lex"], B["lex"]
    words_sp1, words_sp2 = len(w1), len(w2)
    total_words = words_sp1 + words_sp2
    ttr1 = len(set(w1)) / words_sp1 if words_sp1 else 0.0
    ttr2 = len(set(w2)) / words_sp2 if words_sp2 else 0.0

    # ---- silence / non-content / speaker balance ---------------------------
    activity_union = union_dur(A["activity"] + B["activity"])
    silence_ratio = max(0.0, 1.0 - activity_union / dur_s)
    non_content_ratio = (A["noncontent"] + B["noncontent"]) / dur_s
    v1 = union_dur(A["voiced"])
    v2 = union_dur(B["voiced"])
    balance = v1 / (v1 + v2) if (v1 + v2) > 0 else float("nan")

    # ---- turns / speaker changes / FTO -------------------------------------
    ipus = ([(s, e, "A") for s, e in A["floor_ipus"]]
            + [(s, e, "B") for s, e in B["floor_ipus"]])
    turns, pauses = build_turns(ipus)
    turn_durs = [e - s for s, e, _ in turns]
    ftos = [turns[i][0] - turns[i - 1][1] for i in range(1, len(turns))]  # +gap/-overlap
    n_speaker_changes = len(ftos)

    # ---- overlap (both holding the floor; backchannels already excluded) ---
    ov = intersect(A["floor_ipus"], B["floor_ipus"])
    overlap_count = len(ov)
    overlap_dur = sum(e - s for s, e in ov)

    # ---- interruption (APPROXIMATE; no TRP / coop-disruptive info on SWBD) --
    # floor-taking: a speaker change whose onset overlaps the previous turn by
    #   more than trp_tol (the incomer barged in well before the floor was free).
    # non-floor-taking: an overlap segment that does NOT coincide with a floor
    #   change and begins > trp_tol before the incumbent turn's end.
    int_floor = sum(1 for f in ftos if f < -trp_tol)
    turn_spans = {("A", s, e) for s, e, sp in turns if sp == "A"}  # noqa: F841
    change_overlaps = set()
    for i in range(1, len(turns)):
        if turns[i][0] < turns[i - 1][1] - trp_tol:
            change_overlaps.add(round(turns[i][0], 3))
    int_nonfloor = 0
    for s, e in ov:
        if round(s, 3) in change_overlaps:
            continue
        # incumbent = the turn (either speaker) that contains s and started first
        cover = [t for t in turns if t[0] <= s < t[1]]
        if cover and min(t[1] for t in cover) - s > trp_tol:
            int_nonfloor += 1
    int_total = int_floor + int_nonfloor

    # ---- backchannels ------------------------------------------------------
    bc_count = len(bc_a) + len(bc_b)

    # ---- laughter ----------------------------------------------------------
    laughter_count = A["laughter"] + B["laughter"]

    # ---- composite event count --------------------------------------------
    nc_segments = len(merge_intervals(
        [(s, e) for s, e, k, _ in words_a if k == "noise"]
        + [(s, e) for s, e, k, _ in words_b if k == "noise"]))
    n_events = (len(turns) + bc_count + overlap_count + laughter_count + nc_segments)

    def rate(x):
        return x / dur_min if dur_min else float("nan")

    nan = float("nan")
    return {
        "task_id": sess,
        "conversation_type": "Switchboard",
        "duration_min": round(dur_min, 3),
        "n_events_mean": round(float(n_events), 1),
        "event_rate_per_min": round(rate(n_events), 3),
        "word_rate_wpm": round(rate(total_words), 2),
        "words_sp1": words_sp1,
        "words_sp2": words_sp2,
        "wpm_sp1": round(rate(words_sp1), 2),
        "wpm_sp2": round(rate(words_sp2), 2),
        "ttr_sp1": round(ttr1, 4),
        "ttr_sp2": round(ttr2, 4),
        "silence_ratio": round(silence_ratio, 4),
        "turn_count": len(turns),
        "turn_dur_mean": round(float(np.mean(turn_durs)) if turn_durs else 0.0, 3),
        "turn_dur_median": round(float(np.median(turn_durs)) if turn_durs else 0.0, 3),
        "turn_dur_std": round(float(np.std(turn_durs)) if turn_durs else 0.0, 3),
        "n_speaker_changes": n_speaker_changes,
        "fto_mean_s": round(float(np.mean(ftos)), 3) if ftos else 0.0,
        "fto_median_s": round(float(np.median(ftos)), 3) if ftos else 0.0,
        "fto_n": len(ftos),
        "speaker_balance": round(float(balance), 4),
        "bc_rate_per_min": round(rate(bc_count), 3),
        "bc_ack_per_min": nan,    # SWBD: backchannel subtypes unavailable
        "bc_cont_per_min": nan,
        "bc_react_per_min": nan,
        "int_rate_per_min": round(rate(int_total), 3),          # APPROX
        "int_competitive_per_min": nan,  # SWBD: coop/competitive unavailable
        "int_cooperative_per_min": nan,
        "int_floor_taking_per_min": round(rate(int_floor), 3),  # APPROX
        "int_non_floor_per_min": round(rate(int_nonfloor), 3),  # APPROX
        "overlap_count_per_min": round(rate(overlap_count), 3),
        "overlap_dur_s": round(overlap_dur, 2),
        "laughter_rate_per_min": round(rate(laughter_count), 3),
        "non_content_ratio": round(non_content_ratio, 4),
        "question_rate_per_min": (round(rate(q_count), 3)
                                  if q_count is not None else nan),  # SWDA subset
    }, {
        # event-level timing distributions, pooled by the driver
        # (gap durations = the positive FTOs)
        "fto": [round(f, 3) for f in ftos],
        "pause": [round(p, 3) for p in pauses],
    }


# --------------------------------------------------------------------------- #
# driver
# --------------------------------------------------------------------------- #
METRIC_KEYS = [
    "duration_min", "n_events_mean", "event_rate_per_min", "word_rate_wpm",
    "words_sp1", "words_sp2", "wpm_sp1", "wpm_sp2", "ttr_sp1", "ttr_sp2",
    "silence_ratio", "turn_count", "turn_dur_mean", "turn_dur_median",
    "turn_dur_std", "n_speaker_changes", "fto_mean_s", "fto_median_s", "fto_n",
    "speaker_balance", "bc_rate_per_min", "bc_ack_per_min", "bc_cont_per_min",
    "bc_react_per_min", "int_rate_per_min", "int_competitive_per_min",
    "int_cooperative_per_min", "int_floor_taking_per_min", "int_non_floor_per_min",
    "overlap_count_per_min", "overlap_dur_s", "laughter_rate_per_min",
    "non_content_ratio", "question_rate_per_min",
]


def find_sessions(trans_root):
    out = {}
    for wf in glob.glob(os.path.join(trans_root, "*", "*", "sw*A-ms98-a-word.text")):
        base = os.path.basename(wf)            # sw2001A-ms98-a-word.text
        sess = base[2:6]
        out[sess] = os.path.dirname(wf)
    return dict(sorted(out.items()))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--trans-root", default="data/swb_ms98_transcriptions",
                    help="MS-State transcriptions root: <root>/<2-digit>/<session>/"
                         "sw<session>{A,B}-ms98-a-word.text")
    ap.add_argument("--backchannels", default="backchannels.csv",
                    help="VAP/Ekstedt Switchboard backchannels.csv")
    ap.add_argument("--swda-root", default="data/swda",
                    help="SWDA root (sw*utt/*.utt.csv) for question rate; "
                         "'' to disable")
    ap.add_argument("--out-dir", default="results")
    ap.add_argument("--ipu-gap", type=float, default=DEF_IPU_GAP)
    ap.add_argument("--trp-tol", type=float, default=DEF_TRP_TOL)
    ap.add_argument("--label", default="Switchboard",
                    help="type label for the aggregate row")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    sessions = find_sessions(args.trans_root)
    print(f"Found {len(sessions)} Switchboard conversations", file=sys.stderr)
    bc = load_backchannels(args.backchannels)
    print(f"Backchannels: {len(bc)} sessions", file=sys.stderr)
    swda_q = load_swda_questions(args.swda_root)
    print(f"SWDA question coverage: {len(swda_q)} sessions", file=sys.stderr)

    rows = []
    dists = {"fto": [], "pause": []}
    for i, (sess, d) in enumerate(sessions.items(), 1):
        wa = os.path.join(d, f"sw{sess}A-ms98-a-word.text")
        wb = os.path.join(d, f"sw{sess}B-ms98-a-word.text")
        try:
            words_a = parse_word_file(wa)
            words_b = parse_word_file(wb)
            result = analyze(sess, words_a, words_b, bc.get(sess, {}),
                             swda_q.get(sess), args.ipu_gap, args.trp_tol)
            if result:
                row, conv_dists = result
                rows.append(row)
                for k in dists:
                    dists[k].extend(conv_dists[k])
        except Exception as e:  # noqa: BLE001
            print(f"  ! {sess}: {e}", file=sys.stderr)
        if i % 250 == 0:
            print(f"  {i}/{len(sessions)}", file=sys.stderr)

    # ---- per-conversation csv ---------------------------------------------
    pc = os.path.join(args.out_dir, "swbd_per_conversation.csv")
    fields = ["task_id", "conversation_type"] + METRIC_KEYS
    with open(pc, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: ("" if (isinstance(r[k], float) and np.isnan(r[k])) else r[k])
                        for k in fields})

    # ---- aggregate (single row, FLOOR per_type_aggregate flat schema) ------
    n_q = sum(1 for r in rows if not np.isnan(r["question_rate_per_min"]))
    agg = {"type": args.label, "n_conversations": len(rows)}
    detail = {}
    for k in METRIC_KEYS:
        vals = np.array([r[k] for r in rows], dtype=float)
        vals = vals[~np.isnan(vals)]
        if len(vals):
            agg[f"{k}_mean"] = round(float(vals.mean()), 3)
            detail[k] = {
                "mean": round(float(vals.mean()), 3),
                "median": round(float(np.median(vals)), 3),
                "std": round(float(vals.std()), 3),
                "min": round(float(vals.min()), 3),
                "max": round(float(vals.max()), 3),
                "n": int(len(vals)),
            }
        else:
            agg[f"{k}_mean"] = ""
            detail[k] = None

    ag_csv = os.path.join(args.out_dir, "swbd_per_type_aggregate.csv")
    with open(ag_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["type", "n_conversations"] + [f"{k}_mean" for k in METRIC_KEYS])
        w.writerow([agg["type"], agg["n_conversations"]]
                   + [agg[f"{k}_mean"] for k in METRIC_KEYS])

    with open(os.path.join(args.out_dir, "swbd_per_type_aggregate.json"), "w") as f:
        json.dump({"label": args.label, "n_conversations": len(rows),
                   "n_conversations_with_questions": n_q,
                   "ipu_gap": args.ipu_gap, "trp_tol": args.trp_tol,
                   "metrics": detail}, f, indent=2)

    # ---- full timing distributions (FTO / pause), pooled -------------------
    with open(os.path.join(args.out_dir, "swbd_timing_distributions.json"), "w") as f:
        json.dump({"corpus": args.label, "n_conversations": len(rows),
                   "ipu_gap": args.ipu_gap, "trp_tol": args.trp_tol,
                   "groups": {args.label: dists}}, f)

    print(f"\nWrote:\n  {pc}\n  {ag_csv}\n  "
          f"{os.path.join(args.out_dir, 'swbd_per_type_aggregate.json')}")
    print(f"\n{len(rows)} conversations | question_rate computed on "
          f"{n_q} SWDA-covered conversations")
    return 0


if __name__ == "__main__":
    sys.exit(main())
