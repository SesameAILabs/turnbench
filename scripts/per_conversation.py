#!/usr/bin/env python3
"""Per-conversation metrics, optionally aggregated by conversation type.

Outputs:
    stats_out/per_conversation.csv          one row per task_id (all 20 metrics)
    stats_out/per_type_aggregate.json       mean/median/std/min/max grouped by type
    stats_out/per_type_aggregate.csv        flat per-type summary

Conversation-type mapping is read from $CONV_TYPES_CSV (in .env, optional).
The CSV must have columns `task_id,type`. If missing, all rows get type "all".

Metrics computed per conversation (mean across annotators a/b/c unless noted):
  1  duration_min
  2  word_rate_wpm                     | 18 ttr_sp1 / ttr_sp2 (type-token ratio)
  3  event_rate_per_min                | 19 words_sp1/sp2, wpm_sp1/sp2
  4  silence_ratio                     | 20 question_rate_per_min
  5  turn_dur_{mean,median,std}        | 14 iaa_kappa_{ab,ac,bc}
  6  n_speaker_changes                 | 15 iaa_fleiss_kappa
  7  fto_{mean,median,n}               | 16 event_count_cv (coefficient of var across a/b/c)
  8  speaker_balance                   | 17 boundary_f1_{ab,ac,bc} (onset within +/-200ms)
  9  bc_rate (+ ack/cont/react)
  10 interruption_rate (+ subtypes)
  11 overlap_count_per_min, overlap_dur_s
  12 laughter_rate_per_min
  13 non_content_ratio
"""
from __future__ import annotations

import csv
import json
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import soundfile as sf


SRT_TIME = re.compile(r"(\d+):(\d{2}):(\d{2}),(\d{3})")
LABEL = re.compile(r"\[([^\]]+)\]")
WORD = re.compile(r"[A-Za-z][A-Za-z'\-]*")
ANNOTATORS = ("a", "b", "c")
SPEAKERS = (1, 2)
WINDOW_S = 0.1
BOUNDARY_TOL_S = 0.2

BACKCHANNEL = {
    "Acknowledgement Backchannel": "ack",
    "Continuer Backchannel": "cont",
    "Reaction Backchannel": "react",
}
INTERRUPTION = {
    "Non-floor Taking Competitive Interruption": ("non_floor", "competitive"),
    "Non-floor Taking Cooperative Interruption": ("non_floor", "cooperative"),
    "Floor-taking Cooperative Interruption": ("floor_taking", "cooperative"),
    "Floor-taking Competitive Interruption": ("floor_taking", "competitive"),
}
TURN_LABELS = {"Normal Turn", "Strong Floor Hold", "Bounded Response"}
NON_CONTENT = {"Non-Speech Noise", "Channel Bleed", "Speech, Non-Linguistic"}
OVERLAP = {"Overlap"}
LAUGHTER = {"Laughter"}

CATEGORY: dict[str, str] = {}
for l in BACKCHANNEL: CATEGORY[l] = "BC"
for l in INTERRUPTION: CATEGORY[l] = "INT"
for l in TURN_LABELS: CATEGORY[l] = "TURN"
for l in OVERLAP: CATEGORY[l] = "OVL"
for l in NON_CONTENT: CATEGORY[l] = "NC"
for l in LAUGHTER: CATEGORY[l] = "LAUGH"


def load_env(p: Path) -> dict[str, str]:
    env = {}
    for line in p.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            env[k.strip()] = v.strip()
    return env


def srt_seconds(ts: str) -> float:
    h, m, s, ms = SRT_TIME.match(ts).groups()
    return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000


def parse_srt(path: Path) -> list[tuple[float, float, str, str]]:
    out = []
    for block in re.split(r"\n\s*\n", path.read_text(encoding="utf-8").strip()):
        lines = [ln for ln in block.splitlines() if ln.strip()]
        if len(lines) < 2:
            continue
        idx = 1 if lines[0].strip().isdigit() else 0
        if idx >= len(lines) or "-->" not in lines[idx]:
            continue
        a, b = [t.strip() for t in lines[idx].split("-->")]
        body = " ".join(lines[idx + 1:]).strip()
        m = LABEL.match(body)
        label = m.group(1) if m else ""
        text = body[m.end():].strip() if m else body
        out.append((srt_seconds(a), srt_seconds(b), label, text))
    return out


def wav_dur(path: Path) -> float:
    info = sf.info(str(path))
    return info.frames / float(info.samplerate)


def union_dur(intervals: list[tuple[float, float]]) -> float:
    if not intervals:
        return 0.0
    intervals = sorted(intervals)
    total, cs, ce = 0.0, intervals[0][0], intervals[0][1]
    for s, e in intervals[1:]:
        if s > ce:
            total += ce - cs
            cs, ce = s, e
        else:
            ce = max(ce, e)
    return total + (ce - cs)


def windowize(events, n_windows: int) -> np.ndarray:
    arr = np.full(n_windows, "_", dtype="<U5")
    for s, e, l, _ in events:
        cat = CATEGORY.get(l, "OTH" if l else "_")
        i = max(0, int(s / WINDOW_S))
        j = min(n_windows, int(np.ceil(e / WINDOW_S)))
        arr[i:j] = cat
    return arr


def cohens_kappa(a: np.ndarray, b: np.ndarray) -> float:
    cats = sorted(set(a.tolist()) | set(b.tolist()))
    if not cats or len(a) == 0:
        return float("nan")
    idx = {c: i for i, c in enumerate(cats)}
    cm = np.zeros((len(cats), len(cats)), dtype=np.int64)
    for ai, bi in zip(a, b):
        cm[idx[ai], idx[bi]] += 1
    n = len(a)
    po = np.trace(cm) / n
    pe = float(((cm.sum(1) / n) * (cm.sum(0) / n)).sum())
    return 1.0 if pe >= 1.0 else (po - pe) / (1.0 - pe)


def fleiss_kappa(seqs: list[np.ndarray]) -> float:
    """seqs: list of length-N arrays of category strings."""
    n_raters = len(seqs)
    N = len(seqs[0])
    cats = sorted({c for s in seqs for c in set(s.tolist())})
    idx = {c: i for i, c in enumerate(cats)}
    M = np.zeros((N, len(cats)), dtype=np.int64)
    for s in seqs:
        for i, c in enumerate(s):
            M[i, idx[c]] += 1
    P_i = (M ** 2).sum(axis=1) - n_raters
    P_bar = P_i.sum() / (N * n_raters * (n_raters - 1))
    p_j = M.sum(axis=0) / (N * n_raters)
    P_e = (p_j ** 2).sum()
    return 1.0 if P_e >= 1.0 else (P_bar - P_e) / (1.0 - P_e)


def boundary_f1(a_on: list[float], b_on: list[float], tol: float = BOUNDARY_TOL_S) -> float:
    if not a_on and not b_on:
        return float("nan")
    if not a_on or not b_on:
        return 0.0
    a_sorted = sorted(a_on)
    b_used = [False] * len(b_on)
    b_sorted_idx = sorted(range(len(b_on)), key=lambda i: b_on[i])
    b_sorted = [b_on[i] for i in b_sorted_idx]
    tp = 0
    j = 0
    for x in a_sorted:
        while j < len(b_sorted) and b_sorted[j] < x - tol:
            j += 1
        k = j
        while k < len(b_sorted) and b_sorted[k] <= x + tol:
            if not b_used[k]:
                b_used[k] = True
                tp += 1
                break
            k += 1
    p = tp / len(a_on)
    r = tp / len(b_on)
    return 0.0 if (p + r) == 0 else 2 * p * r / (p + r)


def speaker_change_ftos(events_by_speaker: dict[int, list]) -> list[float]:
    turns = []
    for sp, evs in events_by_speaker.items():
        for s, e, l, _ in evs:
            if l in TURN_LABELS:
                turns.append((s, e, sp))
    turns.sort()
    ftos = []
    for i in range(1, len(turns)):
        prev_s, prev_e, prev_sp = turns[i - 1]
        s, e, sp = turns[i]
        if sp != prev_sp:
            ftos.append(s - prev_e)  # +gap / -overlap
    return ftos


def analyze(d: Path) -> dict:
    dur_s = wav_dur(d / "combined_audio.wav")
    dur_min = dur_s / 60
    n_win = int(np.ceil(dur_s / WINDOW_S))

    events = {(sp, ann): parse_srt(d / f"speaker_{sp}_annotation_{ann}.srt")
              for sp in SPEAKERS for ann in ANNOTATORS}

    # Means across annotators
    bc_total, bc_sub = [], {"ack": [], "cont": [], "react": []}
    int_total, int_sub = [], {"competitive": [], "cooperative": [], "floor_taking": [], "non_floor": []}
    overlap_n, overlap_d = [], []
    laughter_n, n_events, q_n = [], [], []
    non_content_d = []
    turn_durs_pooled = []
    fto_pooled = []
    n_speaker_changes_pooled = []
    silence_ratio_pooled = []

    for ann in ANNOTATORS:
        evs = events[(1, ann)] + events[(2, ann)]
        n_events.append(len(evs))
        bc_total.append(sum(1 for *_, l, _ in [(s, e, l, t) for s, e, l, t in evs] if l in BACKCHANNEL))
        for l, sub in BACKCHANNEL.items():
            bc_sub[sub].append(sum(1 for s, e, lab, _ in evs if lab == l))
        int_total.append(sum(1 for s, e, l, _ in evs if l in INTERRUPTION))
        ic = Counter()
        for s, e, l, _ in evs:
            if l in INTERRUPTION:
                typ, comp = INTERRUPTION[l]
                ic[typ] += 1
                ic[comp] += 1
        for k in int_sub:
            int_sub[k].append(ic[k])
        overlap_n.append(sum(1 for s, e, l, _ in evs if l in OVERLAP))
        overlap_d.append(sum(e - s for s, e, l, _ in evs if l in OVERLAP))
        laughter_n.append(sum(1 for s, e, l, _ in evs if l in LAUGHTER))
        q_n.append(sum(1 for s, e, _, t in evs if "?" in t))
        non_content_d.append(sum(e - s for s, e, l, _ in evs if l in NON_CONTENT))
        turn_durs_pooled.extend([e - s for s, e, l, _ in evs if l in TURN_LABELS])
        ftos = speaker_change_ftos({1: events[(1, ann)], 2: events[(2, ann)]})
        fto_pooled.extend(ftos)
        n_speaker_changes_pooled.append(len(ftos))
        # silence: 1 - union of all events (any label except empty)
        ivs = [(s, e) for s, e, l, _ in evs if l]
        sil = 1.0 - (union_dur(ivs) / dur_s if dur_s > 0 else 0.0)
        silence_ratio_pooled.append(max(0.0, sil))

    # words / TTR per speaker (annotator 'a')
    def words_for(sp: int, ann: str) -> list[str]:
        out = []
        for s, e, _, t in events[(sp, ann)]:
            out.extend(w.lower() for w in WORD.findall(t))
        return out
    w1 = words_for(1, "a")
    w2 = words_for(2, "a")
    ttr1 = len(set(w1)) / len(w1) if w1 else 0.0
    ttr2 = len(set(w2)) / len(w2) if w2 else 0.0

    # total words across annotators (mean) for word_rate
    total_words = float(np.mean([
        sum(len(WORD.findall(t)) for s, e, _, t in events[(sp, ann)])
        for ann in ANNOTATORS for sp in SPEAKERS
    ])) * 2  # mean above pools sp1+sp2; *2 undoes the /6 vs /3

    # Actually simpler: words per annotator (sp1+sp2), then mean across annotators
    words_per_ann = [sum(len(WORD.findall(t))
                         for sp in SPEAKERS for s, e, _, t in events[(sp, ann)])
                     for ann in ANNOTATORS]
    total_words = float(np.mean(words_per_ann))

    # speaker balance from voiced (non-NC) duration, mean across annotators
    def voiced(sp: int, ann: str) -> float:
        return sum(e - s for s, e, l, _ in events[(sp, ann)]
                   if l and l not in NON_CONTENT)
    v1 = float(np.mean([voiced(1, a) for a in ANNOTATORS]))
    v2 = float(np.mean([voiced(2, a) for a in ANNOTATORS]))
    balance = v1 / (v1 + v2) if (v1 + v2) > 0 else float("nan")

    # IAA: windowized, concat sp1+sp2
    seq = {a: np.concatenate([windowize(events[(1, a)], n_win),
                              windowize(events[(2, a)], n_win)])
           for a in ANNOTATORS}
    k_ab = cohens_kappa(seq["a"], seq["b"])
    k_ac = cohens_kappa(seq["a"], seq["c"])
    k_bc = cohens_kappa(seq["b"], seq["c"])
    fk = fleiss_kappa([seq["a"], seq["b"], seq["c"]])

    # event count CV across annotators
    ec = np.array(n_events, dtype=float)
    ec_cv = ec.std() / ec.mean() if ec.mean() > 0 else float("nan")

    # boundary F1: per speaker pool onsets, then average across speakers
    def onsets(ann: str) -> list[float]:
        return [s for sp in SPEAKERS for s, e, l, _ in events[(sp, ann)] if l]
    f1_ab = boundary_f1(onsets("a"), onsets("b"))
    f1_ac = boundary_f1(onsets("a"), onsets("c"))
    f1_bc = boundary_f1(onsets("b"), onsets("c"))

    def m(x):
        return round(float(np.mean(x)), 3) if len(x) else 0.0

    fto_arr = np.array(fto_pooled) if fto_pooled else np.array([0.0])

    return {
        "task_id": d.name,
        "duration_min": round(dur_min, 3),
        "n_events_mean": round(float(np.mean(n_events)), 1),
        "event_rate_per_min": round(float(np.mean(n_events)) / dur_min, 3),
        "word_rate_wpm": round(total_words / dur_min, 2),
        "words_sp1": len(w1),
        "words_sp2": len(w2),
        "wpm_sp1": round(len(w1) / dur_min, 2),
        "wpm_sp2": round(len(w2) / dur_min, 2),
        "ttr_sp1": round(ttr1, 4),
        "ttr_sp2": round(ttr2, 4),
        "silence_ratio": round(float(np.mean(silence_ratio_pooled)), 4),
        "turn_count": len(turn_durs_pooled) // 3,
        "turn_dur_mean": round(float(np.mean(turn_durs_pooled)) if turn_durs_pooled else 0.0, 3),
        "turn_dur_median": round(float(np.median(turn_durs_pooled)) if turn_durs_pooled else 0.0, 3),
        "turn_dur_std": round(float(np.std(turn_durs_pooled)) if turn_durs_pooled else 0.0, 3),
        "n_speaker_changes": round(float(np.mean(n_speaker_changes_pooled)), 1),
        "fto_mean_s": round(float(fto_arr.mean()), 3),
        "fto_median_s": round(float(np.median(fto_arr)), 3),
        "fto_n": len(fto_pooled) // 3,
        "speaker_balance": round(float(balance), 4),
        "bc_rate_per_min": round(float(np.mean(bc_total)) / dur_min, 3),
        "bc_ack_per_min": round(float(np.mean(bc_sub["ack"])) / dur_min, 3),
        "bc_cont_per_min": round(float(np.mean(bc_sub["cont"])) / dur_min, 3),
        "bc_react_per_min": round(float(np.mean(bc_sub["react"])) / dur_min, 3),
        "int_rate_per_min": round(float(np.mean(int_total)) / dur_min, 3),
        "int_competitive_per_min": round(float(np.mean(int_sub["competitive"])) / dur_min, 3),
        "int_cooperative_per_min": round(float(np.mean(int_sub["cooperative"])) / dur_min, 3),
        "int_floor_taking_per_min": round(float(np.mean(int_sub["floor_taking"])) / dur_min, 3),
        "int_non_floor_per_min": round(float(np.mean(int_sub["non_floor"])) / dur_min, 3),
        "overlap_count_per_min": round(float(np.mean(overlap_n)) / dur_min, 3),
        "overlap_dur_s": round(float(np.mean(overlap_d)), 2),
        "laughter_rate_per_min": round(float(np.mean(laughter_n)) / dur_min, 3),
        "non_content_ratio": round(float(np.mean(non_content_d)) / dur_s, 4),
        "iaa_kappa_ab": round(k_ab, 4),
        "iaa_kappa_ac": round(k_ac, 4),
        "iaa_kappa_bc": round(k_bc, 4),
        "iaa_fleiss_kappa": round(fk, 4),
        "event_count_cv_abc": round(float(ec_cv), 4),
        "boundary_f1_ab": round(f1_ab, 4),
        "boundary_f1_ac": round(f1_ac, 4),
        "boundary_f1_bc": round(f1_bc, 4),
        "question_rate_per_min": round(float(np.mean(q_n)) / dur_min, 3),
    }


def aggregate_by_type(rows: list[dict], type_map: dict[str, str]) -> dict:
    by: dict[str, list[dict]] = {}
    for r in rows:
        t = type_map.get(r["task_id"], "all")
        by.setdefault(t, []).append(r)
    metrics = [k for k in rows[0] if k != "task_id"]
    out: dict = {}
    for t, group in by.items():
        out[t] = {"n_conversations": len(group)}
        for k in metrics:
            vals = np.array([r[k] for r in group if isinstance(r[k], (int, float))
                             and not (isinstance(r[k], float) and np.isnan(r[k]))], dtype=float)
            if not len(vals):
                continue
            out[t][k] = {
                "mean": round(float(vals.mean()), 3),
                "median": round(float(np.median(vals)), 3),
                "std": round(float(vals.std()), 3),
                "min": round(float(vals.min()), 3),
                "max": round(float(vals.max()), 3),
            }
    return out


def main() -> int:
    repo = Path(__file__).resolve().parent.parent
    env = load_env(repo / ".env")
    root = Path(env["DATA_ROOT"]) / env["BATCH"]
    out_dir = Path(env.get("STATS_DIR", repo / "stats_out"))
    out_dir.mkdir(parents=True, exist_ok=True)

    type_map: dict[str, str] = {}
    types_csv = env.get("CONV_TYPES_CSV", "").strip()
    if types_csv and Path(types_csv).exists():
        with open(types_csv) as f:
            for r in csv.DictReader(f):
                type_map[str(r["task_id"])] = r["type"]
        print(f"Loaded {len(type_map)} type assignments from {types_csv}", file=sys.stderr)
    else:
        print("No CONV_TYPES_CSV set or file missing; aggregating as a single 'all' group.",
              file=sys.stderr)

    sample_dirs = sorted([p for p in root.iterdir() if p.is_dir()],
                         key=lambda p: int(p.name) if p.name.isdigit() else p.name)
    print(f"Analyzing {len(sample_dirs)} conversations...", file=sys.stderr)

    rows = []
    for i, d in enumerate(sample_dirs, 1):
        try:
            rows.append(analyze(d))
        except Exception as e:
            print(f"  ! {d.name}: {e}", file=sys.stderr)
        if i % 20 == 0:
            print(f"  {i}/{len(sample_dirs)}", file=sys.stderr)

    fieldnames = ["task_id", "conv_type"] + [k for k in rows[0] if k != "task_id"]
    with (out_dir / "per_conversation.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            r["conv_type"] = type_map.get(r["task_id"], "all")
            w.writerow(r)

    agg = aggregate_by_type(rows, type_map)
    (out_dir / "per_type_aggregate.json").write_text(json.dumps(agg, indent=2))

    # flat csv: type x metric.mean
    metric_keys = [k for k in rows[0] if k not in ("task_id", "conv_type")]
    with (out_dir / "per_type_aggregate.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["type", "n_conversations"] + [f"{k}_mean" for k in metric_keys])
        for t, stats in agg.items():
            row = [t, stats["n_conversations"]]
            for k in metric_keys:
                row.append(stats.get(k, {}).get("mean", ""))
            w.writerow(row)

    print(f"Wrote per_conversation.csv, per_type_aggregate.json/csv to {out_dir}")
    print(f"Types found: {sorted({r['conv_type'] for r in rows})}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
