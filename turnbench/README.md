# Evaluation: formulation & methodology

How a predictions file is turned into EOT / Interruption scores. The
submission format lives in [`../docs/SUBMISSION_FORMAT.md`](../docs/SUBMISSION_FORMAT.md).

## Tasks

Two boundary events, per speaker (whole-floor agreement is essentially VAD
and uninformative):

- **EOT** — a speaker's turn ends (the floor leaves them).
- **INT** — a speaker takes the floor from the other (a barge-in, on the
  *interrupter's* channel).

A submission declares the times at which it detects each, per conversation
and per speaker. The lists are **committed by the submitter** — they bake in
the model's operating point. The scorer does **not** sweep a threshold, and
timestamps are commit times (see ../docs/SUBMISSION_FORMAT.md): how the model
maps its internal signals to discrete events is the submitter's decision, made
once.

## Gold (`gold.py`)

The dataset publishes the **raw 3-annotator tracks** (the per-speaker
annotation columns, parsed losslessly from the source SRTs); the scorer builds
the gold from them at scoring time. **Code is the
source of truth for the gold** — no consensus artifact is published, so the
gold at a given commit is fully determined by this repo, and scores are only
comparable across runs of the same scorer version.

Each annotated segment maps to an expected floor state. We keep segments with
**majority (2 of 3) inter-annotator agreement** (endpoints within ±0.2 s; the
gold time is the median across the agreeing annotators). A dissenting annotator
the majority outvotes is settled by the majority, not treated as uncertain —
so only spans where **no majority forms at all** become **excluded intervals**
(predicted events falling inside them are neither rewarded nor penalised).

Agreement is judged **at the granularity each task needs**, because the
taxonomy is hierarchical — "Turn" is the coarser level above the
interruption labels:

- The **turn view** asks "does this span *claim the floor*?" Turn-family
  labels, `Overlap`, and *Floor-taking* Interruptions are one class there, so
  a span labelled Normal Turn / Floor-taking Interruption / Overlap reaches a
  floor-claiming majority. This view defines turns, and therefore EOT.
- The **label view** keeps the canonical labels apart and drives the INT task,
  where turn-vs-interruption is the very distinction being tested.

Annotators split labels systematically at contested handovers (the same
overlap is defensibly a turn, an overlap, or a cooperative interruption), so
judging EOT at the fine level would exclude exactly the floor changes the
benchmark cares about. Each view has its own excluded intervals, masking its
own task's predictions.

## Floor construction (also `gold.py`)

The annotation is **VAD-segmented**: one speaker's turn is chopped into many
short `Turn` segments separated by gaps. So most segment-ends are **not**
real turn ends — they are mid-turn pauses. We classify every `Turn`
segment-end:

- **EOT positive** — the floor passes to the *other* speaker (their next
  `Turn` starts before this speaker resumes, or their `Turn` is already in
  progress as this segment ends), or it is the speaker's last turn. An anchor
  *time* (the segment end).
- **EOT negative (mid-turn pause)** — the *same* speaker resumes next, with
  no handover in the gap. A *span*, scored only while it is believably a
  quiet within-turn hold: it runs from the segment end and is **truncated at
  the first contrary evidence** — an excluded interval (either speaker), the
  speaker's own non-floor vocalisation (a backchannel mid-"pause" means they
  are listening), or an other-speaker Interruption — and dropped entirely if
  contaminated from the start. Other-speaker backchannels do *not* truncate:
  they are evidence the speaker holds the floor. Because the span ends at the
  speaker's resumption at the latest, a real EOT can never lie inside one — a
  correct model cannot be penalised by a nearby pause.

Interruption events define the INT sets. An interruption means the other
speaker was actually *interrupted* — the floor changes hands — so only
floor-taking events are scored:

- **INT positive** — a consensus *floor-taking* `Interruption` onset (majority
  on floor-taking, on the interrupter's channel). An anchor time.
- **INT negative** — a `Backchannel` or channel-bleed (`NonContent`) event's
  own extent: the other speaker made a sound but did **not** take the floor.
  (Annotator-level `Overlap` collapses into `Turn` — overlapping speech is
  turn content.)
- **Excluded for INT** — consensus *non-floor-taking* attempts and
  subtype-disputed interruptions. At onset an attempt is indistinguishable
  from a real interruption, so firing on one is neither rewarded nor
  penalised.

On the 38-conversation dev split, majority consensus yields **1904 EOT-pos /
1063 EOT-neg** and **347 INT-pos / 3733 INT-neg** (`python -m turnbench.gold stats`),
with 1447 excluded intervals across the two views (545 EOT, 902 INT). The INT
task stays high-precision: only majority floor-taking interruptions are
positives, while non-floor-taking attempts and genuine no-majority regions are
excluded.

## Scoring (`score.py`)

Scoring is **event-anchored**: per speaker and task, the timeline divides
into scored regions, and predictions are only ever read inside them.

**Positive windows.** For each gold event at time `t`, we search
`[t − τ_pre, min(t + τ_max, next)]` (`τ_pre = 0.25 s`, `τ_max = 3.0 s`) in the
relevant list — the speaker's `eot` times for EOT, the interrupter's
`interruption` times for INT. `τ_max` is the latency deadline; a correct event
after it is a miss. The window's upper end is also clamped to `next`, **this
speaker's next event of the same task**: past it, a fire belongs to that event,
not this one, so the window must not reach in and claim it (each speaker is
scored on their own channel, so only same-speaker anchors bound the window).
`τ_pre` is a **matching tolerance**: the gold boundary is only annotation-exact
(annotators agree within ±0.2 s), so a slightly-early prediction still counts,
with negative latency.

**Negative spans.** Firing inside a mid-turn pause or a backchannel extent is
a false positive — at most **one per span**, however many predictions land in
it. The `τ_pre` tolerance is charged here too (`[start − τ_pre, end]`):
speculation is symmetric, so firing just before a segment end is rewarded iff
the turn really ends there and penalised iff the speaker was only pausing. A
model cannot harvest early-credit TPs by firing constantly during speech.

**Matching is one-to-one.** Positives are processed in time order; each
claims the earliest unclaimed prediction in its window. One prediction can
never satisfy two gold events, and a prediction claimed as a TP — or lying in
any positive window — is never also an FP. A correct model scores recall 1.0,
fp_rate 0.0 by construction (this is tested). These rules are the standard
ones in event-detection evaluation (one-to-one matching, tolerance collars,
per-site false-alarm rates).

| | positive event | negative span |
|---|---|---|
| predicted event in region | **TP** (latency = `t_pred − t`) | **FP** |
| no predicted event in region | **FN** | **TN** |

Predicted events inside excluded intervals are dropped before any of this;
events outside every scored region are invisible — neither rewarded nor
penalised (see caveats).

Per task: `recall = TP/(TP+FN)`, `fp_rate = FP/(FP+TN)`, latency
**p10/p50/p90** over TPs. `fp_rate` reads: *of the moments designed to fool a
turn-taking model — pauses, backchannels — what fraction fooled this one.* It
is a probe of hard negatives with a clean denominator, **not** a
false-alarms-per-hour measure over all behaviour.

### Ranking

The leaderboard gates on `fp_rate ≤ 0.1` on **dev** — the split the operating
point is selected on — and ranks in-budget submissions by test **recall**. A
test `fp_rate` above **1.5× the budget (0.15)** rejects the submission as
miscalibrated: dev is publicly labeled, so the dev gate alone cannot bound
test-side over-firing. EOT remains a latency-vs-false-fire tradeoff, not a
detection problem — a model that never fires has `fp_rate = 0` but infinite
latency. (`qualifies()` / `rank_key()` are generic helpers for the older
budget + recall-floor / latency-ranked view.)

## Pipeline

```
[turnbench.score — identical for dev and test]
annotation tracks ──gold.py (consensus + floor)──▶ event sets ─┐
                                                               ▼
predictions.json ─submission.py (validate)─▶ event times ─▶ score.py ─▶ scores
```

`turnbench.score` is the shared artifact: a submitter runs it on **public dev**,
we run the **same** code on **private test** (`--dataset <private repo>`).
Treat everything in `eval/` as **read-only**: it is the scorer, and scores are
only comparable across the same version of it.

Validation is strict and loud (`submission.py`): every conversation present
exactly once, fixed keys only, event lists strictly increasing, finite, and
inside the conversation's audio. A submission that scores locally will score
server-side.

`baselines/rms_vad/` is the degenerate energy-VAD baseline (fires on every
silence — low latency, high FP), the high-FP corner of the latency-vs-FP
tradeoff. The opposite corner is a model that never fires: `fp_rate` 0,
recall 0, infinite latency.

```bash
# install eval + dev deps (once)
uv sync --extra eval --extra dev

# emit + score a baseline on dev (the dataset is fetched from HF and cached automatically)
uv run python -m baselines.rms_vad.predict --out rms_vad_predictions.json
uv run python -m turnbench.score rms_vad_predictions.json

# validate floor construction / scorer
uv run python -m turnbench.gold stats
uv run pytest
```

## Caveats

- **Event-anchored**: scoring only looks inside the scored regions, so FP
  behaviour is only measured where we sample (mid-turn pauses, backchannels).
  Firing elsewhere — mid-speech away from boundaries, the other speaker's
  turn, dead air — is invisible to the score: stray fires gain nothing, and
  `fp_rate` is not a per-hour false-alarm rate.
- **Causality is affirmed, not enforced**: models run in the submitter's own
  environment, so the commit-time rule (../docs/SUBMISSION_FORMAT.md) is a
  term of submission rather than a property the scorer enforces. Scores are
  only as causal as the pipeline that produced the file.
- **Consensus can empty a conversation**: where annotators rarely reach a
  majority, a conversation can have *zero* `Turn` events survive and contribute
  no EOT events. This is a data property, not a bug.
- **Dev set on HF**: [`mundo-ai/turn-benchmark-dev`](https://huggingface.co/datasets/mundo-ai/turn-benchmark-dev) — audio + raw annotator
  tracks + metadata, one parquet row per conversation, fetched automatically
  into the HF cache (`eval/data.py`). The **public test set**
  ([`mundo-ai/turn-benchmark-test`](https://huggingface.co/datasets/mundo-ai/turn-benchmark-test))
  ships audio only — its annotation columns are blanked. **Test labels are never
  published**: scoring runs internally against a separate private labeled set
  (`--dataset <private repo>`).
