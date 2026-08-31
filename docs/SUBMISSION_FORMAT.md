# Submission format

Turn-taking evaluation for conversational audio models: detecting
**end-of-turn (EOT)** and **interruption (INT)** events in two-channel
recorded conversations, scored on recall, false-positive rate, and detection
latency.

Run your model however you like, in your own repo and environment, and
submit a single JSON file of its predictions.

## Format

For each conversation and each speaker, the times (in seconds) at which your
model detects EOTs and interruptions:

```json
{
  "schema_version": 1,
  "predictions": [
    {
      "conversation_id": "20",
      "speaker_1": {
        "eot": [12.84, 58.43, 104.71],
        "interruption": [31.20]
      },
      "speaker_2": {
        "eot": [27.10, 71.92],
        "interruption": []
      }
    }
  ]
}
```

- `schema_version` — always `1`.
- `predictions` — exactly one entry per conversation in the dataset. If your
  model detected nothing in a conversation, include it with empty lists.
- `conversation_id` — the conversation's id in the dataset (its `conversation_id`).
- `speaker_1` / `speaker_2` — correspond to the conversation's
  `speaker_1_audio` / `speaker_2_audio` channels.
- `eot` — times at which this speaker's turn ends.
- `interruption` — times at which this speaker takes the floor while the
  other speaker holds it.

Times are float seconds on the shared conversation clock. Each list must be
strictly increasing and within the audio's duration; unknown keys are
rejected.

## Causality is a term of submission

A timestamp is the time **by which all audio the decision depended on has
been heard**: your model's output at time `t` may use audio up to `t` and
nothing after it.

If your model uses lookahead, fold it into the timestamp: an event detected
at `0.7` using audio through `1.0` is reported as `1.0`. Lookahead is simply
latency you pay, as in deployment. By submitting you affirm your file
follows this rule.

## Scoring

Full methodology: [turnbench/README.md](../turnbench/README.md). In brief:

- For each gold event at time `t`, the window `[t − 0.25 s, min(t + 3.0 s,
  next)]` is searched for one of your events: found → true positive with
  latency `t_pred − t`; not found → miss. Matching is one-to-one. `next` is
  this speaker's next event of the same task — the window never reaches past it
  to claim a fire that belongs to the next event.
- Firing during a mid-turn pause or a backchannel is a false positive — at
  most one per pause/backchannel.
- Regions where the annotators disagreed are excluded from scoring.
- Your event lists are your operating point. There are no confidence scores
  or thresholds — emit an event exactly when your deployed system would
  commit to acting on one.
- Per task: recall, false-positive rate, latency p10/p50/p90. The leaderboard
  ranks each task separately by test recall. A task qualifies when its test
  false-positive rate is at most 0.15; a submission over that ceiling is still
  shown, ranked below all qualifiers. Baseline operating points were
  selected on **dev** under a 0.1 false-positive budget.

## Score on the dev set

```bash
uv sync

uv run python -m turnbench.score predictions.json
```

The dev set ([`mundo-ai/turn-benchmark-dev`](https://huggingface.co/datasets/mundo-ai/turn-benchmark-dev),
audio + annotations) downloads automatically on first run — it is **gated**, so
request access on the dataset page and run `huggingface-cli login` first. The
same scorer runs server-side against a separate private labeled test set, whose
annotations are never published — a file that validates and scores here will
validate and score there.

A reference baseline emits a valid predictions file:

```bash
uv run python -m baselines.rms_vad.predict --out rms_vad_predictions.json
```

## Submission policy

For an official submission, email turnbench@sesame.com with:

1. Model display name for the leaderboard
2. Organization name
3. `predictions-dev.json` and `predictions-test.json`

We will accept only one submission per model, with the operating point baked
in ahead of time. Entries cannot be withdrawn, and an entry over the FPR
ceiling is ranked below qualifiers. We accept new entries for different models, but each
entry must be a materially different model, and must be the output of the
submitter's own system running on the benchmark audio. Please indicate any
changes if it is not obvious.

Qualification is per task (EOT vs. INT) and judged on the test split. A
submission qualifies on its task when its test false-positive rate is at most
0.15, and then is ranked among other systems based on its task's test recall.
A submission can qualify on one task and rank below the qualifiers on the
other.

A public write-up or repository documenting the system and how its operating
point was chosen is encouraged, not required.

We reserve the right to reasonably decline a submission (eg. invalid or
non-causal files, duplicates of an existing entry), and to keep the
leaderboard up to date by pruning entries that do not belong (eg. outdated
models).

## Sweeping in memory (no JSON files)

How you turn your model's continuous scores into committed event times, and
how you choose an operating point, is yours — that is where a threshold sweep
lives, and it is the submitter's own code. To make sweeping cheap, the scorer
exposes an in-memory entry point so you can score a candidate operating point
without writing a file:

```python
from turnbench.data import resolve_dataset
from turnbench.score import score_submission
from turnbench.submission import SCHEMA_VERSION, ConversationPrediction, SpeakerEvents, Submission

dataset = resolve_dataset()  # public dev set; or resolve_dataset("<hf repo|local dir>")

for threshold in my_thresholds:
    submission = Submission(
        schema_version=SCHEMA_VERSION,
        predictions=[my_discrete_events(conversation_id, threshold) for conversation_id in ...],
    )
    scores = score_submission(submission, dataset)  # discrete events -> aggregate scores
    print(threshold, scores.task_eot.recall, scores.task_eot.fp_rate)
```

`baselines/oracle_annotator/predict.py` is a runnable example (it scores the
gold events themselves, landing at recall 1.0, fp_rate 0.0).
