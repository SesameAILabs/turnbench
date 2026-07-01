# Baseline scores by conversation metadata — findings

Per-conversation scores for the committed baselines, pooled by `conversation_type`
and speaker-gender pairing. Numbers below are the **test set (116 conversations,
golden gold)**; reproduce with:

```bash
HF_TOKEN=<gold-token> uv run --extra eval python data_analysis/scores_by_metadata.py \
    --dataset mundo-ai/turn-benchmark-test-golden
```

(Dev, `uv run --extra eval python data_analysis/scores_by_metadata.py`, runs token-free
but is thin at 38 conversations.) Cells are `recall / fp_rate / median-latency-ms`,
pooling TP/FN/FP/TN within each group. See `per_conversation.py` for the complementary
*dataset*-characterization stats.

## 1. Gold structure (baseline-independent)

| conversation_type | n | EOT+ | INT+ | INT+/conv |
| --- | --: | --: | --: | --: |
| Argumentative/Deliberative | 24 | 1234 | 268 | **11.2** |
| Casual/Spontaneous | 22 | 1303 | 142 | 6.5 |
| Collaborative/Problem-Solving | 20 | 1353 | 142 | 7.1 |
| Instructional | 19 | 877 | 95 | 5.0 |
| Narrative/Storytelling | 15 | 735 | 74 | 4.9 |
| Task-Oriented/Transactional | 16 | 791 | 83 | 5.2 |

- **Argumentative/Deliberative is the interruption hotspot — ~11 real interruptions/conv,
  ~2× every other type.** The benchmark's floor-taking signal is concentrated in arguments.
- Casual/Spontaneous & Collaborative are the most turn-dense (most EOT+) and backchannel-dense
  (most INT-negatives, ~130+/conv) — rapid, overlappy talk.
- Narrative/Storytelling & Instructional are the most monologic (fewest events/conv).

## 2. EOT is strikingly type-robust
Every model's EOT **recall is flat across all six types** (e.g. `openai_server_vad`
0.94–0.96 everywhere; `wavlm_large_anchor` 0.76–0.81). Conversation type mainly moves the
**false-positive rate of acoustic detectors** — `rms_vad` false-fires EOT most on pause-heavy
Narrative (0.70) and Casual (0.70), least on Instructional (0.51). Expected (more mid-turn
pauses → more false turn-ends), now quantified.

## 3. INT false positives concentrate in casual chit-chat, not arguments
Across nearly every model, INT false-positive rate **peaks on Casual/Spontaneous and
Task-Oriented** and is *lowest* on Argumentative/Collaborative:

| model | Casual fp | Task fp | Argumentative fp |
| --- | --: | --: | --: |
| openai_server_vad | **0.54** | **0.54** | 0.38 |
| rms_vad | **0.52** | **0.55** | 0.39 |
| espnet_turntaking_perchannel | **0.32** | 0.31 | 0.24 |

The conversations that trip up interruption detection are **backchannel-heavy casual talk**
(all those "uh-huh"s misfiring as floor-taking) — *not* the argument-heavy ones, despite
arguments carrying 2× the real interruptions. Mildly counterintuitive; worth highlighting.

## 4. Candidate gender effect on interruption detection (needs confirmation)
Several **independent** models detect interruptions worse on **female–female** pairs — lower
recall and/or higher fp — with **no comparable gap on EOT**:

| model (INT) | FF | MM |
| --- | --- | --- |
| wavlm_large_anchor | 0.73 rec / 0.17 fp | 0.84 / 0.09 |
| openai_semantic_vad | 0.40 rec / 0.27 fp | 0.55 / 0.27 |
| espnet_turntaking_perchannel | 0.81 / 0.30 fp | 0.75 / 0.21 |
| causal_wavlm_predictor/large | 0.73 / 0.21 fp | 0.74 / 0.14 |

That it appears across acoustic, semantic, and learned models — and only on INT — makes it
more than noise. **Caveats before claiming it:** small samples (FF=37, MM=26, mixed=53), and
gender pairing may correlate with conversation type. Needs a controlled check (hold type fixed)
before it goes in the paper.

## 5. Model-family regimes hold within every type
Acoustic (high-recall / high-fp), semantic (low / low), anchor (balanced but ~1100 ms INT
latency) — the ordering is unchanged inside all six types. The breakdown enriches the story
without overturning the headline.
