#!/usr/bin/env python3
"""Define speaker-disjoint dev/test splits.

Algorithm:
    1. Build a graph where speakers are nodes and each conversation is an
       edge between its two speakers. Find connected components.
    2. Within a component, all conversations must go to the same split
       (otherwise speakers leak across splits).
    3. Assign components greedily to either `dev` or `test` to hit a target
       size ratio (default 25/75) while keeping per-conversation-type counts
       balanced.

Writes:
    eval/splits/dev.txt   one task_id per line
    eval/splits/test.txt  one task_id per line
    eval/splits/_summary.json  per-split sample/speaker/type counts
"""
from __future__ import annotations

import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path


DEV_FRAC = 0.25
SEED = 0xA11
N_RESTARTS = 20000      # random restarts for the greedy assignment
TYPE_WEIGHT = 5.0       # how much we care about per-type balance vs overall size


def load_env(p: Path) -> dict:
    env = {}
    for line in p.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            env[k.strip()] = v.strip()
    return env


def find_components(speakers_per_sample: dict[str, tuple[str, str]]
                    ) -> list[set[str]]:
    """Union-find on speakers."""
    parent: dict[str, str] = {}

    def find(x: str) -> str:
        while parent.get(x, x) != x:
            parent[x] = parent.get(parent[x], parent[x])
            x = parent[x]
        return x

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for s1, s2 in speakers_per_sample.values():
        parent.setdefault(s1, s1)
        parent.setdefault(s2, s2)
        union(s1, s2)

    groups: dict[str, set[str]] = defaultdict(set)
    for sp in parent:
        groups[find(sp)].add(sp)
    return list(groups.values())


def main() -> int:
    repo = Path(__file__).resolve().parent.parent
    env = load_env(repo / ".env")
    data_root = Path(env["DATA_ROOT"]) / env["BATCH"]

    samples: list[dict] = []
    for d in sorted(data_root.iterdir()):
        if not d.is_dir():
            continue
        meta = json.loads((d / "metadata.json").read_text())
        samples.append({
            "task_id": meta["task_id"],
            "sp1": meta["speaker_1_actor_id"],
            "sp2": meta["speaker_2_actor_id"],
            "type": meta.get("conversation_type", "unknown"),
        })

    speakers_per_sample = {s["task_id"]: (s["sp1"], s["sp2"]) for s in samples}
    components = find_components(speakers_per_sample)
    speaker_to_comp = {}
    for i, comp in enumerate(components):
        for sp in comp:
            speaker_to_comp[sp] = i

    # Group samples by component
    comp_samples: dict[int, list[dict]] = defaultdict(list)
    for s in samples:
        c = speaker_to_comp[s["sp1"]]
        assert c == speaker_to_comp[s["sp2"]], \
            f"{s['task_id']} crosses components"
        comp_samples[c].append(s)

    target_dev = int(round(DEV_FRAC * len(samples)))
    type_totals = Counter(s["type"] for s in samples)
    type_targets = {t: DEV_FRAC * n for t, n in type_totals.items()}
    types_list = sorted(type_totals)

    # Pre-compute per-component sizes and type vectors
    comp_size = {c: len(comp_samples[c]) for c in comp_samples}
    comp_types = {c: Counter(s["type"] for s in comp_samples[c])
                  for c in comp_samples}

    def assignment_cost(dev_size: int, dev_t: Counter) -> float:
        size_err = abs(dev_size - target_dev)
        type_err = sum(abs(dev_t[t] - type_targets[t]) for t in types_list)
        return size_err + TYPE_WEIGHT * type_err

    def greedy(order: list[int]) -> tuple[dict[int, str], float]:
        dev_size = 0
        dev_t: Counter = Counter()
        out: dict[int, str] = {}
        for c in order:
            sz = comp_size[c]
            ct = comp_types[c]
            cost_dev = assignment_cost(dev_size + sz, dev_t + ct)
            cost_test = assignment_cost(dev_size, dev_t)
            if cost_dev < cost_test:
                out[c] = "dev"
                dev_size += sz
                dev_t.update(ct)
            else:
                out[c] = "test"
        return out, assignment_cost(dev_size, dev_t)

    # Random-restart greedy
    rng = random.Random(SEED)
    comp_ids = list(comp_samples.keys())
    best_assign: dict[int, str] | None = None
    best_cost = float("inf")
    for _ in range(N_RESTARTS):
        order = comp_ids[:]
        rng.shuffle(order)
        a, cost = greedy(order)
        if cost < best_cost:
            best_cost, best_assign = cost, a

    # Local search: try flipping each single component
    assigned = dict(best_assign)
    improved = True
    while improved:
        improved = False
        dev_size = sum(comp_size[c] for c, v in assigned.items() if v == "dev")
        dev_t = Counter()
        for c, v in assigned.items():
            if v == "dev":
                dev_t.update(comp_types[c])
        cur = assignment_cost(dev_size, dev_t)
        for c in comp_ids:
            sz, ct = comp_size[c], comp_types[c]
            if assigned[c] == "dev":
                new = assignment_cost(dev_size - sz, dev_t - ct)
                if new < cur:
                    assigned[c] = "test"
                    dev_size -= sz
                    for t, n in ct.items():
                        dev_t[t] -= n
                    cur = new
                    improved = True
            else:
                new = assignment_cost(dev_size + sz, dev_t + ct)
                if new < cur:
                    assigned[c] = "dev"
                    dev_size += sz
                    dev_t.update(ct)
                    cur = new
                    improved = True

    dev_count = sum(comp_size[c] for c, v in assigned.items() if v == "dev")
    test_count = len(samples) - dev_count
    dev_types = Counter()
    test_types = Counter()
    for c, v in assigned.items():
        (dev_types if v == "dev" else test_types).update(comp_types[c])

    # Materialize splits
    split_dir = repo / "eval" / "splits"
    split_dir.mkdir(parents=True, exist_ok=True)
    dev_tasks, test_tasks = [], []
    for c, choice in assigned.items():
        for s in comp_samples[c]:
            (dev_tasks if choice == "dev" else test_tasks).append(s["task_id"])

    dev_tasks.sort(key=lambda t: int(t) if t.isdigit() else t)
    test_tasks.sort(key=lambda t: int(t) if t.isdigit() else t)
    (split_dir / "dev.txt").write_text("\n".join(dev_tasks) + "\n")
    (split_dir / "test.txt").write_text("\n".join(test_tasks) + "\n")

    # ---- Also write a fully random split (no speaker constraint) as a baseline ----
    rng_random = random.Random(SEED)
    shuffled = [s["task_id"] for s in samples]
    rng_random.shuffle(shuffled)
    cut = int(round(DEV_FRAC * len(shuffled)))
    rand_dev = sorted(shuffled[:cut], key=lambda t: int(t) if t.isdigit() else t)
    rand_test = sorted(shuffled[cut:], key=lambda t: int(t) if t.isdigit() else t)
    (split_dir / "random_dev.txt").write_text("\n".join(rand_dev) + "\n")
    (split_dir / "random_test.txt").write_text("\n".join(rand_test) + "\n")

    # Speaker sets per split for the summary
    def speakers_in(tasks: list[str]) -> set[str]:
        out: set[str] = set()
        for t in tasks:
            out.update(speakers_per_sample[t])
        return out

    dev_speakers = speakers_in(dev_tasks)
    test_speakers = speakers_in(test_tasks)
    overlap = dev_speakers & test_speakers
    assert not overlap, f"speaker leak! {overlap}"

    rand_dev_speakers = speakers_in(rand_dev)
    rand_test_speakers = speakers_in(rand_test)
    rand_overlap = rand_dev_speakers & rand_test_speakers
    rand_dev_types = Counter()
    rand_test_types = Counter()
    for s in samples:
        if s["task_id"] in set(rand_dev):
            rand_dev_types[s["type"]] += 1
        else:
            rand_test_types[s["type"]] += 1

    summary = {
        "speaker_disjoint": {
            "n_samples": {"dev": len(dev_tasks), "test": len(test_tasks)},
            "n_speakers": {"dev": len(dev_speakers), "test": len(test_speakers)},
            "speaker_overlap": len(overlap),
            "n_components_dev": sum(1 for v in assigned.values() if v == "dev"),
            "n_components_test": sum(1 for v in assigned.values() if v == "test"),
            "by_type": {t: {"dev": dev_types[t], "test": test_types[t]}
                        for t in sorted(type_totals)},
        },
        "random": {
            "n_samples": {"dev": len(rand_dev), "test": len(rand_test)},
            "n_speakers": {"dev": len(rand_dev_speakers),
                           "test": len(rand_test_speakers)},
            "speaker_overlap": len(rand_overlap),
            "by_type": {t: {"dev": rand_dev_types[t], "test": rand_test_types[t]}
                        for t in sorted(type_totals)},
        },
        "totals": {
            "n_samples": len(samples),
            "n_speakers": len(speaker_to_comp),
            "n_components": len(components),
            "component_sizes": sorted([len(comp_samples[c]) for c in comp_samples],
                                      reverse=True),
            "by_type": dict(type_totals),
        },
        "seed": SEED,
        "dev_frac_target": DEV_FRAC,
    }
    (split_dir / "_summary.json").write_text(json.dumps(summary, indent=2))

    # Print human-readable
    print(f"Total: {len(samples)} samples, {len(speaker_to_comp)} speakers, "
          f"{len(components)} speaker-disjoint components.")
    print(f"Component sizes: {summary['totals']['component_sizes'][:15]}"
          f"{' ...' if len(components) > 15 else ''}")
    print()
    print(f"{'':33s} {'speaker-disjoint':>22s} {'random':>22s}")
    print(f"{'':33s} {'dev':>8s} {'test':>8s} {'overlap':>5s}  {'dev':>8s} {'test':>8s} {'overlap':>5s}")
    print(f"{'samples':33s} {len(dev_tasks):>8d} {len(test_tasks):>8d}        "
          f"{len(rand_dev):>8d} {len(rand_test):>8d}")
    print(f"{'unique speakers':33s} {len(dev_speakers):>8d} {len(test_speakers):>8d} "
          f"{len(overlap):>5d}  {len(rand_dev_speakers):>8d} {len(rand_test_speakers):>8d} "
          f"{len(rand_overlap):>5d}")
    for t in sorted(type_totals):
        print(f"{t:33s} {dev_types[t]:>8d} {test_types[t]:>8d}        "
              f"{rand_dev_types[t]:>8d} {rand_test_types[t]:>8d}")

    print(f"\nWrote dev.txt, test.txt, random_dev.txt, random_test.txt, _summary.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
