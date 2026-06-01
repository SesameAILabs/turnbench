#!/usr/bin/env python3
"""Mimi-based Endpointer (Kyutai).

Uses Kyutai's Mimi neural audio codec as the acoustic backbone for
end-of-turn detection. Mimi is a streaming neural audio codec that
combines semantic and acoustic information into audio tokens at 12.5 Hz
and 1.1 kbps.

The endpointer operates on the Mimi token stream at 12.5 Hz and processes
two audio streams simultaneously — one per speaker channel.

Params: < 50M.
Output label space: 4-class per frame {user, user-end, system, system-end}.

Evaluation:
  EOT:           fire when `user-end` is predicted.
  Interruption:  fire when `system` or `system-end` appears while user
                 was active.
"""
from __future__ import annotations

from pathlib import Path


def predict(sample_dir: Path) -> dict[int, list[tuple[float, float, str]]]:
    raise NotImplementedError("TODO: load Mimi codec + endpointer head")


if __name__ == "__main__":
    raise SystemExit("Not implemented yet.")
