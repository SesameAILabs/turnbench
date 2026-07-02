#!/usr/bin/env python3
"""Parallel Moshi inference: a fleet of servers + a shared job queue.

The Moshi server holds a global session lock (one conversation at a time) and
the client streams in real time, so throughput = number of server instances.
This driver:

  1. starts `--num-servers` `moshi.server` processes, round-robin across GPUs
     (distinct ports from `--base-port`),
  2. builds one job per (conversation, speaker) — a temp dir holding a single
     symlink so `inference_moshi_dev_release.py` sees exactly one file,
  3. runs one worker per server, each popping jobs off a shared queue and
     invoking the client; output lands in `<output>/<conv>/speaker_K/` with a
     `.done` marker written only after the client exits 0 with the output
     present. Resume skips on the marker — a bare output.flac without `.done`
     is treated as partial and re-recorded (bless recordings made before the
     marker existed with `touch <dir>/.done`).

    python baselines/moshi/pipeline/run_fleet.py \
        --input /path/to/turnbench_audio/dev --output /path/to/moshi_out/dev \
        --num-servers 24 --num-gpus 8

Run with the same environment as the client (moshi venv: moshi, scipy,
websockets, soundfile, sphn).
"""
from __future__ import annotations

import argparse
import os
import queue
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
CLIENT = HERE / "inference_moshi_dev_release.py"
SPEAKER_FILES = ["speaker_1_audio.flac", "speaker_2_audio.flac"]


def collect_jobs(input_dir: Path, output_dir: Path) -> list[tuple[str, Path]]:
    """One job per (conversation, speaker) not yet completed, longest audio
    first so stragglers start early. Returns (job_name, input_flac)."""
    jobs = []
    for folder in sorted(input_dir.iterdir(), key=lambda p: int(p.name) if p.name.isdigit() else -1):
        if not folder.is_dir() or not folder.name.isdigit():
            continue
        for speaker_file in SPEAKER_FILES:
            flac = folder / speaker_file
            if not flac.exists():
                continue
            speaker = speaker_file.rsplit("_audio", 1)[0]
            if (output_dir / folder.name / speaker / ".done").exists():
                continue
            jobs.append((f"{folder.name}/{speaker}", flac))
    jobs.sort(key=lambda j: j[1].stat().st_size, reverse=True)
    return jobs


def wait_ready(port: int, proc: subprocess.Popen, timeout_s: float = 600) -> None:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"server on port {port} exited with {proc.returncode}")
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=1):
                return
        except OSError:
            time.sleep(2)
    raise RuntimeError(f"server on port {port} not ready after {timeout_s}s")


def worker(port: int, jobs: queue.Queue, output_dir: Path, work_root: Path,
           log_dir: Path, failures: list[str]) -> None:
    while True:
        try:
            job_name, flac = jobs.get_nowait()
        except queue.Empty:
            return
        # Single-file input dir: <work>/<port>/<conv>/<speaker_file> -> flac
        job_input = work_root / str(port)
        shutil.rmtree(job_input, ignore_errors=True)
        conv_dir = job_input / flac.parent.name
        conv_dir.mkdir(parents=True)
        (conv_dir / flac.name).symlink_to(flac)

        log_path = log_dir / f"job-{job_name.replace('/', '-')}.log"
        with open(log_path, "w") as log:
            rc = subprocess.call(
                [sys.executable, "-u", str(CLIENT), "--server_ip", f"127.0.0.1:{port}",
                 "--input", str(job_input), "--output", str(output_dir),
                 "--sleep-between", "0", "--post-input-wait", "120000"],
                stdout=log, stderr=subprocess.STDOUT,
            )
        done = (output_dir / job_name / "output.flac").exists()
        status = "ok" if (rc == 0 and done) else f"FAIL rc={rc} output={done}"
        print(f"[{time.strftime('%H:%M:%S')}] [{port}] {job_name}: {status}", flush=True)
        if rc == 0 and done:
            (output_dir / job_name / ".done").touch()
        else:
            failures.append(job_name)
        time.sleep(2)  # let the server settle before the next session


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--num-servers", type=int, default=24,
                    help="3/GPU keeps every stream at >=1x real-time on H100; "
                         "4/GPU measured 0.92x (tails would truncate)")
    ap.add_argument("--num-gpus", type=int, default=8)
    ap.add_argument("--base-port", type=int, default=8801)
    ap.add_argument("--log-dir", type=Path, default=Path("logs"))
    args = ap.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    args.log_dir.mkdir(parents=True, exist_ok=True)
    work_root = args.output / ".fleet_work"

    all_jobs = collect_jobs(args.input, args.output)
    print(f"{len(all_jobs)} jobs pending", flush=True)
    if not all_jobs:
        return

    n_servers = min(args.num_servers, len(all_jobs))
    servers: list[subprocess.Popen] = []
    try:
        for i in range(n_servers):
            port = args.base_port + i
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(i % args.num_gpus))
            log = open(args.log_dir / f"server-{port}.log", "w")
            servers.append(subprocess.Popen(
                [sys.executable, "-m", "moshi.server", "--port", str(port),
                 "--host", "127.0.0.1"],
                stdout=log, stderr=subprocess.STDOUT, env=env,
            ))
        print(f"started {n_servers} servers, waiting for readiness...", flush=True)
        for i in range(n_servers):
            wait_ready(args.base_port + i, servers[i])
        print("all servers ready", flush=True)

        job_queue: queue.Queue = queue.Queue()
        for job in all_jobs:
            job_queue.put(job)
        failures: list[str] = []
        threads = [
            threading.Thread(target=worker, args=(
                args.base_port + i, job_queue, args.output, work_root,
                args.log_dir, failures))
            for i in range(n_servers)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        print(f"done: {len(all_jobs) - len(failures)}/{len(all_jobs)} ok", flush=True)
        if failures:
            print("failed jobs: " + ", ".join(sorted(failures)), flush=True)
            sys.exit(1)
    finally:
        for proc in servers:
            if proc.poll() is None:
                proc.send_signal(signal.SIGTERM)
        for proc in servers:
            try:
                proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                proc.kill()
        shutil.rmtree(work_root, ignore_errors=True)


if __name__ == "__main__":
    main()
