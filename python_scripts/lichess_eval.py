# =============================================================================
# Octopus — UCI-compatible chess engine written in Rust
# Copyright (c) 2026 Robin Kaluzny
# SPDX-License-Identifier: MIT
#
# This file is part of the Octopus project.
#
# Licensed under the MIT License; you may not use this file except in
# compliance with the License. See the LICENSE file in the project root
# for full license information.
#
# =============================================================================

import subprocess
import threading
import queue
import time
import random
import signal

ENGINE_PATH = "ENGINE_PATH"
INPUT_FILE = "positions.txt"
OUTPUT_FILE = "lichess8.txt"

NUM_THREADS = 4
MOVETIME_MS = 60
DEEPER_MOVETIME_MS = 200

MIN_EVAL = -500
MAX_EVAL = 500

STABILITY_THRESHOLD = 110
WRITE_BATCH_SIZE = 500

QUEUE_MAXSIZE = 10000

stop_event = threading.Event()


# ENGINE

def start_engine():
    p = subprocess.Popen(
        ENGINE_PATH,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
        bufsize=1,
    )

    send(p, "uci")
    wait_for(p, "uciok", timeout=10)

    send(p, "isready")
    wait_for(p, "readyok", timeout=10)

    return p


def send(p, cmd):
    if stop_event.is_set():
        raise InterruptedError("Stopped by user")
    p.stdin.write(cmd + "\n")
    p.stdin.flush()


def wait_for(p, token, timeout=5):
    start = time.time()
    while True:
        if stop_event.is_set():
            raise InterruptedError("Stopped by user")
        line = p.stdout.readline()
        if not line:
            raise RuntimeError("Engine stopped responding")
        if time.time() - start > timeout:
            raise TimeoutError(f"Engine timeout waiting for {token}")
        if token in line:
            return


def evaluate_once(p, fen, movetime):
    send(p, f"position fen {fen}")
    send(p, f"go movetime {movetime}")

    eval_cp = None
    start = time.time()

    while True:
        if stop_event.is_set():
            return None

        # timeout safety
        if time.time() - start > 10:
            return None

        line = p.stdout.readline()
        if not line:
            continue

        if "score cp" in line:
            try:
                parts = line.split()
                idx = parts.index("cp")
                eval_cp = int(parts[idx + 1])
            except:
                pass

        elif "bestmove" in line:
            break

    return eval_cp


# WORKER

def restart_engine(thread_id):
    print(f"[T{thread_id}] Restarting engine...")
    try:
        return start_engine()
    except (RuntimeError, TimeoutError, InterruptedError) as e:
        print(f"[T{thread_id}] Failed to restart engine: {e}")
        return None


def worker(fen_queue, out_queue, stats, thread_id):
    engine = restart_engine(thread_id)
    if engine is None:
        stats[thread_id] = (0, 0, 0, 0)
        return

    processed = 0
    saved = 0
    skipped = 0
    unstable = 0

    while not stop_event.is_set():
        try:
            fen = fen_queue.get(timeout=0.5)
        except queue.Empty:
            if stop_event.is_set():
                break
            continue

        eval1 = evaluate_once(engine, fen, MOVETIME_MS)
        if eval1 is None:
            if stop_event.is_set():
                break
            print(f"[T{thread_id}] Engine failed on eval1, restarting...")
            engine = restart_engine(thread_id)
            if engine is None:
                break
            skipped += 1
            continue

        eval2 = evaluate_once(engine, fen, MOVETIME_MS)

        if eval2 is None:
            if stop_event.is_set():
                break
            print(f"[T{thread_id}] Engine failed on eval2, restarting...")
            engine = restart_engine(thread_id)
            if engine is None:
                break
            skipped += 1
            continue

        if abs(eval1 - eval2) > STABILITY_THRESHOLD:
            eval3 = evaluate_once(engine, fen, DEEPER_MOVETIME_MS)

            if eval3 is None:
                if stop_event.is_set():
                    break
                print(f"[T{thread_id}] Engine failed on eval3, restarting...")
                engine = restart_engine(thread_id)
                if engine is None:
                    break
                unstable += 1
                continue

            if abs(eval2 - eval3) > STABILITY_THRESHOLD:
                unstable += 1
                continue
            final_eval = eval3
        else:
            final_eval = eval2

        if final_eval < MIN_EVAL or final_eval > MAX_EVAL:
            skipped += 1
            continue

        keep = True
        if abs(final_eval) > 800:
            keep = random.random() < 0.4
        elif abs(final_eval) < 100:
            keep = random.random() < 0.7

        if keep:
            out_queue.put((fen, final_eval))
            saved += 1

        processed += 1

        if processed % 200 == 0:
            print(f"[T{thread_id}] processed={processed} saved={saved} unstable={unstable}")

    try:
        engine.kill()
    except:
        pass
    stats[thread_id] = (processed, saved, skipped, unstable)


# WRITER

def writer_thread(out_queue):
    buffer = []
    total_written = 0

    def flush_buffer(f):
        nonlocal buffer, total_written
        for fen, eval_cp in buffer:
            f.write(f"{fen} | {eval_cp}\n")
        f.flush()
        total_written += len(buffer)
        print(f"[Writer] written={total_written}")
        buffer.clear()

    with open(OUTPUT_FILE, "w") as f:
        while True:
            try:
                item = out_queue.get(timeout=0.5)
            except queue.Empty:
                if stop_event.is_set():
                    break
                continue

            if item is None:
                break

            buffer.append(item)

            if len(buffer) >= WRITE_BATCH_SIZE:
                flush_buffer(f)

        # final flush
        if buffer:
            flush_buffer(f)

    print(f"[Writer] FINAL written={total_written}")


# LOADER

def loader_thread(fen_queue):
    total_loaded = 0

    try:
        with open(INPUT_FILE) as f:
            for line in f:
                if stop_event.is_set():
                    break
                fen = line.strip()
                if fen:
                    fen_queue.put(fen)
                    total_loaded += 1

                    if total_loaded % 10000 == 0:
                        print(f"[Loader] loaded={total_loaded}")
    except InterruptedError:
        pass

    print(f"[Loader] DONE total={total_loaded}")


# MAIN

def signal_handler(sig, frame):
    print("\n\n[Main] Ctrl+C received, stopping gracefully...")
    stop_event.set()


def main():
    signal.signal(signal.SIGINT, signal_handler)
    start_time = time.time()

    fen_queue = queue.Queue(maxsize=QUEUE_MAXSIZE)
    out_queue = queue.Queue(maxsize=QUEUE_MAXSIZE)

    stats = {}

    # loader thread (prevents startup freeze)
    loader = threading.Thread(target=loader_thread, args=(fen_queue,))
    loader.start()

    # writer
    writer = threading.Thread(target=writer_thread, args=(out_queue,))
    writer.start()

    # workers
    threads = []
    for i in range(NUM_THREADS):
        t = threading.Thread(target=worker, args=(fen_queue, out_queue, stats, i))
        t.start()
        threads.append(t)

    try:
        loader.join()
    except KeyboardInterrupt:
        stop_event.set()

    if not stop_event.is_set():
        for t in threads:
            t.join()
    else:
        # Wait briefly for threads to finish
        for t in threads:
            t.join(timeout=2)

    out_queue.put(None)
    writer.join(timeout=5)

    # summary
    total_processed = sum(s[0] for s in stats.values())
    total_saved = sum(s[1] for s in stats.values())
    total_skipped = sum(s[2] for s in stats.values())
    total_unstable = sum(s[3] for s in stats.values())

    elapsed = time.time() - start_time

    print("\n=== SUMMARY ===")
    print(f"Processed: {total_processed}")
    print(f"Saved: {total_saved}")
    print(f"Skipped: {total_skipped}")
    print(f"Unstable: {total_unstable}")
    print(f"Time: {elapsed:.2f}s")
    if total_processed > 0:
        print(f"Speed: {total_processed / elapsed:.2f} pos/sec")


if __name__ == "__main__":
    main()