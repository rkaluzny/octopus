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

# This script creates selfplay games
import subprocess
import threading
import queue
import time
import random
import hashlib
import chess
import chess.pgn
import signal

# CONFIG

ENGINE_PATH = "ENGINE_PATH"
OUTPUT_FILE = "selfplay11.txt"

NUM_THREADS = 4
GAMES_PER_THREAD = 700

MOVETIME_MS = 60
DEEPER_MOVETIME_MS = 200

MAX_PLIES = 100
MIN_PLY_TO_SAVE = 12
SAMPLE_INTERVAL = 1

RANDOM_MOVE_PROB = 0

MIN_EVAL = -700
MAX_EVAL = 700

STABILITY_THRESHOLD = 100

WRITE_BATCH_SIZE = 500

OPENING_BOOK_PGN = "opening.pgn"  # Path to PGN file for opening book, e.g. "openings.pgn"
OPENING_BOOK_PLIES = 12  # Number of plies to play from opening book

stop_event = threading.Event()


# OPENING BOOK

def load_opening_book(pgn_path):
    games = []
    try:
        with open(pgn_path, "r") as f:
            while True:
                game = chess.pgn.read_game(f)
                if game is None:
                    break
                moves = []
                node = game
                while node.variations:
                    node = node.variation(0)
                    moves.append(node.move.uci())
                if moves:
                    games.append(moves)
        print(f"Loaded {len(games)} opening book games from {pgn_path}")
    except Exception as e:
        print(f"Failed to load opening book: {e}")
        return []
    return games


def get_random_opening(book_games, max_plies):
    if not book_games:
        return []
    game_moves = random.choice(book_games)
    plies_to_play = min(max_plies, len(game_moves))
    return game_moves[:plies_to_play]


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


def go_eval(p, fen, movetime):
    send(p, f"position fen {fen}")
    send(p, f"go movetime {movetime}")

    eval_cp = None
    bestmove = None
    start = time.time()

    while True:
        if stop_event.is_set():
            return None, None

        if time.time() - start > 10:
            return None, None

        line = p.stdout.readline()

        if "score cp" in line:
            try:
                parts = line.split()
                idx = parts.index("cp")
                eval_cp = int(parts[idx + 1])
            except:
                pass

        if "bestmove" in line:
            bestmove = line.split()[1]
            break

    return eval_cp, bestmove


# UTILS

def hash_fen(fen):
    return hashlib.md5(fen.encode()).hexdigest()


def should_save(fen, eval_cp, ply):
    if ply < MIN_PLY_TO_SAVE:
        return False

    if eval_cp is None:
        return False

    if abs(eval_cp) > MAX_EVAL:
        return False

    return True


# WORKER

def restart_engine(thread_id):
    print(f"[T{thread_id}] Restarting engine...")
    try:
        return start_engine()
    except (RuntimeError, TimeoutError, InterruptedError) as e:
        print(f"[T{thread_id}] Failed to restart engine: {e}")
        return None


def worker(thread_id, out_queue, opening_book_games=None):
    engine = restart_engine(thread_id)
    if engine is None:
        return

    seen = set()
    saved = 0
    skipped = 0
    unstable = 0

    for game in range(GAMES_PER_THREAD):
        if stop_event.is_set():
            break

        print(f"[T{thread_id}] Game {game+1}/{GAMES_PER_THREAD}")

        board = chess.Board()
        moves = []

        if opening_book_games:
            opening_moves = get_random_opening(opening_book_games, OPENING_BOOK_PLIES)
            for uci_move in opening_moves:
                board.push_uci(uci_move)
                moves.append(uci_move)
            print(f"[T{thread_id}] Applied {len(opening_moves)} opening book moves")

        ply = len(moves)

        while ply < MAX_PLIES and not stop_event.is_set():
            if moves:
                position_cmd = "position startpos moves " + " ".join(moves)
            else:
                position_cmd = "position startpos"

            try:
                send(engine, position_cmd)
            except (RuntimeError, TimeoutError, InterruptedError, OSError):
                engine = restart_engine(thread_id)
                if engine is None:
                    return
                continue

            fen = board.fen()

            eval1, bestmove = go_eval(engine, fen, MOVETIME_MS)
            if stop_event.is_set():
                break

            if eval1 is None or bestmove is None:
                print(f"[T{thread_id}] Engine failed, restarting...")
                engine = restart_engine(thread_id)
                if engine is None:
                    return
                break

            eval2, _ = go_eval(engine, fen, MOVETIME_MS)

            if eval2 is None:
                print(f"[T{thread_id}] Engine failed on eval2, restarting...")
                engine = restart_engine(thread_id)
                if engine is None:
                    return
                break

            if abs(eval1 - eval2) > STABILITY_THRESHOLD:
                eval3, _ = go_eval(engine, fen, DEEPER_MOVETIME_MS)

                if eval3 is None or abs(eval2 - eval3) > STABILITY_THRESHOLD:
                    unstable += 1
                    break
                eval_final = eval3
            else:
                eval_final = eval2

            if should_save(fen, eval_final, ply) and ply % SAMPLE_INTERVAL == 0:
                h = hash_fen(fen)
                if h not in seen:
                    seen.add(h)
                    out_queue.put((fen, eval_final))
                    saved += 1
                else:
                    skipped += 1

            if random.random() < RANDOM_MOVE_PROB:
                if stop_event.is_set():
                    break
                try:
                    send(engine, position_cmd)
                    send(engine, "go movetime 10")
                except (RuntimeError, TimeoutError, InterruptedError, OSError):
                    engine = restart_engine(thread_id)
                    if engine is None:
                        return
                    break

                start = time.time()
                while True:
                    if stop_event.is_set():
                        break
                    if time.time() - start > 5:
                        break
                    line = engine.stdout.readline()
                    if "bestmove" in line:
                        bestmove = line.split()[1]
                        break

            if bestmove is None or bestmove == "(none)":
                break

            board.push_uci(bestmove)
            moves.append(bestmove)
            ply += 1

        if game % 5 == 0:
            print(f"[T{thread_id}] saved={saved} skipped={skipped} unstable={unstable}")

    try:
        engine.kill()
    except:
        pass


# WRITER

def writer(out_queue):
    buffer = []
    total = 0

    def flush_buffer(f):
        nonlocal buffer, total
        for fen, eval_cp in buffer:
            f.write(f"{fen} | {eval_cp}\n")
        f.flush()
        total += len(buffer)
        print(f"[Writer] total={total}")
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

        if buffer:
            flush_buffer(f)

    print(f"[Writer] FINAL total={total}")


# MAIN

def signal_handler(sig, frame):
    print("\n\n[Main] Ctrl+C received, stopping gracefully...")
    stop_event.set()


def main():
    signal.signal(signal.SIGINT, signal_handler)
    start = time.time()

    opening_book_games = []
    if OPENING_BOOK_PGN:
        opening_book_games = load_opening_book(OPENING_BOOK_PGN)

    out_queue = queue.Queue(maxsize=10000)

    threads = []
    for i in range(NUM_THREADS):
        t = threading.Thread(target=worker, args=(i, out_queue, opening_book_games))
        t.start()
        threads.append(t)

    w = threading.Thread(target=writer, args=(out_queue,))
    w.start()

    try:
        for t in threads:
            t.join()
    except KeyboardInterrupt:
        stop_event.set()

    if stop_event.is_set():
        for t in threads:
            t.join(timeout=2)

    out_queue.put(None)
    w.join(timeout=5)

    print(f"Done in {time.time() - start:.2f}s")


if __name__ == "__main__":
    main()