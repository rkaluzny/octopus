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

import chess
import chess.pgn

INPUT_PGN = "input.pgn"
OUTPUT_FILE = "positions.txt"

SKIP_FULL_MOVES = 4
SAMPLE_INTERVAL = 3
MAX_POSITIONS = 0

def main():
    seen = set()
    total_games = 0
    total_positions = 0

    with open(INPUT_PGN, "r", encoding="utf-8", errors="ignore") as pgn, \
         open(OUTPUT_FILE, "w") as out:

        while True:
            game = chess.pgn.read_game(pgn)
            if game is None:
                break

            total_games += 1
            board = game.board()

            ply = 0
            for move in game.mainline_moves():
                board.push(move)
                ply += 1

                # skip opening
                if ply < SKIP_FULL_MOVES * 2:
                    continue

                if ply % SAMPLE_INTERVAL != 0:
                    continue

                fen = board.fen()

                if fen not in seen:
                    seen.add(fen)
                    out.write(fen + "\n")
                    total_positions += 1

                    if MAX_POSITIONS and total_positions >= MAX_POSITIONS:
                        print("Reached max positions.")
                        return

            if total_games % 1000 == 0:
                print(f"Games: {total_games}, Positions: {total_positions}")

    print("\n=== Done ===")
    print(f"Games processed: {total_games}")
    print(f"Positions saved: {total_positions}")

if __name__ == "__main__":
    main()