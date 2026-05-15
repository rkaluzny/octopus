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

# Convert the txt files into the binary format
import chess
import chess.polyglot
import struct

INPUT_FILE = "data_txt/selfplay17.txt"
OUTPUT_FILE = "../data/selfplay17.bin"

# Format: hash(8) + 6 white piece BBs(8*6) + 6 black piece BBs(8*6) + eval(4) + stm(1) + castling(1) + ep(1) + pad(1)
STRUCT_FORMAT = "<QQQQQQQQQQQQQiBBBB"  # 13 Qs + i + 4 Bs = 112 bytes

def encode_castling(board: chess.Board) -> int:
    rights = 0
    if board.has_kingside_castling_rights(chess.WHITE):
        rights |= 1
    if board.has_queenside_castling_rights(chess.WHITE):
        rights |= 2
    if board.has_kingside_castling_rights(chess.BLACK):
        rights |= 4
    if board.has_queenside_castling_rights(chess.BLACK):
        rights |= 8
    return rights


def main():
    total = 0
    skipped = 0

    with open(INPUT_FILE, "r") as f_in, open(OUTPUT_FILE, "wb") as f_out:
        for i, line in enumerate(f_in, 1):
            line = line.strip()

            if not line or "|" not in line:
                skipped += 1
                continue

            try:
                fen_part, eval_part = line.split("|", 1)
                fen = fen_part.strip()
                eval_cp = int(eval_part.strip())

                board = chess.Board(fen)

                # Per-piece bitboards for white (pawn, knight, bishop, rook, queen, king)
                white_pieces = []
                for piece_type in [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.KING]:
                    bb = int(board.pieces(piece_type, chess.WHITE))
                    white_pieces.append(bb)
                
                # Per-piece bitboards for black
                black_pieces = []
                for piece_type in [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.KING]:
                    bb = int(board.pieces(piece_type, chess.BLACK))
                    black_pieces.append(bb)

                stm = 0 if board.turn == chess.WHITE else 1
                castling = encode_castling(board)
                ep = board.ep_square if board.ep_square is not None else 255

                # ✅ FIXED
                hash_key = chess.polyglot.zobrist_hash(board)

                packed = struct.pack(
                    STRUCT_FORMAT,
                    hash_key,
                    *white_pieces,  # 6 white piece bitboards
                    *black_pieces,  # 6 black piece bitboards
                    eval_cp,
                    stm,
                    castling,
                    ep,
                    0
                )

                f_out.write(packed)
                total += 1

                if total % 100000 == 0:
                    print(f"Converted: {total} | Skipped: {skipped}")

            except Exception as e:
                skipped += 1

                # Print first few errors for debugging
                if skipped < 10:
                    print(f"[ERROR] Line {i}: {e}")
                    print(line)

    print("\n=== DONE ===")
    print(f"Converted: {total}")
    print(f"Skipped: {skipped}")


if __name__ == "__main__":
    main()