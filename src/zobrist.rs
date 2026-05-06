// =============================================================================
// Octopus — UCI-compatible chess engine written in Rust
// Copyright (c) 2026 Robin Kaluzny
// SPDX-License-Identifier: MIT
//
// This file is part of the Octopus project.
//
// Licensed under the MIT License; you may not use this file except in
// compliance with the License. See the LICENSE file in the project root
// for full license information.
//
// =============================================================================

use crate::board::{Color, PieceType, Square};
use lazy_static::lazy_static;
use rand::{Rng, SeedableRng};

const ZOBRIST_SEED: u64 = 10703728;

struct ZobristKeys {
    pieces: [[[u64; 64]; 6]; 2],
    side_to_move: u64,
    castling: [u64; 16],
    en_passant: [u64; 8],
}

lazy_static! {
    static ref ZOBRIST: ZobristKeys = {
        let mut rng = rand::rngs::StdRng::seed_from_u64(ZOBRIST_SEED);
        let mut keys = ZobristKeys {
            pieces: [[[0; 64]; 6]; 2],
            side_to_move: rng.gen::<u64>(),
            castling: [0; 16],
            en_passant: [0; 8],
        };

        for i in 0..16 {
            keys.castling[i] = rng.gen::<u64>();
        }
        for i in 0..8 {
            keys.en_passant[i] = rng.gen::<u64>();
        }
        for c in 0..2 {
            for p in 0..6 {
                for s in 0..64 {
                    keys.pieces[c][p][s] = rng.gen::<u64>();
                }
            }
        }
        keys
    };
}

pub fn piece_key(piece: PieceType, color: Color, sq: u8) -> u64 {
    ZOBRIST.pieces[color as usize][piece as usize][sq as usize]
}

pub fn side_to_move_key() -> u64 {
    ZOBRIST.side_to_move
}

pub fn castling_key(rights: u8) -> u64 {
    ZOBRIST.castling[rights as usize]
}

pub fn en_passant_key(square: Square) -> Option<u64> {
    if square == Square::NoSquare {
        return None;
    }
    let file = (square as u8) % 8;
    Some(ZOBRIST.en_passant[file as usize])
}
