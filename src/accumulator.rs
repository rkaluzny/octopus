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

// NNUE accumulator state management with incremental updates.

use crate::board::{Board, Color, PieceType};
use crate::features::{feature_index_table, ACCUMULATOR_SIZE};
use crate::nnue::NnueWeights;

#[repr(align(64))]
#[derive(Clone)]
pub struct NnueState {
    pub white: [i32; ACCUMULATOR_SIZE],
    pub black: [i32; ACCUMULATOR_SIZE],
    pub white_king_sq: u8,
    pub black_king_sq: u8,
}

#[derive(Clone)]
pub struct NnueUndo {
    pub previous_state: Option<NnueState>,
    pub deltas: [PieceDelta; 8],
    pub delta_len: u8,
}

impl Default for NnueState {
    fn default() -> Self {
        Self {
            white: [0; ACCUMULATOR_SIZE],
            black: [0; ACCUMULATOR_SIZE],
            white_king_sq: 0,
            black_king_sq: 0,
        }
    }
}

#[derive(Copy, Clone)]
pub struct PieceDelta {
    pub piece_color: Color,
    pub piece: PieceType,
    pub square: u8,
    pub sign: i32,
}

impl Default for PieceDelta {
    fn default() -> Self {
        Self {
            piece_color: Color::White,
            piece: PieceType::Pawn,
            square: 0,
            sign: 0,
        }
    }
}

impl NnueState {
    pub fn rebuild_from_board(&mut self, board: &Board, weights: &NnueWeights) {
        self.white.fill(0);
        self.black.fill(0);
        self.white_king_sq = board.king_square(Color::White);
        self.black_king_sq = board.king_square(Color::Black);

        for piece_idx in 0..6 {
            let piece = match piece_idx {
                0 => PieceType::Pawn,
                1 => PieceType::Knight,
                2 => PieceType::Bishop,
                3 => PieceType::Rook,
                4 => PieceType::Queen,
                _ => PieceType::King,
            };
            if piece == PieceType::King {
                continue;
            }

            for color_idx in 0..2 {
                let color = if color_idx == 0 { Color::White } else { Color::Black };
                let mut pieces = board.bitboards[piece_idx] & board.color_bitboards[color_idx];
                while pieces != 0 {
                    let square = pieces.trailing_zeros() as u8;
                    self.apply_piece_delta(color, piece, square, 1, weights);
                    pieces &= pieces - 1;
                }
            }
        }
    }

    pub fn clear(&mut self) {
        self.white.fill(0);
        self.black.fill(0);
        self.white_king_sq = 0;
        self.black_king_sq = 0;
    }

    #[inline(always)]
    pub fn apply_piece_deltas(&mut self, deltas: &[PieceDelta], weights: &NnueWeights) {
        for delta in deltas {
            self.apply_piece_delta(delta.piece_color, delta.piece, delta.square, delta.sign, weights);
        }
    }

    #[inline(always)]
    pub fn apply_piece_delta(
        &mut self,
        piece_color: Color,
        piece: PieceType,
        square: u8,
        sign: i32,
        weights: &NnueWeights,
    ) {
        if piece == PieceType::King {
            return;
        }
        let delta = if sign >= 0 { 1 } else { -1 };
        self.apply_piece_delta_for_side(Color::White, piece_color, piece, square, delta, weights);
        self.apply_piece_delta_for_side(Color::Black, piece_color, piece, square, delta, weights);
    }

    #[inline(always)]
    pub fn apply_piece_delta_for_side(
        &mut self,
        side: Color,
        piece_color: Color,
        piece: PieceType,
        square: u8,
        delta: i32,
        weights: &NnueWeights,
    ) {
        if piece == PieceType::King {
            return;
        }
        let king_sq = self.king_square(side) as usize;
        self.apply_piece_delta_for_side_with_king_sq(side, king_sq as u8, piece_color, piece, square, delta, weights);
    }

    #[inline(always)]
    pub fn apply_piece_delta_for_side_with_king_sq(
        &mut self,
        side: Color,
        king_sq: u8,
        piece_color: Color,
        piece: PieceType,
        square: u8,
        delta: i32,
        weights: &NnueWeights,
    ) {
        if piece == PieceType::King {
            return;
        }

        let king_sq = king_sq as usize;
        let table = feature_index_table();
        let feature = table[side as usize][king_sq][piece_color as usize][piece as usize]
            [square as usize] as usize;
        let feature_weights = weights.feature_weights_for_feature(feature);
        let acc = match side {
            Color::White => &mut self.white,
            Color::Black => &mut self.black,
        };

        for i in 0..ACCUMULATOR_SIZE {
            unsafe {
                *acc.get_unchecked_mut(i) += delta * *feature_weights.get_unchecked(i) as i32;
            }
        }
    }

    #[inline(always)]
    pub fn current(&self, side_to_move: Color) -> &[i32; ACCUMULATOR_SIZE] {
        match side_to_move {
            Color::White => &self.white,
            Color::Black => &self.black,
        }
    }

    #[inline(always)]
    pub fn king_square(&self, side: Color) -> u8 {
        match side {
            Color::White => self.white_king_sq,
            Color::Black => self.black_king_sq,
        }
    }
}
