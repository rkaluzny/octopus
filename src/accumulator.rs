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
use crate::features::{feature_index_table, FeatureIndexTable, ACCUMULATOR_SIZE};
use crate::nnue::{EvalBackend, NnueWeights};
use crate::movegen::{Move, MoveType};

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

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
    pub fn rebuild_from_board(&mut self, board: &Board, weights: &NnueWeights, backend: EvalBackend) {
        self.white.fill(0);
        self.black.fill(0);
        self.white_king_sq = board.king_square(Color::White);
        self.black_king_sq = board.king_square(Color::Black);
        let table = feature_index_table();

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
                    self.apply_piece_delta(color, piece, square, 1, weights, backend, table);
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
    pub fn apply_piece_deltas(
        &mut self,
        deltas: &[PieceDelta],
        weights: &NnueWeights,
        backend: EvalBackend,
        table: &FeatureIndexTable,
    ) {
        for delta in deltas {
            self.apply_piece_delta(delta.piece_color, delta.piece, delta.square, delta.sign, weights, backend, table);
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
        backend: EvalBackend,
        table: &FeatureIndexTable,
    ) {
        if piece == PieceType::King {
            return;
        }
        let delta = if sign >= 0 { 1 } else { -1 };
        self.apply_piece_delta_for_side(Color::White, piece_color, piece, square, delta, backend, weights, table);
        self.apply_piece_delta_for_side(Color::Black, piece_color, piece, square, delta, backend, weights, table);
    }

    #[inline(always)]
    pub fn apply_piece_delta_for_side(
        &mut self,
        side: Color,
        piece_color: Color,
        piece: PieceType,
        square: u8,
        delta: i32,
        backend: EvalBackend,
        weights: &NnueWeights,
        table: &FeatureIndexTable,
    ) {
        if piece == PieceType::King {
            return;
        }
        let king_sq = self.king_square(side) as usize;
        self.apply_piece_delta_for_side_with_king_sq(
            side,
            king_sq as u8,
            piece_color,
            piece,
            square,
            delta,
            backend,
            weights,
            table,
        );
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
        backend: EvalBackend,
        weights: &NnueWeights,
        table: &FeatureIndexTable,
    ) {
        if piece == PieceType::King {
            return;
        }

        let king_sq = king_sq as usize;
        let feature = table[side as usize][king_sq][piece_color as usize][piece as usize]
            [square as usize] as usize;
        let feature_weights = weights.feature_weights_for_feature(feature);
        let acc = match side {
            Color::White => &mut self.white,
            Color::Black => &mut self.black,
        };

        // Dispatch based on pre-selected backend
        match backend {
            #[cfg(target_arch = "x86_64")]
            EvalBackend::Avx2 => {
                unsafe {
                    apply_piece_delta_avx2(acc, feature_weights, delta);
                }
                return;
            }
            #[cfg(target_arch = "x86_64")]
            EvalBackend::Sse2 => {
                unsafe {
                    apply_piece_delta_sse2(acc, feature_weights, delta);
                }
                return;
            }
            _ => {}
        }

        // Fallback to scalar
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

    #[inline(always)]
    pub fn rebuild_side_after_king_move(
        &mut self,
        board: &Board,
        side: Color,
        king_sq: u8,
        moving_color: Color,
        mv: &Move,
        weights: &NnueWeights,
        backend: EvalBackend,
    ) {
        let table = feature_index_table();
        let acc = match side {
            Color::White => &mut self.white,
            Color::Black => &mut self.black,
        };
        acc.fill(0);

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
                let piece_color = if color_idx == 0 { Color::White } else { Color::Black };
                let mut pieces = board.bitboards[piece_idx] & board.color_bitboards[color_idx];
                while pieces != 0 {
                    let square = pieces.trailing_zeros() as u8;
                    if mv.capture.is_some() && piece_color != moving_color && square == mv.to as u8 {
                        pieces &= pieces - 1;
                        continue;
                    }
                    if piece == PieceType::Rook && piece_color == moving_color {
                        match mv.move_type {
                            MoveType::KingCastle if square == 63 && moving_color == Color::White => {
                                pieces &= pieces - 1;
                                continue;
                            }
                            MoveType::KingCastle if square == 7 && moving_color == Color::Black => {
                                pieces &= pieces - 1;
                                continue;
                            }
                            MoveType::QueenCastle if square == 56 && moving_color == Color::White => {
                                pieces &= pieces - 1;
                                continue;
                            }
                            MoveType::QueenCastle if square == 0 && moving_color == Color::Black => {
                                pieces &= pieces - 1;
                                continue;
                            }
                            _ => {}
                        }
                    }
                    self.apply_piece_delta_for_side_with_king_sq(
                        side,
                        king_sq,
                        piece_color,
                        piece,
                        square,
                        1,
                        backend,
                        weights,
                        table,
                    );
                    pieces &= pieces - 1;
                }
            }
        }

        match mv.move_type {
            MoveType::KingCastle => {
                let (rook_to, rook_color) = if moving_color == Color::White {
                    (61u8, Color::White)
                } else {
                    (5u8, Color::Black)
                };
                self.apply_piece_delta_for_side_with_king_sq(
                    side,
                    king_sq,
                    rook_color,
                    PieceType::Rook,
                    rook_to,
                    1,
                    backend,
                    weights,
                    table,
                );
            }
            MoveType::QueenCastle => {
                let (rook_to, rook_color) = if moving_color == Color::White {
                    (59u8, Color::White)
                } else {
                    (3u8, Color::Black)
                };
                self.apply_piece_delta_for_side_with_king_sq(
                    side,
                    king_sq,
                    rook_color,
                    PieceType::Rook,
                    rook_to,
                    1,
                    backend,
                    weights,
                    table,
                );
            }
            _ => {}
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse2")]
unsafe fn apply_piece_delta_sse2(acc: &mut [i32; ACCUMULATOR_SIZE], feature_weights: &[i8], delta: i32) {
    let delta_vec = _mm_set1_epi32(delta);
    let zero = _mm_setzero_si128();

    for i in (0..ACCUMULATOR_SIZE).step_by(16) {
        let w8 = _mm_loadu_si128(feature_weights.as_ptr().add(i) as *const __m128i);
        let sign = _mm_cmpgt_epi8(zero, w8);
        let w16_lo = _mm_unpacklo_epi8(w8, sign);
        let w16_hi = _mm_unpackhi_epi8(w8, sign);

        let sign_lo = _mm_srai_epi16(w16_lo, 15);
        let sign_hi = _mm_srai_epi16(w16_hi, 15);

        let w32_0 = _mm_unpacklo_epi16(w16_lo, sign_lo);
        let w32_1 = _mm_unpackhi_epi16(w16_lo, sign_lo);
        let w32_2 = _mm_unpacklo_epi16(w16_hi, sign_hi);
        let w32_3 = _mm_unpackhi_epi16(w16_hi, sign_hi);

        let w32_0 = _mm_mullo_epi32(w32_0, delta_vec);
        let w32_1 = _mm_mullo_epi32(w32_1, delta_vec);
        let w32_2 = _mm_mullo_epi32(w32_2, delta_vec);
        let w32_3 = _mm_mullo_epi32(w32_3, delta_vec);

        let acc_ptr = acc.as_mut_ptr().add(i);
        let a0 = _mm_loadu_si128(acc_ptr as *const __m128i);
        let a1 = _mm_loadu_si128(acc_ptr.add(4) as *const __m128i);
        let a2 = _mm_loadu_si128(acc_ptr.add(8) as *const __m128i);
        let a3 = _mm_loadu_si128(acc_ptr.add(12) as *const __m128i);

        let a0 = _mm_add_epi32(a0, w32_0);
        let a1 = _mm_add_epi32(a1, w32_1);
        let a2 = _mm_add_epi32(a2, w32_2);
        let a3 = _mm_add_epi32(a3, w32_3);

        _mm_storeu_si128(acc_ptr as *mut __m128i, a0);
        _mm_storeu_si128(acc_ptr.add(4) as *mut __m128i, a1);
        _mm_storeu_si128(acc_ptr.add(8) as *mut __m128i, a2);
        _mm_storeu_si128(acc_ptr.add(12) as *mut __m128i, a3);
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn apply_piece_delta_avx2(acc: &mut [i32; ACCUMULATOR_SIZE], feature_weights: &[i8], delta: i32) {
    let delta_vec = _mm256_set1_epi32(delta);

    for i in (0..ACCUMULATOR_SIZE).step_by(8) {
        let w8 = _mm_loadl_epi64(feature_weights.as_ptr().add(i) as *const __m128i);
        let w32 = _mm256_cvtepi8_epi32(w8);
        let w32 = _mm256_mullo_epi32(w32, delta_vec);

        let acc_ptr = acc.as_mut_ptr().add(i);
        let a = _mm256_loadu_si256(acc_ptr as *const __m256i);
        let a = _mm256_add_epi32(a, w32);
        _mm256_storeu_si256(acc_ptr as *mut __m256i, a);
    }
}
