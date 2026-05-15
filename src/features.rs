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

use std::sync::OnceLock;

use crate::board::{Color, PieceType};

pub const INPUT_FEATURES: usize = 768;
pub const ACCUMULATOR_SIZE: usize = 512;
pub const HIDDEN_SIZE: usize = 128;
pub const KING_SQUARES: usize = 64;
pub const PERSPECTIVES: usize = 2;

pub type FeatureIndexTable =
    [[[[[u16; KING_SQUARES]; 6]; PERSPECTIVES]; KING_SQUARES]; PERSPECTIVES];

static FEATURE_INDEX_TABLE: OnceLock<FeatureIndexTable> = OnceLock::new();

#[inline(always)]
pub fn feature_index_table() -> &'static FeatureIndexTable {
    FEATURE_INDEX_TABLE.get_or_init(build_feature_index_table)
}

#[inline(always)]
pub fn feature_index(
    stm: Color,
    king_square: u8,
    piece_color: Color,
    piece: PieceType,
    square: u8,
) -> usize {
    feature_index_table()[stm as usize][king_square as usize][piece_color as usize][piece as usize]
        [square as usize] as usize
}

fn build_feature_index_table() -> FeatureIndexTable {
    std::array::from_fn(|stm_idx| {
        std::array::from_fn(|king_sq| {
            std::array::from_fn(|piece_color_idx| {
                std::array::from_fn(|piece_idx| {
                    std::array::from_fn(|sq| {
                        feature_index_raw(
                            stm_idx as usize,
                            king_sq as u8,
                            piece_color_idx as usize,
                            piece_idx as usize,
                            sq as u8,
                        )
                    })
                })
            })
        })
    })
}

#[inline(always)]
fn feature_index_raw(
    stm_idx: usize,
    king_square: u8,
    piece_color_idx: usize,
    piece_idx: usize,
    square: u8,
) -> u16 {
    let rel_square = relative_square(stm_idx as u8, king_square, square) as usize;
    let side = if piece_color_idx == stm_idx { 0 } else { 1 };
    (piece_idx * 64 + rel_square + side * 384) as u16
}

#[inline(always)]
fn relative_square(stm_idx: u8, king_square: u8, square: u8) -> u8 {
    let king_rank = king_square / 8;
    let king_file = king_square % 8;
    let square_rank = square / 8;
    let square_file = square % 8;

    let (rank, file) = if stm_idx == 0 {
        (
            (square_rank + 7 - king_rank) & 7,
            (square_file + 7 - king_file) & 7,
        )
    } else {
        (
            (king_rank + 7 - square_rank) & 7,
            (king_file + 7 - square_file) & 7,
        )
    };

    rank * 8 + file
}
