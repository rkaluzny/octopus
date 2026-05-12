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

use crate::attacks;
use crate::board::{Board, Color, PieceType, Bitboard};
use std::sync::OnceLock;

// A Score struct holds Middlegame and Endgame values respectively.
#[derive(Copy, Clone, Default, Debug)]
struct Score(i32, i32);

impl std::ops::AddAssign for Score {
    fn add_assign(&mut self, other: Self) {
        self.0 += other.0;
        self.1 += other.1;
    }
}

impl std::ops::SubAssign for Score {
    fn sub_assign(&mut self, other: Self) {
        self.0 -= other.0;
        self.1 -= other.1;
    }
}

impl std::ops::Add for Score {
    type Output = Self;
    fn add(self, other: Self) -> Self::Output {
        Score(self.0 + other.0, self.1 + other.1)
    }
}

impl std::ops::Sub for Score {
    type Output = Self;
    fn sub(self, other: Self) -> Self::Output {
        Score(self.0 - other.0, self.1 - other.1)
    }
}

impl std::ops::Mul<i32> for Score {
    type Output = Self;
    fn mul(self, rhs: i32) -> Self::Output {
        Score(self.0 * rhs, self.1 * rhs)
    }
}

/// Helper const function to create a Score tuple.
const fn m(mg: i32, eg: i32) -> Score {
    Score(mg, eg)
}

// --- CONSTANTS ---

// Piece values: (MG, EG)
const PAWN_VALUE: Score = m(128, 213);
const KNIGHT_VALUE: Score = m(780, 850); 
const BISHOP_VALUE: Score = m(820, 890); 
const ROOK_VALUE: Score = m(1273, 1378);
const QUEEN_VALUE: Score = m(2521, 2687);

const PIECE_VALUES: [Score; 6] = [
    PAWN_VALUE,
    KNIGHT_VALUE,
    BISHOP_VALUE,
    ROOK_VALUE,
    QUEEN_VALUE,
    m(0, 0), // King
];

// Phase values for tapered eval.
const PHASE_VALUES: [i32; 6] = [0, 1, 1, 2, 4, 0]; 
const TOTAL_PHASE: i32 = 24;

// --- PIECE-SQUARE TABLES (PeSTO Derived) ---

const PAWN_PST: [Score; 64] = [
    m(0, 0), m(0, 0), m(0, 0), m(0, 0), m(0, 0), m(0, 0), m(0, 0), m(0, 0),
    m(16, 12), m(18, 14), m(20, 16), m(22, 18), m(22, 18), m(20, 16), m(18, 14), m(16, 12),
    m(10, 10), m(12, 12), m(14, 14), m(16, 16), m(16, 16), m(14, 14), m(12, 12), m(10, 10),
    m(6, 8), m(8, 10), m(10, 12), m(12, 14), m(12, 14), m(10, 12), m(8, 10), m(6, 8),
    m(3, 6), m(4, 8), m(5, 10), m(10, 12), m(10, 12), m(5, 10), m(4, 8), m(3, 6),
    m(2, 8), m(3, 10), m(4, 12), m(3, 14), m(3, 14), m(4, 12), m(3, 10), m(2, 8),
    m(1, 12), m(2, 14), m(3, 16), m(2, 18), m(2, 18), m(3, 16), m(2, 14), m(1, 12),
    m(0, 0), m(0, 0), m(0, 0), m(0, 0), m(0, 0), m(0, 0), m(0, 0), m(0, 0),
];

const KNIGHT_PST: [Score; 64] = [
    m(-18, -16), m(-12, -10), m(-8, -6), m(-6, -4), m(-6, -4), m(-8, -6), m(-12, -10), m(-18, -16),
    m(-12, -10), m(-6, -4), m(0, 0), m(2, 2), m(2, 2), m(0, 0), m(-6, -4), m(-12, -10),
    m(-8, -6), m(0, 0), m(6, 4), m(8, 6), m(8, 6), m(6, 4), m(0, 0), m(-8, -6),
    m(-6, -4), m(2, 2), m(8, 6), m(12, 8), m(12, 8), m(8, 6), m(2, 2), m(-6, -4),
    m(-6, -4), m(2, 2), m(8, 6), m(12, 8), m(12, 8), m(8, 6), m(2, 2), m(-6, -4),
    m(-8, -6), m(0, 0), m(6, 4), m(8, 6), m(8, 6), m(6, 4), m(0, 0), m(-8, -6),
    m(-12, -10), m(-6, -4), m(0, 0), m(2, 2), m(2, 2), m(0, 0), m(-6, -4), m(-12, -10),
    m(-18, -16), m(-12, -10), m(-8, -6), m(-6, -4), m(-6, -4), m(-8, -6), m(-12, -10), m(-18, -16),
];

const BISHOP_PST: [Score; 64] = [
    m(-8, -6), m(-4, -3), m(-2, -1), m(-2, 0), m(-2, 0), m(-2, -1), m(-4, -3), m(-8, -6),
    m(-4, -3), m(-1, 0), m(1, 2), m(2, 4), m(2, 4), m(1, 2), m(-1, 0), m(-4, -3),
    m(-2, -1), m(1, 2), m(4, 5), m(5, 6), m(5, 6), m(4, 5), m(1, 2), m(-2, -1),
    m(-1, 0), m(2, 4), m(5, 6), m(7, 8), m(7, 8), m(5, 6), m(2, 4), m(-1, 0),
    m(-1, 0), m(2, 4), m(5, 6), m(7, 8), m(7, 8), m(5, 6), m(2, 4), m(-1, 0),
    m(-2, -1), m(1, 2), m(4, 5), m(5, 6), m(5, 6), m(4, 5), m(1, 2), m(-2, -1),
    m(-4, -3), m(-1, 0), m(1, 2), m(2, 4), m(2, 4), m(1, 2), m(-1, 0), m(-4, -3),
    m(-8, -6), m(-4, -3), m(-2, -1), m(-2, 0), m(-2, 0), m(-2, -1), m(-4, -3), m(-8, -6),
];

const ROOK_PST: [Score; 64] = [
    m(-6, -4), m(-4, -2), m(-2, 0), m(0, 2), m(0, 2), m(-2, 0), m(-4, -2), m(-6, -4),
    m(-4, -2), m(-2, 0), m(0, 2), m(1, 4), m(1, 4), m(0, 2), m(-2, 0), m(-4, -2),
    m(-3, -1), m(-1, 1), m(0, 3), m(2, 5), m(2, 5), m(0, 3), m(-1, 1), m(-3, -1),
    m(-2, 0), m(0, 2), m(1, 4), m(3, 6), m(3, 6), m(1, 4), m(0, 2), m(-2, 0),
    m(-2, 0), m(0, 2), m(1, 4), m(3, 6), m(3, 6), m(1, 4), m(0, 2), m(-2, 0),
    m(-3, -1), m(-1, 1), m(0, 3), m(2, 5), m(2, 5), m(0, 3), m(-1, 1), m(-3, -1),
    m(-4, -2), m(-2, 0), m(0, 2), m(1, 4), m(1, 4), m(0, 2), m(-2, 0), m(-4, -2),
    m(-6, -4), m(-4, -2), m(-2, 0), m(0, 2), m(0, 2), m(-2, 0), m(-4, -2), m(-6, -4),
];

const QUEEN_PST: [Score; 64] = [
    m(-6, -4), m(-4, -2), m(-3, -1), m(-2, 0), m(-2, 0), m(-3, -1), m(-4, -2), m(-6, -4),
    m(-4, -2), m(-2, 0), m(-1, 1), m(0, 2), m(0, 2), m(-1, 1), m(-2, 0), m(-4, -2),
    m(-3, -1), m(-1, 1), m(1, 3), m(2, 4), m(2, 4), m(1, 3), m(-1, 1), m(-3, -1),
    m(-2, 0), m(0, 2), m(2, 4), m(4, 6), m(4, 6), m(2, 4), m(0, 2), m(-2, 0),
    m(-2, 0), m(0, 2), m(2, 4), m(4, 6), m(4, 6), m(2, 4), m(0, 2), m(-2, 0),
    m(-3, -1), m(-1, 1), m(1, 3), m(2, 4), m(2, 4), m(1, 3), m(-1, 1), m(-3, -1),
    m(-4, -2), m(-2, 0), m(-1, 1), m(0, 2), m(0, 2), m(-1, 1), m(-2, 0), m(-4, -2),
    m(-6, -4), m(-4, -2), m(-3, -1), m(-2, 0), m(-2, 0), m(-3, -1), m(-4, -2), m(-6, -4),
];

const KING_PST_MG: [Score; 64] = [
    m(24, -20), m(30, -10), m(18, -5), m(8, 0), m(8, 0), m(18, -5), m(30, -10), m(24, -20),
    m(20, -10), m(26, 0), m(12, 5), m(0, 10), m(0, 10), m(12, 5), m(26, 0), m(20, -10),
    m(12, -5), m(18, 5), m(0, 12), m(-10, 18), m(-10, 18), m(0, 12), m(18, 5), m(12, -5),
    m(0, 0), m(8, 10), m(-8, 18), m(-18, 24), m(-18, 24), m(-8, 18), m(8, 10), m(0, 0),
    m(-2, 5), m(4, 12), m(-10, 20), m(-20, 28), m(-20, 28), m(-10, 20), m(4, 12), m(-2, 5),
    m(-6, 8), m(0, 15), m(-12, 24), m(-22, 32), m(-22, 32), m(-12, 24), m(0, 15), m(-6, 8),
    m(-10, 10), m(-4, 18), m(-14, 26), m(-24, 34), m(-24, 34), m(-14, 26), m(-4, 18), m(-10, 10),
    m(-12, 12), m(-6, 20), m(-14, 28), m(-18, 36), m(-18, 36), m(-14, 28), m(-6, 20), m(-12, 12),
];

struct PassedPawnMasks {
    white: [Bitboard; 64],
    black: [Bitboard; 64],
}

static PASSED_PAWN_MASKS: OnceLock<PassedPawnMasks> = OnceLock::new();


// --- EVALUATION FUNCTION ---

pub fn evaluate(board: &Board) -> i32 {
    let mut score = m(0, 0);
    let mut phase = 0;

    let all_pieces = board.color_bitboards[Color::White as usize] | board.color_bitboards[Color::Black as usize];
    let white_pieces = board.color_bitboards[Color::White as usize];
    let black_pieces = board.color_bitboards[Color::Black as usize];

    // 1. Material and Piece-Square Tables
    for piece_idx in 0..6 {
        let piece_type = unsafe { std::mem::transmute(piece_idx as u8) };
        let piece_value = PIECE_VALUES[piece_idx];
        
        // White Pieces
        let mut white_bb = board.bitboards[piece_idx] & white_pieces;
        phase += PHASE_VALUES[piece_idx] * white_bb.count_ones() as i32;
        
        while white_bb != 0 {
            let sq = white_bb.trailing_zeros() as usize;
            score += piece_value;
            score += get_pst(piece_type, sq);
            white_bb &= white_bb - 1;
        }

        // Black Pieces
        let mut black_bb = board.bitboards[piece_idx] & black_pieces;
        phase += PHASE_VALUES[piece_idx] * black_bb.count_ones() as i32;
        
        while black_bb != 0 {
            let sq = black_bb.trailing_zeros() as usize;
            score -= piece_value;
            let mirrored_sq = sq ^ 56; 
            score -= get_pst(piece_type, mirrored_sq);
            black_bb &= black_bb - 1;
        }
    }

    // 1b. Castling Bonus (encourages king safety in opening/middlegame)
    let white_king_sq = (board.bitboards[PieceType::King as usize] & white_pieces).trailing_zeros() as u8;
    let black_king_sq = (board.bitboards[PieceType::King as usize] & black_pieces).trailing_zeros() as u8;

    // White castled king-side (g1)
    if white_king_sq == 6 {
        score += m(40, 10);
    }
    // White castled queen-side (c1)
    else if white_king_sq == 2 {
        score += m(30, 5);
    }

    // Black castled king-side (g8)
    if black_king_sq == 62 {
        score -= m(40, 10);
    }
    // Black castled queen-side (c8)
    else if black_king_sq == 58 {
        score -= m(30, 5);
    }

    // =============================================================
    // Opening phase development bonuses
    // =============================================================
    let total_phase = TOTAL_PHASE;
    let opening_factor = phase.min(total_phase) as f32 / total_phase as f32;

    // Knight and bishop development bonus
    let white_minor = (board.bitboards[PieceType::Knight as usize] | board.bitboards[PieceType::Bishop as usize]) & white_pieces;
    let black_minor = (board.bitboards[PieceType::Knight as usize] | board.bitboards[PieceType::Bishop as usize]) & black_pieces;

    let mut white_dev = 0;
    let mut black_dev = 0;

    let mut minors = white_minor;
    while minors != 0 {
        let sq = minors.trailing_zeros() as usize;
        let rank = sq / 8;
        if rank != 0 { white_dev += 10; }
        minors &= minors - 1;
    }
    let mut minors = black_minor;
    while minors != 0 {
        let sq = minors.trailing_zeros() as usize;
        let rank = sq / 8;
        if rank != 7 { black_dev += 10; }
        minors &= minors - 1;
    }

    // Queen early move penalty (small penalty to discourage premature queen moves)
    let white_queen_on_start = (board.bitboards[PieceType::Queen as usize] & white_pieces & (1u64 << 3)) != 0;
    let black_queen_on_start = (board.bitboards[PieceType::Queen as usize] & black_pieces & (1u64 << 59)) != 0;
    let queen_penalty = if !white_queen_on_start && (board.bitboards[PieceType::Queen as usize] & white_pieces) != 0 { -30 } else { 0 };
    let queen_penalty_black = if !black_queen_on_start && (board.bitboards[PieceType::Queen as usize] & black_pieces) != 0 { 30 } else { 0 };

    // Extra penalty if queen moved before all knights and bishops have moved
    let white_minors_starting = (board.bitboards[PieceType::Knight as usize] | board.bitboards[PieceType::Bishop as usize]) & white_pieces & ((1u64 << 1) | (1u64 << 2) | (1u64 << 5) | (1u64 << 6));
    let black_minors_starting = (board.bitboards[PieceType::Knight as usize] | board.bitboards[PieceType::Bishop as usize]) & black_pieces & ((1u64 << 57) | (1u64 << 58) | (1u64 << 61) | (1u64 << 62));
    let white_queen_extra = if !white_queen_on_start && (board.bitboards[PieceType::Queen as usize] & white_pieces) != 0 && white_minors_starting != 0 { -25 } else { 0 };
    let black_queen_extra = if !black_queen_on_start && (board.bitboards[PieceType::Queen as usize] & black_pieces) != 0 && black_minors_starting != 0 { 25 } else { 0 };

    // Center occupation bonus (pawns prioritized)
    let center_squares = [27u8, 28, 35, 36]; // d4, e4, d5, e5
    let mut white_center = 0;
    let mut black_center = 0;
    for &sq in &center_squares {
        let bb = 1u64 << sq;
        if (bb & white_pieces) != 0 {
            if (bb & board.bitboards[PieceType::Pawn as usize]) != 0 {
                white_center += 20; // Pawn on center = higher bonus
            } else {
                white_center += 10; // Other pieces on center
            }
        }
        if (bb & black_pieces) != 0 {
            if (bb & board.bitboards[PieceType::Pawn as usize]) != 0 {
                black_center += 20;
            } else {
                black_center += 10;
            }
        }
    }
    let center_bonus = white_center - black_center;

    // Space advantage (pieces in enemy territory)
    let white_enemy_territory = 0xFFu64 << 32; // ranks 4-7
    let black_enemy_territory = (1u64 << 32) - 1; // ranks 0-3
    let white_space = (white_pieces & white_enemy_territory).count_ones() as i32;
    let black_space = (black_pieces & black_enemy_territory).count_ones() as i32;
    let white_pawns_space = (board.bitboards[PieceType::Pawn as usize] & white_pieces & white_enemy_territory).count_ones() as i32;
    let black_pawns_space = (board.bitboards[PieceType::Pawn as usize] & black_pieces & black_enemy_territory).count_ones() as i32;
    let space_bonus = (white_space + white_pawns_space * 2 - black_space - black_pawns_space * 2) * 2;

    // Combine and scale by opening factor (fade in endgame)
    let dev_bonus = (white_dev - black_dev) + queen_penalty + white_queen_extra + queen_penalty_black + black_queen_extra + center_bonus + space_bonus;
    let dev_bonus_scaled = (dev_bonus as f32 * opening_factor) as i32;
    score += m(dev_bonus_scaled, 0);

    // =============================================================

    // 2. Pawn Structure
    score += evaluate_pawns(board, Color::White);
    score -= evaluate_pawns(board, Color::Black);
    
    // 2b. Passed Pawns
    score += evaluate_passed_pawns(board, Color::White);
    score -= evaluate_passed_pawns(board, Color::Black);

    // 3. Mobility (phase-aware)
    score += evaluate_mobility(board, Color::White, all_pieces, phase, total_phase);
    score -= evaluate_mobility(board, Color::Black, all_pieces, phase, total_phase);
    
    // 3b. Outposts and Minor Pieces
    score += evaluate_outposts_and_minors(board, Color::White, all_pieces);
    score -= evaluate_outposts_and_minors(board, Color::Black, all_pieces);
    
    // 3c. Trapped Bishops
    score += evaluate_trapped_bishops(board, Color::White);
    score -= evaluate_trapped_bishops(board, Color::Black);

    // 4. King Safety (enhanced, phase-aware)
    score += evaluate_king_safety_enhanced(board, Color::White, phase, total_phase);
    score -= evaluate_king_safety_enhanced(board, Color::Black, phase, total_phase);

    // 5. Bishop Pair Bonus
    let white_bishops = (board.bitboards[PieceType::Bishop as usize] & white_pieces).count_ones();
    let black_bishops = (board.bitboards[PieceType::Bishop as usize] & black_pieces).count_ones();
    
    if white_bishops >= 2 { score += m(30, 60); }
    if black_bishops >= 2 { score -= m(30, 60); }
    
    // 6. Tempo Bonus (~10 cp in midgame, fades in endgame)
    let tempo_bonus = (10 * (total_phase - phase)) / total_phase;
    score += m(tempo_bonus, 0);

    // Tapered Evaluation
    phase = phase.min(TOTAL_PHASE);
    let final_score = (score.0 * phase + score.1 * (TOTAL_PHASE - phase)) / TOTAL_PHASE;

    // Return from perspective of side to move
    if board.side_to_move == Color::White {
        final_score
    } else {
        -final_score
    }
}

// --- HELPER FUNCTIONS ---

fn get_pst(piece: PieceType, sq: usize) -> Score {
    match piece {
        PieceType::Pawn => PAWN_PST[sq],
        PieceType::Knight => KNIGHT_PST[sq],
        PieceType::Bishop => BISHOP_PST[sq],
        PieceType::Rook => ROOK_PST[sq],
        PieceType::Queen => QUEEN_PST[sq],
        PieceType::King => KING_PST_MG[sq],
    }
}

fn pawn_attacks(pawns: Bitboard, color: Color) -> Bitboard {
    const FILE_A: Bitboard = 0x0101_0101_0101_0101;
    const FILE_H: Bitboard = 0x8080_8080_8080_8080;

    match color {
        Color::White => ((pawns & !FILE_A) << 7) | ((pawns & !FILE_H) << 9),
        Color::Black => ((pawns & !FILE_H) >> 7) | ((pawns & !FILE_A) >> 9),
    }
}

fn passed_pawn_masks() -> &'static PassedPawnMasks {
    PASSED_PAWN_MASKS.get_or_init(|| {
        let mut masks = PassedPawnMasks {
            white: [0; 64],
            black: [0; 64],
        };

        for sq in 0..64 {
            let file = sq % 8;
            let rank = sq / 8;

            let mut white_mask = 0u64;
            for r in (rank + 1)..8 {
                for df in -1i32..=1 {
                    let nf = file as i32 + df;
                    if (0..8).contains(&nf) {
                        white_mask |= 1u64 << (r * 8 + nf as usize);
                    }
                }
            }
            masks.white[sq] = white_mask;

            let mut black_mask = 0u64;
            let mut r = rank as i32 - 1;
            while r >= 0 {
                for df in -1i32..=1 {
                    let nf = file as i32 + df;
                    if (0..8).contains(&nf) {
                        black_mask |= 1u64 << ((r as usize) * 8 + nf as usize);
                    }
                }
                r -= 1;
            }
            masks.black[sq] = black_mask;
        }

        masks
    })
}

fn evaluate_pawns(board: &Board, color: Color) -> Score {
    let pawns = board.bitboards[PieceType::Pawn as usize] & board.color_bitboards[color as usize];
    let mut pawn_score = m(0, 0);
    
    // Isolated Pawns
    let west_pawns = (pawns >> 1) & 0xFEFEFEFEFEFEFEFE; 
    let east_pawns = (pawns << 1) & 0x7F7F7F7F7F7F7F7F; 
    
    let supported_pawns = west_pawns | east_pawns;
    let isolated_pawns = pawns & !supported_pawns;
    
    pawn_score -= m(15, 25) * isolated_pawns.count_ones() as i32;
    
    // Doubled Pawns
    let enemy_pawns = board.bitboards[PieceType::Pawn as usize] & board.color_bitboards[color.opponent() as usize];
    for file in 0..8 {
        let file_mask = 0x0101010101010101u64 << file;
        let file_pawns = pawns & file_mask;
        let count = file_pawns.count_ones();
        if count > 1 {
            pawn_score -= m(15, 20) * (count - 1) as i32;
            // Extra penalty if enemy pawns are on the same file
            if (enemy_pawns & file_mask) != 0 {
                pawn_score -= m(10, 10) * (count - 1) as i32;
            }
        }
    }
    
    pawn_score
}

fn is_pawn_passed(board: &Board, pawn_sq: u8, color: Color) -> bool {
    let enemy_pawns = board.bitboards[PieceType::Pawn as usize] & board.color_bitboards[color.opponent() as usize];
    let masks = passed_pawn_masks();
    let passed_mask = match color {
        Color::White => masks.white[pawn_sq as usize],
        Color::Black => masks.black[pawn_sq as usize],
    };
    (enemy_pawns & passed_mask) == 0
}

fn evaluate_passed_pawns(board: &Board, color: Color) -> Score {
    let pawns = board.bitboards[PieceType::Pawn as usize] & board.color_bitboards[color as usize];
    let mut passed_score = m(0, 0);
    let our_king = board.bitboards[PieceType::King as usize] & board.color_bitboards[color as usize];
    let king_sq = if our_king != 0 { our_king.trailing_zeros() as u8 } else { 0 };
    let enemy_king = board.bitboards[PieceType::King as usize] & board.color_bitboards[color.opponent() as usize];
    let enemy_king_sq = if enemy_king != 0 { enemy_king.trailing_zeros() as u8 } else { 255 };

    // First pass: build passed pawn bitboard
    let mut passed_bb = 0u64;
    let mut bb = pawns;
    while bb != 0 {
        let sq = bb.trailing_zeros() as u8;
        if is_pawn_passed(board, sq, color) {
            passed_bb |= 1u64 << sq;
        }
        bb &= bb - 1;
    }

    // Second pass: evaluate each passed pawn
    bb = passed_bb;
    while bb != 0 {
        let sq = bb.trailing_zeros() as u8;
        let rank = sq / 8;
        let rank_from_promotion = if color == Color::White {
            7 - rank
        } else {
            rank
        };

        let bonus = match rank_from_promotion {
            0 => m(50, 140),
            1 => m(30, 90),
            2 => m(20, 60),
            3 => m(10, 35),
            4 => m(5, 15),
            _ => m(0, 5),
        };

        passed_score += bonus;

        // Own king proximity bonus for support
        let king_dist = ((king_sq % 8) as i32 - (sq % 8) as i32).abs()
                      + ((king_sq / 8) as i32 - (sq / 8) as i32).abs();
        if king_dist <= 3 {
            passed_score += m(5, 10);
        }

        // Enemy king far away bonus (endgames)
        if enemy_king_sq != 255 {
            let enemy_king_dist = ((enemy_king_sq % 8) as i32 - (sq % 8) as i32).abs()
                                + ((enemy_king_sq / 8) as i32 - (sq / 8) as i32).abs();
            if enemy_king_dist > 3 {
                passed_score += m(10, 30);
            }
        }

        bb &= bb - 1;
    }

    // Connected passers bonus
    bb = passed_bb;
    while bb != 0 {
        let sq = bb.trailing_zeros() as u8;
        let file = sq % 8;
        let adjacent = 0u64
            | if file > 0 { passed_bb & (0x0101010101010101u64 << (file - 1)) } else { 0 }
            | if file < 7 { passed_bb & (0x0101010101010101u64 << (file + 1)) } else { 0 };
        if adjacent != 0 {
            passed_score += m(10, 20);
        }
        bb &= bb - 1;
    }

    passed_score
}

fn evaluate_king_safety_enhanced(board: &Board, color: Color, phase: i32, total_phase: i32) -> Score {
    let king_bb = board.bitboards[PieceType::King as usize] 
                & board.color_bitboards[color as usize];
    if king_bb == 0 { return m(0,0); }

    let king_idx = king_bb.trailing_zeros() as u8;
    let enemy_color = color.opponent();
    let enemy_pieces = board.color_bitboards[enemy_color as usize];

    let opening_factor = (total_phase - phase.min(total_phase)) as f32 / total_phase as f32;
    if opening_factor <= 0.0 { return m(0, 0); }

    let mut danger = 0;

    // King ring (king + adjacent squares)
    let king_ring = attacks::get_king_attacks(king_idx) | (1u64 << king_idx);

    // Extended zone: ring + squares one step further
    let mut extended_zone = king_ring;
    let mut ring_bb = king_ring;
    while ring_bb != 0 {
        let sq = ring_bb.trailing_zeros() as u8;
        extended_zone |= attacks::get_king_attacks(sq);
        ring_bb &= ring_bb - 1;
    }

    // 1. Weighted attackers in extended king zone
    let queens = board.bitboards[PieceType::Queen as usize] & enemy_pieces & extended_zone;
    danger += queens.count_ones() as i32 * 40;

    let rooks = board.bitboards[PieceType::Rook as usize] & enemy_pieces & extended_zone;
    danger += rooks.count_ones() as i32 * 25;

    let minors = (board.bitboards[PieceType::Bishop as usize] | board.bitboards[PieceType::Knight as usize])
                & enemy_pieces & extended_zone;
    danger += minors.count_ones() as i32 * 15;

    // 2. Semi-open files near king with enemy heavy pieces
    let king_file = king_idx % 8;
    let enemy_rooks_queens = (board.bitboards[PieceType::Rook as usize] | board.bitboards[PieceType::Queen as usize]) & enemy_pieces;
    let friendly_pawns = board.bitboards[PieceType::Pawn as usize] & board.color_bitboards[color as usize];

    for file_offset in -1..=1i32 {
        let file = (king_file as i32 + file_offset) as u8;
        if file < 8 {
            let file_mask = 0x0101010101010101u64 << file;
            if (friendly_pawns & file_mask) == 0 && (enemy_rooks_queens & file_mask) != 0 {
                danger += 15;
            }
        }
    }

    // 3. Pawn shield quality
    let pawns = board.bitboards[PieceType::Pawn as usize] & board.color_bitboards[color as usize];
    let mut shield_count = 0;
    if color == Color::White && king_idx < 56 {
        let front_sq = king_idx + 8;
        if (pawns >> front_sq) & 1 != 0 { shield_count += 1; }
        if king_idx % 8 > 0 && (pawns >> (front_sq - 1)) & 1 != 0 { shield_count += 1; }
        if king_idx % 8 < 7 && (pawns >> (front_sq + 1)) & 1 != 0 { shield_count += 1; }
    } else if color == Color::Black && king_idx > 7 {
        let front_sq = king_idx - 8;
        if (pawns >> front_sq) & 1 != 0 { shield_count += 1; }
        if king_idx % 8 > 0 && (pawns >> (front_sq - 1)) & 1 != 0 { shield_count += 1; }
        if king_idx % 8 < 7 && (pawns >> (front_sq + 1)) & 1 != 0 { shield_count += 1; }
    }

    danger += match shield_count {
        3 => 0,
        2 => 15,
        1 => 40,
        _ => 60,
    };

    danger = (danger as f32 * opening_factor) as i32;

    Score(-danger, 0)
}

fn evaluate_mobility(board: &Board, color: Color, all_pieces: Bitboard, _phase: i32, _total_phase: i32) -> Score {
    let our_pieces = board.color_bitboards[color as usize];
    let enemy_pawns = board.bitboards[PieceType::Pawn as usize] & board.color_bitboards[color.opponent() as usize];
    let mut mobility_score = m(0, 0);
    
    // Compute enemy pawn attack map (squares where our pieces can't safely move)
    let pawn_attacks = pawn_attacks(enemy_pawns, color.opponent());
    
    // Knight Mobility - decreases in value towards endgame
    let knight_weight = m(4, 4);
    let knights = board.bitboards[PieceType::Knight as usize] & our_pieces;
    let mut bb = knights;
    while bb != 0 {
        let sq = bb.trailing_zeros() as u8;
        let moves = (attacks::get_knight_attacks(sq) & !our_pieces & !pawn_attacks).count_ones() as i32;
        mobility_score += knight_weight * moves;
        bb &= bb - 1;
    }
    
    // Bishop Mobility - increases in value towards endgame  
    let bishop_weight  = m(4, 4);
    let bishops = board.bitboards[PieceType::Bishop as usize] & our_pieces;
    let mut bb = bishops;
    while bb != 0 {
        let sq = bb.trailing_zeros() as u8;
        let moves = (attacks::get_bishop_attacks(sq, all_pieces) & !our_pieces & !pawn_attacks).count_ones() as i32;
        mobility_score += bishop_weight * moves;
        bb &= bb - 1;
    }
    
    // Rook Mobility
    let rook_weight   = m(3, 4);
    let rooks = board.bitboards[PieceType::Rook as usize] & our_pieces;
    let mut bb = rooks;
    while bb != 0 {
        let sq = bb.trailing_zeros() as u8;
        let moves = (attacks::get_rook_attacks(sq, all_pieces) & !our_pieces & !pawn_attacks).count_ones() as i32;
        mobility_score += rook_weight * moves;
        bb &= bb - 1;
    }
    
    // Queen Mobility
    let queen_weight  = m(1, 2);
    let queens = board.bitboards[PieceType::Queen as usize] & our_pieces;
    let mut bb = queens;
    while bb != 0 {
        let sq = bb.trailing_zeros() as u8;
        let moves = ((attacks::get_rook_attacks(sq, all_pieces) | attacks::get_bishop_attacks(sq, all_pieces)) 
                    & !our_pieces & !pawn_attacks).count_ones() as i32;
        mobility_score += queen_weight * moves;
        bb &= bb - 1;
    }
    
    mobility_score
}

fn evaluate_outposts_and_minors(board: &Board, color: Color, _all_pieces: Bitboard) -> Score {
    let our_pieces = board.color_bitboards[color as usize];
    let enemy_pawns = board.bitboards[PieceType::Pawn as usize] & board.color_bitboards[color.opponent() as usize];
    let mut outpost_score = m(0, 0);
    
    // Compute squares not attacked by enemy pawns
    let pawn_safe_squares = !pawn_attacks(enemy_pawns, color.opponent());
    
    let _enemy_back_rank = if color == Color::White { 0x00000000000000FFu64 } else { 0xFF00000000000000u64 };
    let in_enemy_half = if color == Color::White { 0xFFFFFFFF00000000u64 } else { 0x00000000FFFFFFFFu64 };
    
    // Knight outposts
    let knights = board.bitboards[PieceType::Knight as usize] & our_pieces;
    let mut bb = knights;
    while bb != 0 {
        let sq = bb.trailing_zeros() as u8;
        let sq_bb = 1u64 << sq;
        if (sq_bb & pawn_safe_squares & in_enemy_half) != 0 {
            // Check if the square is protected by our pieces
            let is_protected = (attacks::get_pawn_attacks(sq, color.opponent())
                               & board.bitboards[PieceType::Pawn as usize]
                               & our_pieces) != 0;
            if is_protected || (sq_bb & attacks::get_king_attacks(board.bitboards[PieceType::King as usize].trailing_zeros() as u8)) != 0 {
                outpost_score += m(20, 15);
            }
        }
        bb &= bb - 1;
    }
    
    // Bishop outposts (slightly less valuable than knights)
    let bishops = board.bitboards[PieceType::Bishop as usize] & our_pieces;
    let mut bb = bishops;
    while bb != 0 {
        let sq = bb.trailing_zeros() as u8;
        let sq_bb = 1u64 << sq;
        if (sq_bb & pawn_safe_squares & in_enemy_half) != 0 {
            let is_protected = (attacks::get_pawn_attacks(sq, color.opponent())
                               & board.bitboards[PieceType::Pawn as usize]
                               & our_pieces) != 0;
            if is_protected {
                outpost_score += m(15, 13);
            }
        }
        bb &= bb - 1;
    }
    
    // Rook on open file bonus
    let rooks = board.bitboards[PieceType::Rook as usize] & our_pieces;
    let mut bb = rooks;
    while bb != 0 {
        let sq = bb.trailing_zeros() as u8;
        let file = sq % 8;
        let file_mask = 0x0101010101010101u64 << file;
        
        let has_friendly = (board.bitboards[PieceType::Pawn as usize] & our_pieces & file_mask) != 0;
        let has_enemy = (board.bitboards[PieceType::Pawn as usize] & board.color_bitboards[color.opponent() as usize] & file_mask) != 0;
        
        if !has_friendly && !has_enemy {
            outpost_score += m(10, 15);
        } else if !has_friendly && has_enemy {
            outpost_score += m(5, 8);
        }

        // Rook on 7th rank bonus (enemy's second rank)
        let rook_on_7th = if color == Color::White {
            0x00FF000000000000u64
        } else {
            0x000000000000FF00u64
        };
        if ((1u64 << sq) & rook_on_7th) != 0 && !has_friendly {
            outpost_score += m(15, 25);
        }
        
        bb &= bb - 1;
    }
    
    outpost_score
}

fn evaluate_trapped_bishops(board: &Board, color: Color) -> Score {
    let our_pieces = board.color_bitboards[color as usize];
    let bishops = board.bitboards[PieceType::Bishop as usize] & our_pieces;
    let pawns = board.bitboards[PieceType::Pawn as usize] & our_pieces;
    let mut trapped_score = m(0, 0);
    
    let mut bb = bishops;
    while bb != 0 {
        let sq = bb.trailing_zeros() as u8;
        
        let is_trapped = if color == Color::White {
            // a2 (16) blocked by pawn on b3 (17)
            (sq == 16 && (pawns & (1u64 << 17)) != 0) ||
            // h2 (23) blocked by pawn on g3 (22)
            (sq == 23 && (pawns & (1u64 << 22)) != 0) ||
            // b2 (17) blocked by pawns on a3 (24) and c3 (26)
            (sq == 17 && (pawns & (1u64 << 24)) != 0 && (pawns & (1u64 << 26)) != 0) ||
            // g2 (22) blocked by pawns on f3 (21) and h3 (23)
            (sq == 22 && (pawns & (1u64 << 21)) != 0 && (pawns & (1u64 << 23)) != 0)
        } else {
            // a7 (48) blocked by pawn on b6 (41)
            (sq == 48 && (pawns & (1u64 << 41)) != 0) ||
            // h7 (55) blocked by pawn on g6 (46)
            (sq == 55 && (pawns & (1u64 << 46)) != 0) ||
            // b7 (49) blocked by pawns on a6 (40) and c6 (42)
            (sq == 49 && (pawns & (1u64 << 40)) != 0 && (pawns & (1u64 << 42)) != 0) ||
            // g7 (54) blocked by pawns on f6 (45) and h6 (47)
            (sq == 54 && (pawns & (1u64 << 45)) != 0 && (pawns & (1u64 << 47)) != 0)
        };
        
        if is_trapped {
            trapped_score -= m(30, 15);
        }
        
        bb &= bb - 1;
    }
    
    trapped_score
}
