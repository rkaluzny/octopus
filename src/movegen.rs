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
use crate::board::{Board, Color, PieceType, Square};

pub const SEE_VALUE: [i32; 6] = [100, 300, 325, 500, 900, 10000];

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum MoveType {
    Normal,
    DoublePawnPush,
    KingCastle,
    QueenCastle,
    EnPassant,
    Promotion,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct Move {
    pub from: Square,
    pub to: Square,
    pub promotion: Option<PieceType>,
    pub capture: Option<PieceType>,
    pub move_type: MoveType,
}

impl Move {
    pub fn new(
        from: Square,
        to: Square,
        promotion: Option<PieceType>,
        capture: Option<PieceType>,
        move_type: MoveType,
    ) -> Self {
        Self { from, to, promotion, capture, move_type }
    }

    pub fn to_uci_string(&self) -> String {
        let mut uci = String::new();
        uci.push_str(&square_to_uci_string(self.from));
        uci.push_str(&square_to_uci_string(self.to));
        if let Some(promotion) = self.promotion {
            uci.push(match promotion {
                PieceType::Queen => 'q',
                PieceType::Rook => 'r',
                PieceType::Bishop => 'b',
                PieceType::Knight => 'n',
                _ => unreachable!(),
            });
        }
        uci
    }

    pub fn from_uci(uci: &str) -> Self {
        let from = uci_str_to_square(&uci[0..2]);
        let to = uci_str_to_square(&uci[2..4]);
        let promotion = if uci.len() == 5 {
            Some(match uci.chars().nth(4).unwrap() {
                'q' => PieceType::Queen,
                'r' => PieceType::Rook,
                'b' => PieceType::Bishop,
                'n' => PieceType::Knight,
                _ => unreachable!(),
            })
        } else {
            None
        };
        Self { from, to, promotion, capture: None, move_type: MoveType::Normal }
    }
}

const WHITE_KING_SIDE_CASTLE: u8 = 0b1000;
const WHITE_QUEEN_SIDE_CASTLE: u8 = 0b0100;
const BLACK_KING_SIDE_CASTLE: u8 = 0b0010;
const BLACK_QUEEN_SIDE_CASTLE: u8 = 0b0001;

pub fn generate_moves(board: &Board) -> Vec<Move> {
    generate_legal_moves(board, false)
}

pub fn generate_all_captures(board: &Board) -> Vec<Move> {
    generate_legal_moves(board, true)
}

pub fn find_legal_move(board: &Board, uci: &str) -> Option<Move> {
    generate_moves(board)
        .into_iter()
        .find(|mv| mv.to_uci_string() == uci)
}

fn generate_legal_moves(board: &Board, captures_only: bool) -> Vec<Move> {
    let color = board.side_to_move;
    let our_pieces = board.color_bitboards[color as usize];
    let their_pieces = board.color_bitboards[color.opponent() as usize];
    let all_pieces = our_pieces | their_pieces;

    let mut moves = Vec::new();
    generate_pawn_moves(&mut moves, board, color, all_pieces, their_pieces, captures_only);
    generate_knight_moves(&mut moves, board, color, our_pieces, their_pieces, captures_only);
    generate_sliding_moves(&mut moves, board, color, our_pieces, their_pieces, PieceType::Rook, captures_only);
    generate_sliding_moves(&mut moves, board, color, our_pieces, their_pieces, PieceType::Bishop, captures_only);
    generate_sliding_moves(&mut moves, board, color, our_pieces, their_pieces, PieceType::Queen, captures_only);
    generate_king_moves(&mut moves, board, color, our_pieces, their_pieces, captures_only);

    moves.into_iter().filter(|mv| is_legal_move(board, *mv)).collect()
}

pub fn is_move_legal(board: &Board, mv: Move) -> bool {
    is_legal_move(board, mv)
}

fn is_legal_move(board: &Board, mv: Move) -> bool {
    if let Some((_, piece_color)) = board.get_piece_at(mv.from as u8) {
        if piece_color != board.side_to_move {
            return false;
        }
    } else {
        return false;
    }

    if !verify_sliding_move(board, &mv) {
        return false;
    }

    let moving_color = board.side_to_move;
    let mut board_copy = *board;
    let undo = board_copy.make_move(&mv);

    let king_bb = board_copy.bitboards[PieceType::King as usize]
        & board_copy.color_bitboards[moving_color as usize];

    let is_legal = if king_bb == 0 {
        false
    } else {
        let king_sq = king_bb.trailing_zeros() as u8;
        !board_copy.is_square_attacked(king_sq, moving_color.opponent())
    };

    board_copy.unmake_move(undo);
    is_legal
}

fn generate_pawn_moves(
    moves: &mut Vec<Move>,
    board: &Board,
    color: Color,
    all_pieces: u64,
    their_pieces: u64,
    captures_only: bool,
) {
    let pawns = board.bitboards[PieceType::Pawn as usize] & board.color_bitboards[color as usize];
    let promotion_rank = if color == Color::White { 6 } else { 1 };

    for from_idx in 0..64u8 {
        if (pawns >> from_idx) & 1 == 0 { continue; }
        let from_square = index_to_square(from_idx);
        let rank = from_idx / 8;

        if !captures_only {
            let forward = if color == Color::White { 8i8 } else { -8i8 };
            let one_step = from_idx as i8 + forward;
            if (0..64).contains(&one_step) {
                let to_idx = one_step as u8;
                if (all_pieces >> to_idx) & 1 == 0 {
                    let to_square = index_to_square(to_idx);
                    if rank == promotion_rank {
                        push_promotions(moves, from_square, to_square, None);
                    } else {
                        moves.push(Move::new(from_square, to_square, None, None, MoveType::Normal));
                    }

                    let start_rank = if color == Color::White { 1 } else { 6 };
                    if rank == start_rank {
                        let two_step = from_idx as i8 + if color == Color::White { 16 } else { -16 };
                        if (0..64).contains(&two_step) {
                            let to_idx_double = two_step as u8;
                            if (all_pieces >> to_idx_double) & 1 == 0 {
                                moves.push(Move::new(
                                    from_square, index_to_square(to_idx_double),
                                    None, None, MoveType::DoublePawnPush,
                                ));
                            }
                        }
                    }
                }
            }
        }

        let attacks = attacks::get_pawn_attacks(from_idx, color);
        let mut capture_targets = attacks & their_pieces;
        if let Some(ep_square) = board.en_passant {
            capture_targets |= attacks & (1u64 << ep_square as u8);
        }

        while capture_targets != 0 {
            let to_idx = capture_targets.trailing_zeros() as u8;
            capture_targets &= capture_targets - 1;
            let to_square = index_to_square(to_idx);
            let captured_piece = board.get_piece_at(to_idx).map(|(piece, _)| piece);

            if Some(to_square) == board.en_passant && board.get_piece_at(to_idx).is_none() {
                moves.push(Move::new(from_square, to_square, None, Some(PieceType::Pawn), MoveType::EnPassant));
                continue;
            }

            if rank == promotion_rank {
                push_promotions(moves, from_square, to_square, captured_piece);
            } else {
                moves.push(Move::new(from_square, to_square, None, captured_piece, MoveType::Normal));
            }
        }
    }
}

fn generate_knight_moves(
    moves: &mut Vec<Move>,
    board: &Board,
    color: Color,
    our_pieces: u64,
    their_pieces: u64,
    captures_only: bool,
) {
    let knights = board.bitboards[PieceType::Knight as usize] & board.color_bitboards[color as usize];
    generate_piece_moves(moves, board, knights, our_pieces, their_pieces, captures_only, |sq| {
        attacks::get_knight_attacks(sq)
    });
}

fn generate_sliding_moves(
    moves: &mut Vec<Move>,
    board: &Board,
    color: Color,
    our_pieces: u64,
    their_pieces: u64,
    piece_type: PieceType,
    captures_only: bool,
) {
    let pieces = board.bitboards[piece_type as usize] & board.color_bitboards[color as usize];
    let all_pieces = our_pieces | their_pieces;
    generate_piece_moves(moves, board, pieces, our_pieces, their_pieces, captures_only, |sq| {
        match piece_type {
            PieceType::Rook => attacks::get_rook_attacks(sq, all_pieces),
            PieceType::Bishop => attacks::get_bishop_attacks(sq, all_pieces),
            PieceType::Queen => {
                attacks::get_rook_attacks(sq, all_pieces) | attacks::get_bishop_attacks(sq, all_pieces)
            }
            _ => 0,
        }
    });
}

fn generate_king_moves(
    moves: &mut Vec<Move>,
    board: &Board,
    color: Color,
    our_pieces: u64,
    their_pieces: u64,
    captures_only: bool,
) {
    let king = board.bitboards[PieceType::King as usize] & board.color_bitboards[color as usize];
    if king == 0 { return; }
    let from_idx = king.trailing_zeros() as u8;
    let from_square = index_to_square(from_idx);
    let attacks = attacks::get_king_attacks(from_idx) & !our_pieces;

    let mut targets = attacks;
    if captures_only { targets &= their_pieces; }

    while targets != 0 {
        let to_idx = targets.trailing_zeros() as u8;
        targets &= targets - 1;
        let to_square = index_to_square(to_idx);
        let captured_piece = if (their_pieces >> to_idx) & 1 != 0 {
            board.get_piece_at(to_idx).map(|(piece, _)| piece)
        } else {
            None
        };
        if !captures_only || captured_piece.is_some() {
            moves.push(Move::new(from_square, to_square, None, captured_piece, MoveType::Normal));
        }
    }

    if captures_only { return; }

    let enemy = color.opponent();
    if board.is_square_attacked(from_idx, enemy) { return; }

    match color {
        Color::White => {
            if board.castling_rights & WHITE_KING_SIDE_CASTLE != 0
                && board.get_piece_at(Square::F1 as u8).is_none()
                && board.get_piece_at(Square::G1 as u8).is_none()
                && !board.is_square_attacked(Square::F1 as u8, enemy)
                && !board.is_square_attacked(Square::G1 as u8, enemy)
            {
                moves.push(Move::new(Square::E1, Square::G1, None, None, MoveType::KingCastle));
            }
            if board.castling_rights & WHITE_QUEEN_SIDE_CASTLE != 0
                && board.get_piece_at(Square::D1 as u8).is_none()
                && board.get_piece_at(Square::C1 as u8).is_none()
                && board.get_piece_at(Square::B1 as u8).is_none()
                && !board.is_square_attacked(Square::D1 as u8, enemy)
                && !board.is_square_attacked(Square::C1 as u8, enemy)
            {
                moves.push(Move::new(Square::E1, Square::C1, None, None, MoveType::QueenCastle));
            }
        }
        Color::Black => {
            if board.castling_rights & BLACK_KING_SIDE_CASTLE != 0
                && board.get_piece_at(Square::F8 as u8).is_none()
                && board.get_piece_at(Square::G8 as u8).is_none()
                && !board.is_square_attacked(Square::F8 as u8, enemy)
                && !board.is_square_attacked(Square::G8 as u8, enemy)
            {
                moves.push(Move::new(Square::E8, Square::G8, None, None, MoveType::KingCastle));
            }
            if board.castling_rights & BLACK_QUEEN_SIDE_CASTLE != 0
                && board.get_piece_at(Square::D8 as u8).is_none()
                && board.get_piece_at(Square::C8 as u8).is_none()
                && board.get_piece_at(Square::B8 as u8).is_none()
                && !board.is_square_attacked(Square::D8 as u8, enemy)
                && !board.is_square_attacked(Square::C8 as u8, enemy)
            {
                moves.push(Move::new(Square::E8, Square::C8, None, None, MoveType::QueenCastle));
            }
        }
    }
}

fn generate_piece_moves<F>(
    moves: &mut Vec<Move>,
    board: &Board,
    pieces: u64,
    our_pieces: u64,
    their_pieces: u64,
    captures_only: bool,
    attack_fn: F,
) where
    F: Fn(u8) -> u64,
{
    let mut bitboard = pieces;
    while bitboard != 0 {
        let from_idx = bitboard.trailing_zeros() as u8;
        bitboard &= bitboard - 1;

        let from_square = index_to_square(from_idx);
        let mut targets = attack_fn(from_idx) & !our_pieces;
        if captures_only { targets &= their_pieces; }

        while targets != 0 {
            let to_idx = targets.trailing_zeros() as u8;
            targets &= targets - 1;
            let to_square = index_to_square(to_idx);
            let captured_piece = if (their_pieces >> to_idx) & 1 != 0 {
                board.get_piece_at(to_idx).map(|(piece, _)| piece)
            } else {
                None
            };
            if !captures_only || captured_piece.is_some() {
                moves.push(Move::new(from_square, to_square, None, captured_piece, MoveType::Normal));
            }
        }
    }
}

fn push_promotions(moves: &mut Vec<Move>, from: Square, to: Square, capture: Option<PieceType>) {
    for promotion in [PieceType::Queen, PieceType::Rook, PieceType::Bishop, PieceType::Knight] {
        moves.push(Move::new(from, to, Some(promotion), capture, MoveType::Promotion));
    }
}

fn index_to_square(index: u8) -> Square {
    unsafe { std::mem::transmute(index) }
}

// Verify that a sliding piece move has a clear path.
fn verify_sliding_move(board: &Board, mv: &Move) -> bool {
    let from_idx = mv.from as u8;
    let to_idx = mv.to as u8;
    let from_rank = from_idx / 8;
    let from_file = from_idx % 8;
    let to_rank = to_idx / 8;
    let to_file = to_idx % 8;

    let dr = to_rank as i8 - from_rank as i8;
    let df = to_file as i8 - from_file as i8;

    let is_diagonal = dr.abs() == df.abs() && dr != 0;
    let is_straight = (dr == 0 && df != 0) || (df == 0 && dr != 0);

    if !is_diagonal && !is_straight { return true; }

    let step_r = dr.signum();
    let step_f = df.signum();
    let mut r = from_rank as i8 + step_r;
    let mut f = from_file as i8 + step_f;

    while (r as u8) != to_idx / 8 || (f as u8) != to_idx % 8 {
        if r < 0 || r > 7 || f < 0 || f > 7 { return false; }
        let sq = (r * 8 + f) as u8;
        if board.get_piece_at(sq).is_some() { return false; }
        r += step_r;
        f += step_f;
    }
    true
}

fn square_to_uci_string(square: Square) -> String {
    let sq_idx = square as u8;
    let file = (sq_idx % 8) as u8 + b'a';
    let rank = (sq_idx / 8) as u8 + b'1';
    format!("{}{}", file as char, rank as char)
}

fn uci_str_to_square(s: &str) -> Square {
    let mut chars = s.chars();
    let file = chars.next().unwrap() as u8 - b'a';
    let rank = chars.next().unwrap() as u8 - b'1';
    index_to_square(rank * 8 + file)
}
