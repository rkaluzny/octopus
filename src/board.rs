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

// Board representation with bitboards, FEN parsing, and incremental hash updates.

use crate::attacks;
use crate::movegen::MoveType;
use crate::zobrist;
use std::fmt;

pub type Bitboard = u64;

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum Color {
    White,
    Black,
}

impl Color {
    pub fn opponent(&self) -> Color {
        match self {
            Color::White => Color::Black,
            Color::Black => Color::White,
        }
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum PieceType {
    Pawn,
    Knight,
    Bishop,
    Rook,
    Queen,
    King,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
#[allow(dead_code)]
#[repr(u8)]
pub enum Square {
    A1, B1, C1, D1, E1, F1, G1, H1,
    A2, B2, C2, D2, E2, F2, G2, H2,
    A3, B3, C3, D3, E3, F3, G3, H3,
    A4, B4, C4, D4, E4, F4, G4, H4,
    A5, B5, C5, D5, E5, F5, G5, H5,
    A6, B6, C6, D6, E6, F6, G6, H6,
    A7, B7, C7, D7, E7, F7, G7, H7,
    A8, B8, C8, D8, E8, F8, G8, H8,
    NoSquare,
}

pub const STARTING_FEN: &str = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";

// Castling rights: bit 3=WK, bit 2=WQ, bit 1=BK, bit 0=BQ
const WHITE_KING_SIDE_CASTLE: u8 = 0b1000;
const WHITE_QUEEN_SIDE_CASTLE: u8 = 0b0100;
const BLACK_KING_SIDE_CASTLE: u8 = 0b0010;
const BLACK_QUEEN_SIDE_CASTLE: u8 = 0b0001;

#[derive(Copy, Clone)]
pub struct Board {
    pub bitboards: [Bitboard; 6],
    pub color_bitboards: [Bitboard; 2],
    pub side_to_move: Color,
    pub en_passant: Option<Square>,
    pub castling_rights: u8,
    pub hash: u64,
    pub piece_on_square: [Option<PieceType>; 64],
    pub color_on_square: [Option<Color>; 64],
}

// Snapshot for move undo, storing all mutable board state.
#[derive(Copy, Clone)]
pub struct Undo {
    pub bitboards: [Bitboard; 6],
    pub color_bitboards: [Bitboard; 2],
    pub side_to_move: Color,
    pub en_passant: Option<Square>,
    pub castling_rights: u8,
    pub hash: u64,
    pub piece_on_square: [Option<PieceType>; 64],
    pub color_on_square: [Option<Color>; 64],
}

impl Board {
    pub fn new() -> Self {
        Self::from_fen(STARTING_FEN).unwrap()
    }

    pub fn from_fen(fen: &str) -> Result<Self, &str> {
        let mut board = Board {
            bitboards: [0; 6],
            color_bitboards: [0; 2],
            side_to_move: Color::White,
            en_passant: None,
            castling_rights: 0,
            hash: 0,
            piece_on_square: [None; 64],
            color_on_square: [None; 64],
        };

        let mut parts = fen.split_whitespace();

        // Piece placement
        let piece_placement = parts.next().ok_or("Missing piece placement")?;
        let mut rank = 7;
        let mut file = 0;
        for c in piece_placement.chars() {
            if c == '/' {
                rank -= 1;
                file = 0;
            } else if let Some(digit) = c.to_digit(10) {
                file += digit;
            } else {
                let color = if c.is_uppercase() {
                    Color::White
                } else {
                    Color::Black
                };
                let piece_type = match c.to_ascii_lowercase() {
                    'p' => PieceType::Pawn,
                    'n' => PieceType::Knight,
                    'b' => PieceType::Bishop,
                    'r' => PieceType::Rook,
                    'q' => PieceType::Queen,
                    'k' => PieceType::King,
                    _ => return Err("Invalid piece type in FEN"),
                };
                let square_index = rank * 8 + file;
                board.set_piece(piece_type, color, 1 << square_index);
                file += 1;
            }
        }

        // Side to move
        let side_to_move = parts.next().ok_or("Missing side to move")?;
        board.side_to_move = if side_to_move == "w" {
            Color::White
        } else {
            Color::Black
        };

        // Castling rights
        let castling = parts.next().ok_or("Missing castling rights")?;
        for c in castling.chars() {
            match c {
                'K' => board.castling_rights |= WHITE_KING_SIDE_CASTLE,
                'Q' => board.castling_rights |= WHITE_QUEEN_SIDE_CASTLE,
                'k' => board.castling_rights |= BLACK_KING_SIDE_CASTLE,
                'q' => board.castling_rights |= BLACK_QUEEN_SIDE_CASTLE,
                '-' => {}
                _ => return Err("Invalid castling rights in FEN"),
            }
        }

        // En passant
        let en_passant = parts.next().ok_or("Missing en passant square")?;
        if en_passant != "-" {
            let square = Self::square_from_str(en_passant)?;
            board.en_passant = Some(square);
        }

        board.hash = board.calculate_hash();
        Ok(board)
    }

    fn set_piece(&mut self, piece: PieceType, color: Color, bit: Bitboard) {
        self.bitboards[piece as usize] |= bit;
        self.color_bitboards[color as usize] |= bit;
        let sq = bit.trailing_zeros() as usize;
        self.piece_on_square[sq] = Some(piece);
        self.color_on_square[sq] = Some(color);
    }

    fn remove_piece(&mut self, piece: PieceType, color: Color, bit: Bitboard) {
        self.bitboards[piece as usize] &= !bit;
        self.color_bitboards[color as usize] &= !bit;
        let sq = bit.trailing_zeros() as usize;
        self.piece_on_square[sq] = None;
        self.color_on_square[sq] = None;
    }

    pub fn square_from_str(s: &str) -> Result<Square, &str> {
        if s.len() != 2 {
            return Err("Invalid square string");
        }
        let mut chars = s.chars();
        let file = chars.next().unwrap() as u8 - b'a';
        let rank = chars.next().unwrap() as u8 - b'1';
        if file > 7 || rank > 7 {
            return Err("Invalid square string");
        }
        Ok(unsafe { std::mem::transmute((rank * 8 + file) as u8) })
    }

    pub fn get_piece_at(&self, square_index: u8) -> Option<(PieceType, Color)> {
        let piece = self.piece_on_square[square_index as usize]?;
        let color = self.color_on_square[square_index as usize]?;
        Some((piece, color))
    }

    pub fn king_square(&self, color: Color) -> u8 {
        let king_bb = self.bitboards[PieceType::King as usize] & self.color_bitboards[color as usize];
        king_bb.trailing_zeros() as u8
    }

    pub fn apply_move(&mut self, mv: &crate::movegen::Move) {
        self.update_hash_before_move(mv);

        let from_bit = 1 << (mv.from as u8);
        let to_bit = 1 << (mv.to as u8);
        let (piece_moving, color_moving) = self.get_piece_at(mv.from as u8).unwrap();

        self.remove_piece(piece_moving, color_moving, from_bit);

        // Handle captures
        if let Some((captured_piece, captured_color)) = self.get_piece_at(mv.to as u8) {
            self.remove_piece(captured_piece, captured_color, to_bit);
        } else if mv.move_type == MoveType::EnPassant {
            let captured_pawn_sq = if color_moving == Color::White {
                mv.to as u8 - 8
            } else {
                mv.to as u8 + 8
            };
            self.remove_piece(PieceType::Pawn, color_moving.opponent(), 1 << captured_pawn_sq);
        }

        // Place piece at destination
        let piece_to_add = mv.promotion.unwrap_or(piece_moving);
        self.set_piece(piece_to_add, color_moving, to_bit);

        // Handle castling rook movement
        match mv.move_type {
            MoveType::KingCastle => {
                let (rook_from_sq, rook_to_sq) = if color_moving == Color::White {
                    (Square::H1, Square::F1)
                } else {
                    (Square::H8, Square::F8)
                };
                self.remove_piece(PieceType::Rook, color_moving, 1 << (rook_from_sq as u8));
                self.set_piece(PieceType::Rook, color_moving, 1 << (rook_to_sq as u8));
            }
            MoveType::QueenCastle => {
                let (rook_from_sq, rook_to_sq) = if color_moving == Color::White {
                    (Square::A1, Square::D1)
                } else {
                    (Square::A8, Square::D8)
                };
                self.remove_piece(PieceType::Rook, color_moving, 1 << (rook_from_sq as u8));
                self.set_piece(PieceType::Rook, color_moving, 1 << (rook_to_sq as u8));
            }
            _ => {}
        }

        // Update side to move
        self.side_to_move = self.side_to_move.opponent();

        // Update en passant square
        self.en_passant = if mv.move_type == MoveType::DoublePawnPush {
            if color_moving == Color::White {
                Some(unsafe { std::mem::transmute(mv.to as u8 - 8) })
            } else {
                Some(unsafe { std::mem::transmute(mv.to as u8 + 8) })
            }
        } else {
            None
        };

        // Update castling rights
        let mut new_rights = self.castling_rights;
        if piece_moving == PieceType::King {
            if color_moving == Color::White {
                new_rights &= !WHITE_KING_SIDE_CASTLE;
                new_rights &= !WHITE_QUEEN_SIDE_CASTLE;
            } else {
                new_rights &= !BLACK_KING_SIDE_CASTLE;
                new_rights &= !BLACK_QUEEN_SIDE_CASTLE;
            }
        }
        if from_bit == (1 << Square::A1 as u8) || to_bit == (1 << Square::A1 as u8) {
            new_rights &= !WHITE_QUEEN_SIDE_CASTLE;
        }
        if from_bit == (1 << Square::H1 as u8) || to_bit == (1 << Square::H1 as u8) {
            new_rights &= !WHITE_KING_SIDE_CASTLE;
        }
        if from_bit == (1 << Square::A8 as u8) || to_bit == (1 << Square::A8 as u8) {
            new_rights &= !BLACK_QUEEN_SIDE_CASTLE;
        }
        if from_bit == (1 << Square::H8 as u8) || to_bit == (1 << Square::H8 as u8) {
            new_rights &= !BLACK_KING_SIDE_CASTLE;
        }
        self.castling_rights = new_rights;

        self.update_hash_after_move(mv, piece_moving, color_moving);
    }

    pub fn make_move(&mut self, mv: &crate::movegen::Move) -> Undo {
        let undo = Undo {
            bitboards: self.bitboards,
            color_bitboards: self.color_bitboards,
            side_to_move: self.side_to_move,
            en_passant: self.en_passant,
            castling_rights: self.castling_rights,
            hash: self.hash,
            piece_on_square: self.piece_on_square,
            color_on_square: self.color_on_square,
        };
        self.apply_move(mv);
        undo
    }

    // Null move for null move pruning.
    pub fn make_move_null(&mut self) -> Undo {
        let undo = Undo {
            bitboards: self.bitboards,
            color_bitboards: self.color_bitboards,
            side_to_move: self.side_to_move,
            en_passant: self.en_passant,
            castling_rights: self.castling_rights,
            hash: self.hash,
            piece_on_square: self.piece_on_square,
            color_on_square: self.color_on_square,
        };
        if let Some(ep_sq) = self.en_passant {
            if let Some(ep_key) = zobrist::en_passant_key(ep_sq) {
                self.hash ^= ep_key;
            }
        }
        self.en_passant = None;
        self.side_to_move = self.side_to_move.opponent();
        self.hash ^= zobrist::side_to_move_key();
        undo
    }

    pub fn unmake_move(&mut self, undo: Undo) {
        self.bitboards = undo.bitboards;
        self.color_bitboards = undo.color_bitboards;
        self.side_to_move = undo.side_to_move;
        self.en_passant = undo.en_passant;
        self.castling_rights = undo.castling_rights;
        self.hash = undo.hash;
        self.piece_on_square = undo.piece_on_square;
        self.color_on_square = undo.color_on_square;
    }

    pub fn unmake_move_null(&mut self, undo: Undo) {
        self.unmake_move(undo);
    }

    // XOR out state components that will change before the move is applied.
    pub fn update_hash_before_move(&mut self, mv: &crate::movegen::Move) {
        self.hash ^= zobrist::side_to_move_key();

        if let Some(ep_sq) = self.en_passant {
            if let Some(ep_key) = zobrist::en_passant_key(ep_sq) {
                self.hash ^= ep_key;
            }
        }

        self.hash ^= zobrist::castling_key(self.castling_rights);

        let (piece_moving, color_moving) = self.get_piece_at(mv.from as u8).unwrap();
        self.hash ^= zobrist::piece_key(piece_moving, color_moving, mv.from as u8);

        if let Some((captured_piece, captured_color)) = self.get_piece_at(mv.to as u8) {
            self.hash ^= zobrist::piece_key(captured_piece, captured_color, mv.to as u8);
        } else if mv.move_type == MoveType::EnPassant {
            let captured_pawn_sq = if color_moving == Color::White {
                mv.to as u8 - 8
            } else {
                mv.to as u8 + 8
            };
            self.hash ^= zobrist::piece_key(PieceType::Pawn, color_moving.opponent(), captured_pawn_sq);
        }

        match mv.move_type {
            MoveType::KingCastle => {
                let rook_from_sq = if color_moving == Color::White { Square::H1 } else { Square::H8 };
                self.hash ^= zobrist::piece_key(PieceType::Rook, color_moving, rook_from_sq as u8);
            }
            MoveType::QueenCastle => {
                let rook_from_sq = if color_moving == Color::White { Square::A1 } else { Square::A8 };
                self.hash ^= zobrist::piece_key(PieceType::Rook, color_moving, rook_from_sq as u8);
            }
            _ => {}
        }
    }

    // XOR in state components after the move is applied.
    pub fn update_hash_after_move(
        &mut self,
        mv: &crate::movegen::Move,
        piece_moving: PieceType,
        color_moving: Color,
    ) {
        let piece_after_move = mv.promotion.unwrap_or(piece_moving);
        self.hash ^= zobrist::piece_key(piece_after_move, color_moving, mv.to as u8);

        match mv.move_type {
            MoveType::KingCastle => {
                let rook_to_sq = if color_moving == Color::White { Square::F1 } else { Square::F8 };
                self.hash ^= zobrist::piece_key(PieceType::Rook, color_moving, rook_to_sq as u8);
            }
            MoveType::QueenCastle => {
                let rook_to_sq = if color_moving == Color::White { Square::D1 } else { Square::D8 };
                self.hash ^= zobrist::piece_key(PieceType::Rook, color_moving, rook_to_sq as u8);
            }
            _ => {}
        }

        if let Some(ep_sq) = self.en_passant {
            if let Some(ep_key) = zobrist::en_passant_key(ep_sq) {
                self.hash ^= ep_key;
            }
        }

        self.hash ^= zobrist::castling_key(self.castling_rights);
    }

    pub fn calculate_hash(&self) -> u64 {
        let mut hash = 0;
        for color_idx in 0..2 {
            for piece_idx in 0..6 {
                let mut pieces = self.bitboards[piece_idx] & self.color_bitboards[color_idx];
                let piece_type: PieceType = unsafe { std::mem::transmute(piece_idx as u8) };
                let color: Color = unsafe { std::mem::transmute(color_idx as u8) };
                while pieces != 0 {
                    let sq = pieces.trailing_zeros() as u8;
                    hash ^= zobrist::piece_key(piece_type, color, sq);
                    pieces &= pieces - 1;
                }
            }
        }

        if self.side_to_move == Color::Black {
            hash ^= zobrist::side_to_move_key();
        }

        hash ^= zobrist::castling_key(self.castling_rights);

        if let Some(en_passant_sq) = self.en_passant {
            if let Some(ep_key) = zobrist::en_passant_key(en_passant_sq) {
                hash ^= ep_key;
            }
        }
        hash
    }

    pub fn to_fen(&self) -> String {
        let mut fen = String::new();

        for rank in (0..8).rev() {
            let mut empty_squares = 0;
            for file in 0..8 {
                let square_index = rank * 8 + file;
                if let Some((piece, color)) = self.get_piece_at(square_index) {
                    if empty_squares > 0 {
                        fen.push_str(&empty_squares.to_string());
                        empty_squares = 0;
                    }
                    let piece_char = match piece {
                        PieceType::Pawn => 'p',
                        PieceType::Knight => 'n',
                        PieceType::Bishop => 'b',
                        PieceType::Rook => 'r',
                        PieceType::Queen => 'q',
                        PieceType::King => 'k',
                    };
                    if color == Color::White {
                        fen.push(piece_char.to_ascii_uppercase());
                    } else {
                        fen.push(piece_char);
                    }
                } else {
                    empty_squares += 1;
                }
            }
            if empty_squares > 0 {
                fen.push_str(&empty_squares.to_string());
            }
            if rank > 0 {
                fen.push('/');
            }
        }

        fen.push(' ');
        fen.push(if self.side_to_move == Color::White { 'w' } else { 'b' });
        fen.push(' ');

        let mut castling_str = String::new();
        if self.castling_rights & WHITE_KING_SIDE_CASTLE != 0 { castling_str.push('K'); }
        if self.castling_rights & WHITE_QUEEN_SIDE_CASTLE != 0 { castling_str.push('Q'); }
        if self.castling_rights & BLACK_KING_SIDE_CASTLE != 0 { castling_str.push('k'); }
        if self.castling_rights & BLACK_QUEEN_SIDE_CASTLE != 0 { castling_str.push('q'); }
        if castling_str.is_empty() {
            fen.push('-');
        } else {
            fen.push_str(&castling_str);
        }

        fen.push(' ');
        if let Some(sq) = self.en_passant {
            fen.push_str(&Self::square_to_string(sq));
        } else {
            fen.push('-');
        }

        fen.push_str(" 0 1");
        fen
    }

    pub fn square_to_string(square: Square) -> String {
        let sq_idx = square as u8;
        let file = (sq_idx % 8) as u8 + b'a';
        let rank = (sq_idx / 8) as u8 + b'1';
        format!("{}{}", file as char, rank as char)
    }

    pub fn is_in_check(&self) -> bool {
        let king_bb = self.bitboards[PieceType::King as usize]
            & self.color_bitboards[self.side_to_move as usize];
        if king_bb == 0 {
            return false;
        }
        let king_sq = king_bb.trailing_zeros() as u8;
        self.is_square_attacked(king_sq, self.side_to_move.opponent())
    }

    pub fn is_square_attacked(&self, sq: u8, attacker_color: Color) -> bool {
        let all_pieces = self.color_bitboards[0] | self.color_bitboards[1];

        if (attacks::get_pawn_attacks(sq, attacker_color.opponent())
            & self.bitboards[PieceType::Pawn as usize]
            & self.color_bitboards[attacker_color as usize]) != 0 { return true; }
        if (attacks::get_knight_attacks(sq)
            & self.bitboards[PieceType::Knight as usize]
            & self.color_bitboards[attacker_color as usize]) != 0 { return true; }
        if (attacks::get_king_attacks(sq)
            & self.bitboards[PieceType::King as usize]
            & self.color_bitboards[attacker_color as usize]) != 0 { return true; }
        if (attacks::get_rook_attacks(sq, all_pieces)
            & (self.bitboards[PieceType::Rook as usize] | self.bitboards[PieceType::Queen as usize])
            & self.color_bitboards[attacker_color as usize]) != 0 { return true; }
        if (attacks::get_bishop_attacks(sq, all_pieces)
            & (self.bitboards[PieceType::Bishop as usize] | self.bitboards[PieceType::Queen as usize])
            & self.color_bitboards[attacker_color as usize]) != 0 { return true; }

        false
    }
}

impl Default for Board {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Display for Board {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let mut board_str = String::new();
        for rank in (0..8).rev() {
            for file in 0..8 {
                let square_index = rank * 8 + file;
                if let Some((piece, color)) = self.get_piece_at(square_index) {
                    let piece_char = match piece {
                        PieceType::Pawn => 'p',
                        PieceType::Knight => 'n',
                        PieceType::Bishop => 'b',
                        PieceType::Rook => 'r',
                        PieceType::Queen => 'q',
                        PieceType::King => 'k',
                    };
                    if color == Color::White {
                        board_str.push(piece_char.to_ascii_uppercase());
                    } else {
                        board_str.push(piece_char);
                    }
                } else {
                    board_str.push('.');
                }
                board_str.push(' ');
            }
            board_str.push('\n');
        }
        write!(f, "{}", board_str)
    }
}
