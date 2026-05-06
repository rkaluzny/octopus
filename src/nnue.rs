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

use std::fs;
use std::path::Path;

use crate::accumulator::{NnueState, NnueUndo, PieceDelta};
use crate::board::{Board, Color, PieceType};
use crate::features::{ACCUMULATOR_SIZE, HIDDEN_SIZE, INPUT_FEATURES};
use crate::movegen::{Move, MoveType};

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

// Default path for NNUE weights
pub const DEFAULT_WEIGHTS_PATH: &str = "output/nnue_weights.bin";
const FEATURE_SCALE: f32 = 127.0;
const HIDDEN_SCALE: f32 = 64.0;
const OUTPUT_SCALE: f32 = 128.0;
const CP_SCALE: f32 = 600.0;

const FEATURE_WEIGHT_COUNT: usize = INPUT_FEATURES * ACCUMULATOR_SIZE;
const HIDDEN_WEIGHT_COUNT: usize = HIDDEN_SIZE * (ACCUMULATOR_SIZE * 2);

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
enum EvalBackend {
    Scalar,
    Sse2,
    #[cfg(nnue_level_v3)]
    Avx2,
}

fn select_backend() -> EvalBackend {
    #[cfg(target_arch = "x86_64")]
    {
        #[cfg(nnue_level_v2)]
        {
            if is_x86_feature_detected!("sse2") {
                return EvalBackend::Sse2;
            }
            return EvalBackend::Scalar;
        }
        #[cfg(nnue_level_v3)]
        {
            if is_x86_feature_detected!("avx2") {
                return EvalBackend::Avx2;
            }
            if is_x86_feature_detected!("sse2") {
                return EvalBackend::Sse2;
            }
            return EvalBackend::Scalar;
        }
    }

    #[cfg(not(target_arch = "x86_64"))]
    {
        EvalBackend::Scalar
    }
}

#[repr(align(64))]
pub struct NnueWeights {
    feature_weights: Vec<i8>,
    hidden_weights: Vec<i16>,
    hidden_bias: Vec<i16>,
    output_weights: Vec<i16>,
    output_bias: i16,
    feature_scale: f32,
    hidden_scale: f32,
    output_scale: f32,
    cp_scale: f32,
}

impl NnueWeights {
    fn zeroed() -> Self {
        Self {
            feature_weights: vec![0i8; FEATURE_WEIGHT_COUNT],
            hidden_weights: vec![0i16; HIDDEN_WEIGHT_COUNT],
            hidden_bias: vec![0i16; HIDDEN_SIZE],
            output_weights: vec![0i16; HIDDEN_SIZE],
            output_bias: 0,
            feature_scale: FEATURE_SCALE,
            hidden_scale: HIDDEN_SCALE,
            output_scale: OUTPUT_SCALE,
            cp_scale: CP_SCALE,
        }
    }

    pub fn load_from_path(path: impl AsRef<Path>) -> Option<Self> {
        let bytes = fs::read(path).ok()?;
        fn read_u32(bytes: &[u8], cursor: &mut usize) -> Option<u32> {
            let end = *cursor + 4;
            let chunk = bytes.get(*cursor..end)?;
            *cursor = end;
            Some(u32::from_le_bytes(chunk.try_into().ok()?))
        }

        fn read_f32(bytes: &[u8], cursor: &mut usize) -> Option<f32> {
            let end = *cursor + 4;
            let chunk = bytes.get(*cursor..end)?;
            *cursor = end;
            Some(f32::from_le_bytes(chunk.try_into().ok()?))
        }

        fn read_i8_vec(bytes: &[u8], cursor: &mut usize, len: usize) -> Option<Vec<i8>> {
            let end = *cursor + len;
            let chunk = bytes.get(*cursor..end)?;
            *cursor = end;
            Some(chunk.iter().map(|b| *b as i8).collect())
        }

        fn read_i16_vec(bytes: &[u8], cursor: &mut usize, len: usize) -> Option<Vec<i16>> {
            let byte_len = len.checked_mul(2)?;
            let end = *cursor + byte_len;
            let chunk = bytes.get(*cursor..end)?;
            *cursor = end;
            let mut out = Vec::with_capacity(len);
            for item in chunk.chunks_exact(2) {
                out.push(i16::from_le_bytes([item[0], item[1]]));
            }
            Some(out)
        }

        if bytes.get(0..4)? != b"ONUE" {
            return None;
        }

        let mut cursor = 4usize;
        let version = read_u32(&bytes, &mut cursor)?;
        if version != 1 {
            return None;
        }

        let input_features = read_u32(&bytes, &mut cursor)? as usize;
        let accumulator_size = read_u32(&bytes, &mut cursor)? as usize;
        let hidden_size = read_u32(&bytes, &mut cursor)? as usize;
        let _output_size = read_u32(&bytes, &mut cursor)? as usize;
        let feature_scale = read_f32(&bytes, &mut cursor)?;
        let hidden_scale = read_f32(&bytes, &mut cursor)?;
        let output_scale = read_f32(&bytes, &mut cursor)?;
        let cp_scale = read_f32(&bytes, &mut cursor)?;

        let feature_count = input_features.checked_mul(accumulator_size)?;
        let hidden_count = hidden_size.checked_mul(accumulator_size.checked_mul(2)?)?;
        let feature_weights_raw = read_i8_vec(&bytes, &mut cursor, feature_count)?;
        let hidden_weights_raw = read_i16_vec(&bytes, &mut cursor, hidden_count)?;
        let hidden_bias_raw = read_i16_vec(&bytes, &mut cursor, hidden_size)?;
        let output_weights_raw = read_i16_vec(&bytes, &mut cursor, hidden_size)?;
        let output_bias_raw = read_i16_vec(&bytes, &mut cursor, 1)?;

        if cursor != bytes.len() {
            return None;
        }

        Some(Self {
            feature_weights: feature_weights_raw,
            hidden_weights: hidden_weights_raw,
            hidden_bias: hidden_bias_raw,
            output_weights: output_weights_raw,
            output_bias: output_bias_raw[0],
            feature_scale,
            hidden_scale,
            output_scale,
            cp_scale,
        })
    }

    #[inline(always)]
    pub fn feature_weights_for_feature(&self, feature_index: usize) -> &[i8] {
        let start = feature_index * ACCUMULATOR_SIZE;
        &self.feature_weights[start..start + ACCUMULATOR_SIZE]
    }

    #[inline(always)]
    pub fn hidden_weights_for_neuron(&self, neuron: usize) -> &[i16] {
        let start = neuron * (ACCUMULATOR_SIZE * 2);
        &self.hidden_weights[start..start + (ACCUMULATOR_SIZE * 2)]
    }
}

pub struct NnueEvaluator {
    pub weights: NnueWeights,
    state: NnueState,
    undo_stack: Vec<NnueUndo>,
    initialized: bool,
    weights_path: String,
    pub weights_loaded: bool,
    backend: EvalBackend,
}

impl NnueEvaluator {
    pub fn new<P: Into<String>>(path: P) -> Self {
        let path_str = path.into();

        if let Some(weights) = NnueWeights::load_from_path(&path_str) {
            return Self {
                weights,
                state: NnueState::default(),
                undo_stack: Vec::with_capacity(128),
                initialized: false,
                weights_path: path_str,
                weights_loaded: true,
                backend: select_backend(),
            };
        }

        if let Ok(exe_path) = std::env::current_exe() {
            if let Some(exe_dir) = exe_path.parent() {
                let path_buf = exe_dir.join(&path_str);
                if let Some(weights) = NnueWeights::load_from_path(path_buf.to_str().unwrap_or("")) {
                    return Self {
                        weights,
                        state: NnueState::default(),
                        undo_stack: Vec::with_capacity(128),
                        initialized: false,
                        weights_path: path_buf.to_string_lossy().into_owned(),
                        weights_loaded: true,
                        backend: select_backend(),
                    };
                }
            }
        }

        Self {
            weights: NnueWeights::zeroed(),
            state: NnueState::default(),
            undo_stack: Vec::with_capacity(128),
            initialized: false,
            weights_path: path_str,
            weights_loaded: false,
            backend: select_backend(),
        }
    }

    pub fn set_weights_path<P: Into<String>>(&mut self, path: P) {
        self.weights_path = path.into();
    }

    pub fn reload(&mut self) {
        eprintln!("NnueEvaluator: Trying to load weights from: {}", self.weights_path);

        if let Some(weights) = NnueWeights::load_from_path(&self.weights_path) {
            self.weights = weights;
            self.weights_loaded = true;
            self.initialized = false;
            self.backend = select_backend();
            eprintln!("NnueEvaluator: Weights loaded successfully from specified path");
            return;
        }

        eprintln!("NnueEvaluator: Failed to load from specified path, trying executable directory");

        if let Ok(exe_path) = std::env::current_exe() {
            eprintln!("NnueEvaluator: Executable path: {:?}", exe_path);
            if let Some(exe_dir) = exe_path.parent() {
                let path_buf = exe_dir.join(&self.weights_path);
                eprintln!("NnueEvaluator: Trying executable-relative path: {:?}", path_buf);
                if let Some(weights) = NnueWeights::load_from_path(path_buf.to_str().unwrap_or("")) {
                    self.weights = weights;
                    self.weights_loaded = true;
                    self.initialized = false;
                    self.backend = select_backend();
                    self.weights_path = path_buf.to_string_lossy().into_owned();
                    eprintln!("NnueEvaluator: Weights loaded successfully from executable-relative path");
                    return;
                }
            }
        }

        eprintln!("NnueEvaluator: Failed to load weights, using zeroed weights");
    }

    pub fn reset(&mut self, board: &Board) {
        self.state.rebuild_from_board(board, &self.weights);
        self.undo_stack.clear();
        self.initialized = true;
    }

    pub fn disable(&mut self) {
        self.initialized = false;
    }

    pub fn is_ready(&self) -> bool {
        self.initialized
    }

    fn update_state_for_move(&mut self, board: &Board, mv: &Move) {
        let (moving_piece, moving_color) = match board.get_piece_at(mv.from as u8) {
            Some(v) => v,
            None => return,
        };

        let mut deltas = [PieceDelta::default(); 8];
        let mut len = 0usize;

        deltas[len] = PieceDelta {
            piece_color: moving_color,
            piece: moving_piece,
            square: mv.from as u8,
            sign: -1,
        };
        len += 1;

        if let Some((captured_piece, captured_color)) = board.get_piece_at(mv.to as u8) {
            deltas[len] = PieceDelta {
                piece_color: captured_color,
                piece: captured_piece,
                square: mv.to as u8,
                sign: -1,
            };
            len += 1;
        } else if mv.move_type == MoveType::EnPassant {
            let capture_sq = if moving_color == Color::White {
                mv.to as u8 - 8
            } else {
                mv.to as u8 + 8
            };
            deltas[len] = PieceDelta {
                piece_color: moving_color.opponent(),
                piece: PieceType::Pawn,
                square: capture_sq,
                sign: -1,
            };
            len += 1;
        }

        let piece_after_move = mv.promotion.unwrap_or(moving_piece);
        deltas[len] = PieceDelta {
            piece_color: moving_color,
            piece: piece_after_move,
            square: mv.to as u8,
            sign: 1,
        };
        len += 1;

        match mv.move_type {
            MoveType::KingCastle => {
                let (rook_from, rook_to) = if moving_color == Color::White {
                    (63u8, 61u8)
                } else {
                    (7u8, 5u8)
                };
                deltas[len] = PieceDelta {
                    piece_color: moving_color,
                    piece: PieceType::Rook,
                    square: rook_from,
                    sign: -1,
                };
                len += 1;
                deltas[len] = PieceDelta {
                    piece_color: moving_color,
                    piece: PieceType::Rook,
                    square: rook_to,
                    sign: 1,
                };
                len += 1;
            }
            MoveType::QueenCastle => {
                let (rook_from, rook_to) = if moving_color == Color::White {
                    (56u8, 59u8)
                } else {
                    (0u8, 3u8)
                };
                deltas[len] = PieceDelta {
                    piece_color: moving_color,
                    piece: PieceType::Rook,
                    square: rook_from,
                    sign: -1,
                };
                len += 1;
                deltas[len] = PieceDelta {
                    piece_color: moving_color,
                    piece: PieceType::Rook,
                    square: rook_to,
                    sign: 1,
                };
                len += 1;
            }
            _ => {}
        }

        let mut undo = NnueUndo {
            previous_state: None,
            deltas,
            delta_len: len as u8,
        };

        if moving_piece == PieceType::King {
            undo.previous_state = Some(self.state.clone());
            let mut temp = board.clone();
            temp.apply_move(mv);
            if moving_color == Color::White {
                self.state.white_king_sq = temp.king_square(Color::White);
                self.state.black_king_sq = board.king_square(Color::Black);
                self.state.apply_piece_deltas(&deltas[..len], &self.weights);
                self.rebuild_side(&temp, Color::White);
            } else {
                self.state.white_king_sq = board.king_square(Color::White);
                self.state.black_king_sq = temp.king_square(Color::Black);
                self.state.apply_piece_deltas(&deltas[..len], &self.weights);
                self.rebuild_side(&temp, Color::Black);
            }
        } else {
            self.state.apply_piece_deltas(&deltas[..len], &self.weights);
        }

        self.undo_stack.push(undo);
    }

    fn rebuild_side(&mut self, board: &Board, side: Color) {
        let acc = match side {
            Color::White => &mut self.state.white,
            Color::Black => &mut self.state.black,
        };
        acc.fill(0);
        let king_sq = self.state.king_square(side);
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
                    self.state.apply_piece_delta_for_side_with_king_sq(
                        side,
                        king_sq,
                        color,
                        piece,
                        square,
                        1,
                        &self.weights,
                    );
                    pieces &= pieces - 1;
                }
            }
        }
    }

    pub fn apply_move(&mut self, board: &Board, mv: &Move) {
        if !self.initialized {
            return;
        }
        self.update_state_for_move(board, mv);
    }

    pub fn unapply_move(&mut self, _board: &Board, _mv: &Move) {
        if !self.initialized {
            return;
        }
        if let Some(undo) = self.undo_stack.pop() {
            if let Some(prev) = undo.previous_state {
                self.state = prev;
            } else {
                for i in (0..undo.delta_len as usize).rev() {
                    let delta = undo.deltas[i];
                    self.state.apply_piece_delta(
                        delta.piece_color,
                        delta.piece,
                        delta.square,
                        -delta.sign,
                        &self.weights,
                    );
                }
            }
        }
    }

    pub fn evaluate(&self, board: &Board) -> i32 {
        if !self.initialized {
            return 0;
        }

        let acc = self.state.current(board.side_to_move);

        let feature_scale = self.weights.feature_scale.max(1.0) as f64;
        let hidden_scale = self.weights.hidden_scale.max(1.0) as f64;
        let output_scale = self.weights.output_scale.max(1.0) as f64;
        let cp_scale = self.weights.cp_scale.max(1.0) as f64;
        let hidden_divisor = feature_scale * hidden_scale;
        let hidden_bias_divisor = hidden_scale;
        let output_factor = cp_scale / output_scale;

        // Stack allocate the clamped accumulators
        let mut activated_acc: [i16; ACCUMULATOR_SIZE] = [0; ACCUMULATOR_SIZE];
        let mut mirrored_acc: [i16; ACCUMULATOR_SIZE] = [0; ACCUMULATOR_SIZE];
        
        // Clamp values once upfront - vectorized where possible
        match self.backend {
            #[cfg(nnue_level_v3)]
            EvalBackend::Avx2 => {
                #[cfg(target_arch = "x86_64")]
                unsafe { clamp_avx2(acc, &mut activated_acc, &mut mirrored_acc); }
                #[cfg(not(target_arch = "x86_64"))]
                { clamp_scalar(acc, &mut activated_acc, &mut mirrored_acc); }
            }
            EvalBackend::Sse2 => {
                #[cfg(target_arch = "x86_64")]
                unsafe { clamp_sse2(acc, &mut activated_acc, &mut mirrored_acc); }
                #[cfg(not(target_arch = "x86_64"))]
                { clamp_scalar(acc, &mut activated_acc, &mut mirrored_acc); }
            }
            EvalBackend::Scalar => clamp_scalar(acc, &mut activated_acc, &mut mirrored_acc),
        }

        let mut hidden = [0i16; HIDDEN_SIZE];
        for neuron in 0..HIDDEN_SIZE {
            let weights = self.weights.hidden_weights_for_neuron(neuron);
            let dot = match self.backend {
                #[cfg(nnue_level_v3)]
                EvalBackend::Avx2 => {
                    #[cfg(target_arch = "x86_64")]
                    unsafe {
                        dot_product_avx2(&activated_acc, &weights[..ACCUMULATOR_SIZE])
                            + dot_product_avx2(&mirrored_acc, &weights[ACCUMULATOR_SIZE..])
                    }
                    #[cfg(not(target_arch = "x86_64"))]
                    {
                        dot_product_scalar(&activated_acc, &weights[..ACCUMULATOR_SIZE])
                            + dot_product_scalar(&mirrored_acc, &weights[ACCUMULATOR_SIZE..])
                    }
                }
                EvalBackend::Sse2 => {
                    #[cfg(target_arch = "x86_64")]
                    unsafe {
                        dot_product_sse2(&activated_acc, &weights[..ACCUMULATOR_SIZE])
                            + dot_product_sse2(&mirrored_acc, &weights[ACCUMULATOR_SIZE..])
                    }
                    #[cfg(not(target_arch = "x86_64"))]
                    {
                        dot_product_scalar(&activated_acc, &weights[..ACCUMULATOR_SIZE])
                            + dot_product_scalar(&mirrored_acc, &weights[ACCUMULATOR_SIZE..])
                    }
                }
                EvalBackend::Scalar => {
                    dot_product_scalar(&activated_acc, &weights[..ACCUMULATOR_SIZE])
                        + dot_product_scalar(&mirrored_acc, &weights[ACCUMULATOR_SIZE..])
                }
            };

            let bias = self.weights.hidden_bias[neuron] as f64;
            let value = ((dot as f64) / hidden_divisor + bias / hidden_bias_divisor).round() as i32;
            hidden[neuron] = value.clamp(0, 127) as i16;
        }

        let out = match self.backend {
            #[cfg(nnue_level_v3)]
            EvalBackend::Avx2 => {
                #[cfg(target_arch = "x86_64")]
                unsafe { dot_product_i16_avx2(&hidden, &self.weights.output_weights) }
                #[cfg(not(target_arch = "x86_64"))]
                {
                    dot_product_scalar(&hidden, &self.weights.output_weights)
                }
            }
            EvalBackend::Sse2 => {
                #[cfg(target_arch = "x86_64")]
                unsafe { dot_product_i16_sse2(&hidden, &self.weights.output_weights) }
                #[cfg(not(target_arch = "x86_64"))]
                {
                    dot_product_scalar(&hidden, &self.weights.output_weights)
                }
            }
            EvalBackend::Scalar => dot_product_scalar(&hidden, &self.weights.output_weights),
        } + self.weights.output_bias as i64;

        let cp_score = ((out as f64) * output_factor).round() as i32;
        cp_score.clamp(-30_000, 30_000)
    }
}

fn dot_product_scalar(lhs: &[i16], rhs: &[i16]) -> i64 {
    lhs.iter()
        .zip(rhs.iter())
        .fold(0i64, |acc, (&a, &b)| acc + a as i64 * b as i64)
}

#[inline(always)]
fn clamp_scalar(acc: &[i32], activated: &mut [i16; ACCUMULATOR_SIZE], mirrored: &mut [i16; ACCUMULATOR_SIZE]) {
    for i in 0..ACCUMULATOR_SIZE {
        let clamped = acc[i].clamp(0, 127) as i16;
        activated[i] = clamped;
        mirrored[ACCUMULATOR_SIZE - 1 - i] = clamped;
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse2")]
unsafe fn clamp_sse2(acc: &[i32], activated: &mut [i16; ACCUMULATOR_SIZE], mirrored: &mut [i16; ACCUMULATOR_SIZE]) {
    clamp_scalar(acc, activated, mirrored);
}

#[cfg(all(target_arch = "x86_64", nnue_level_v3))]
#[target_feature(enable = "avx2")]
unsafe fn clamp_avx2(acc: &[i32], activated: &mut [i16; ACCUMULATOR_SIZE], mirrored: &mut [i16; ACCUMULATOR_SIZE]) {
    clamp_scalar(acc, activated, mirrored);
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse2")]
unsafe fn dot_product_sse2(lhs: &[i16], rhs: &[i16]) -> i64 {
    let mut sum = _mm_setzero_si128();
    let mut i = 0usize;
    while i < lhs.len() {
        let a = _mm_loadu_si128(lhs.as_ptr().add(i) as *const __m128i);
        let b = _mm_loadu_si128(rhs.as_ptr().add(i) as *const __m128i);
        sum = _mm_add_epi32(sum, _mm_madd_epi16(a, b));
        i += 8;
    }

    let mut lanes = [0i32; 4];
    _mm_storeu_si128(lanes.as_mut_ptr() as *mut __m128i, sum);
    lanes.iter().map(|&lane| lane as i64).sum()
}

#[cfg(all(target_arch = "x86_64", nnue_level_v3))]
#[target_feature(enable = "avx2")]
unsafe fn dot_product_avx2(lhs: &[i16], rhs: &[i16]) -> i64 {
    let mut sum = _mm256_setzero_si256();
    let mut i = 0usize;
    while i < lhs.len() {
        let a = _mm256_loadu_si256(lhs.as_ptr().add(i) as *const __m256i);
        let b = _mm256_loadu_si256(rhs.as_ptr().add(i) as *const __m256i);
        sum = _mm256_add_epi32(sum, _mm256_madd_epi16(a, b));
        i += 16;
    }

    let mut lanes = [0i32; 8];
    _mm256_storeu_si256(lanes.as_mut_ptr() as *mut __m256i, sum);
    lanes.iter().map(|&lane| lane as i64).sum()
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse2")]
unsafe fn dot_product_i16_sse2(lhs: &[i16], rhs: &[i16]) -> i64 {
    dot_product_sse2(lhs, rhs)
}

#[cfg(all(target_arch = "x86_64", nnue_level_v3))]
#[target_feature(enable = "avx2")]
unsafe fn dot_product_i16_avx2(lhs: &[i16], rhs: &[i16]) -> i64 {
    dot_product_avx2(lhs, rhs)
}
