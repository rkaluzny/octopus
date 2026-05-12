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
// =============================================================================

use std::fs;
use std::path::Path;
use std::time::Instant;

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
const HIDDEN_INPUT_SIZE: usize = ACCUMULATOR_SIZE;
const HIDDEN_WEIGHT_COUNT: usize = HIDDEN_SIZE * HIDDEN_INPUT_SIZE;

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
    output_factor: f32,
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
            output_factor: CP_SCALE / OUTPUT_SCALE,
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
        let output_factor = cp_scale / output_scale;

        let feature_count = input_features.checked_mul(accumulator_size)?;
        let hidden_count = hidden_size.checked_mul(accumulator_size.checked_mul(2)?)?;
        let feature_weights_raw = read_i8_vec(&bytes, &mut cursor, feature_count)?;
        let hidden_weights_raw = read_i16_vec(&bytes, &mut cursor, hidden_count)?;
        
        // Transform 1024-wide hidden weights to 512-wide by combining mirrored pairs.
        // Original format: hidden_inputs[i] == hidden_inputs[1023-i], so we combine
        // weights[i] + weights[1023-i] for each pair.
        let mut hidden_weights = Vec::with_capacity(HIDDEN_SIZE * HIDDEN_INPUT_SIZE);
        for neuron in 0..HIDDEN_SIZE {
            let base = neuron * (ACCUMULATOR_SIZE * 2);
            for i in 0..ACCUMULATOR_SIZE {
                let w1 = hidden_weights_raw[base + i] as i32;
                let w2 = hidden_weights_raw[base + (ACCUMULATOR_SIZE * 2 - 1 - i)] as i32;
                hidden_weights.push((w1 + w2) as i16);
            }
        }
        
        let hidden_bias_raw = read_i16_vec(&bytes, &mut cursor, hidden_size)?;
        let output_weights_raw = read_i16_vec(&bytes, &mut cursor, hidden_size)?;
        let output_bias_raw = read_i16_vec(&bytes, &mut cursor, 1)?;

        if cursor != bytes.len() {
            return None;
        }

        Some(Self {
            feature_weights: feature_weights_raw,
            hidden_weights,
            hidden_bias: hidden_bias_raw,
            output_weights: output_weights_raw,
            output_bias: output_bias_raw[0],
            feature_scale,
            hidden_scale,
            output_scale,
            cp_scale,
            output_factor,
        })
    }

    #[inline(always)]
    pub fn feature_weights_for_feature(&self, feature_index: usize) -> &[i8] {
        let start = feature_index * ACCUMULATOR_SIZE;
        &self.feature_weights[start..start + ACCUMULATOR_SIZE]
    }

    #[inline(always)]
    pub fn hidden_weights_for_neuron(&self, neuron: usize) -> &[i16] {
        let start = neuron * HIDDEN_INPUT_SIZE;
        &self.hidden_weights[start..start + HIDDEN_INPUT_SIZE]
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum EvalBackend {
    Scalar,
    Sse2,
    Avx2,
}

fn select_backend() -> EvalBackend {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            return EvalBackend::Avx2;
        }
        if is_x86_feature_detected!("sse2") {
            return EvalBackend::Sse2;
        }
    }
    EvalBackend::Scalar
}

pub struct NnueEvaluator {
    pub weights: NnueWeights,
    state: NnueState,
    undo_stack: Vec<NnueUndo>,
    initialized: bool,
    weights_path: String,
    pub weights_loaded: bool,
    pub backend: EvalBackend,
}

impl NnueEvaluator {
    pub fn new<P: Into<String>>(path: P) -> Self {
        let path_str = path.into();
        let backend = select_backend();

        eprintln!("NNUE backend selected: {:?}", backend);

        if let Some(weights) = NnueWeights::load_from_path(&path_str) {
            return Self {
                weights,
                state: NnueState::default(),
                undo_stack: Vec::with_capacity(128),
                initialized: false,
                weights_path: path_str,
                weights_loaded: true,
                backend,
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
                        backend,
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
            backend,
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
                    self.weights_path = path_buf.to_string_lossy().into_owned();
                    eprintln!("NnueEvaluator: Weights loaded successfully from executable-relative path");
                    return;
                }
            }
        }

        eprintln!("NnueEvaluator: Failed to load weights, disabling NNUE");
        self.weights = NnueWeights::zeroed();
        self.weights_loaded = false;
        self.initialized = false;
        self.undo_stack.clear();
    }

    pub fn reset(&mut self, board: &Board) {
        if !self.weights_loaded {
            self.undo_stack.clear();
            self.initialized = false;
            return;
        }
        self.state.rebuild_from_board(board, &self.weights, self.backend);
        self.undo_stack.clear();
        self.initialized = true;
    }

    pub fn disable(&mut self) {
        self.initialized = false;
    }

    pub fn is_ready(&self) -> bool {
        self.initialized && self.weights_loaded
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
        let table = crate::features::feature_index_table();

        if moving_piece == PieceType::King {
            undo.previous_state = Some(self.state.clone());
            if moving_color == Color::White {
                self.state.white_king_sq = mv.to as u8;
                self.state.black_king_sq = board.king_square(Color::Black);
                self.state.apply_piece_deltas(&deltas[..len], &self.weights, self.backend, table);
                self.state.rebuild_side_after_king_move(
                    board,
                    Color::White,
                    mv.to as u8,
                    moving_color,
                    mv,
                    &self.weights,
                    self.backend,
                );
            } else {
                self.state.white_king_sq = board.king_square(Color::White);
                self.state.black_king_sq = mv.to as u8;
                self.state.apply_piece_deltas(&deltas[..len], &self.weights, self.backend, table);
                self.state.rebuild_side_after_king_move(
                    board,
                    Color::Black,
                    mv.to as u8,
                    moving_color,
                    mv,
                    &self.weights,
                    self.backend,
                );
            }
        } else {
            self.state.apply_piece_deltas(&deltas[..len], &self.weights, self.backend, table);
        }

        self.undo_stack.push(undo);
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
                let table = crate::features::feature_index_table();
                for i in (0..undo.delta_len as usize).rev() {
                    let delta = undo.deltas[i];
                    self.state.apply_piece_delta(
                        delta.piece_color,
                        delta.piece,
                        delta.square,
                        -delta.sign,
                        &self.weights,
                        self.backend,
                        table,
                    );
                }
            }
        }
    }

    pub fn evaluate(&self, board: &Board) -> i32 {
        if !self.initialized || !self.weights_loaded {
            return 0;
        }

        let acc = self.state.current(board.side_to_move);
        let output_factor = self.weights.output_factor;

        // Pack the accumulator into the hidden input vector using SIMD if available.
        let mut hidden_inputs = [0i16; HIDDEN_INPUT_SIZE];
        match self.backend {
            #[cfg(target_arch = "x86_64")]
            EvalBackend::Avx2 => unsafe { clamp_avx2(acc, &mut hidden_inputs); }
            #[cfg(target_arch = "x86_64")]
            EvalBackend::Sse2 => unsafe { clamp_sse2(acc, &mut hidden_inputs); }
            _ => clamp_scalar(acc, &mut hidden_inputs),
        }

        // Fast integer arithmetic for hidden layer - compute all 256 neurons.
        let mut hidden = [0i16; HIDDEN_SIZE];
        match self.backend {
            EvalBackend::Avx2 => unsafe {
                compute_hidden_layer_avx2(&self.weights, &hidden_inputs, &mut hidden);
            }
            EvalBackend::Sse2 => unsafe {
                compute_hidden_layer_sse2(&self.weights, &hidden_inputs, &mut hidden);
            }
            EvalBackend::Scalar => {
                compute_hidden_layer_scalar(&self.weights, &hidden_inputs, &mut hidden);
            }
        }

        // Output layer - use backend-specific dot product
        let out = match self.backend {
            EvalBackend::Avx2 => unsafe { dot_product_avx2(&hidden, &self.weights.output_weights) }
            EvalBackend::Sse2 => unsafe { dot_product_sse2(&hidden, &self.weights.output_weights) }
            EvalBackend::Scalar => dot_product_scalar(&hidden, &self.weights.output_weights),
        } + self.weights.output_bias as i64;

        let cp_score = ((out as f64) * (output_factor as f64)).round() as i32;
        cp_score.clamp(-30_000, 30_000)
    }

    /// Run a single evaluation with detailed timing breakdown for profiling.
    /// Returns (score, clamp_time_ns, hidden_time_ns, output_time_ns)
    pub fn profile_evaluate(&self, board: &Board) -> (i32, u64, u64, u64) {
        if !self.initialized || !self.weights_loaded {
            return (0, 0, 0, 0);
        }

        let acc = self.state.current(board.side_to_move);
        let output_factor = self.weights.output_factor;

        let mut hidden_inputs = [0i16; HIDDEN_INPUT_SIZE];

        // Time clamping
        let clamp_start = Instant::now();
        match self.backend {
            #[cfg(target_arch = "x86_64")]
            EvalBackend::Avx2 => unsafe { clamp_avx2(acc, &mut hidden_inputs); }
            #[cfg(target_arch = "x86_64")]
            EvalBackend::Sse2 => unsafe { clamp_sse2(acc, &mut hidden_inputs); }
            _ => clamp_scalar(acc, &mut hidden_inputs),
        }
        let clamp_time = clamp_start.elapsed().as_nanos() as u64;

        // Time hidden layer
        let mut hidden = [0i16; HIDDEN_SIZE];
        let hidden_start = Instant::now();
        match self.backend {
            EvalBackend::Avx2 => unsafe {
                compute_hidden_layer_avx2(&self.weights, &hidden_inputs, &mut hidden);
            }
            EvalBackend::Sse2 => unsafe {
                compute_hidden_layer_sse2(&self.weights, &hidden_inputs, &mut hidden);
            }
            EvalBackend::Scalar => {
                compute_hidden_layer_scalar(&self.weights, &hidden_inputs, &mut hidden);
            }
        }
        let hidden_time = hidden_start.elapsed().as_nanos() as u64;

        // Time output layer
        let out_start = Instant::now();
        let out = match self.backend {
            EvalBackend::Avx2 => unsafe { dot_product_avx2(&hidden, &self.weights.output_weights) }
            EvalBackend::Sse2 => unsafe { dot_product_sse2(&hidden, &self.weights.output_weights) }
            EvalBackend::Scalar => dot_product_scalar(&hidden, &self.weights.output_weights),
        } + self.weights.output_bias as i64;
        let output_time = out_start.elapsed().as_nanos() as u64;

        let cp_score = ((out as f64) * (output_factor as f64)).round() as i32;
        let score = cp_score.clamp(-30_000, 30_000);

        (score, clamp_time, hidden_time, output_time)
    }
}

#[inline(always)]
fn dot_product_scalar(lhs: &[i16], rhs: &[i16]) -> i64 {
    lhs.iter()
        .zip(rhs.iter())
        .fold(0i64, |acc, (&a, &b)| acc + (a as i64) * (b as i64))
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse2")]
unsafe fn dot_product_sse2(lhs: &[i16], rhs: &[i16]) -> i64 {
    let mut sum = _mm_setzero_si128();
    let mut i = 0usize;

    while i + 8 <= lhs.len() {
        let a = _mm_loadu_si128(lhs.as_ptr().add(i) as *const __m128i);
        let b = _mm_loadu_si128(rhs.as_ptr().add(i) as *const __m128i);
        sum = _mm_add_epi32(sum, _mm_madd_epi16(a, b));
        i += 8;
    }

    let mut total = horizontal_sum_i32_sse2(sum);
    while i < lhs.len() {
        total += (lhs[i] as i64) * (rhs[i] as i64);
        i += 1;
    }

    total
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn dot_product_avx2(lhs: &[i16], rhs: &[i16]) -> i64 {
    let mut sum = _mm256_setzero_si256();
    let mut i = 0usize;

    while i + 16 <= lhs.len() {
        let a = _mm256_loadu_si256(lhs.as_ptr().add(i) as *const __m256i);
        let b = _mm256_loadu_si256(rhs.as_ptr().add(i) as *const __m256i);
        sum = _mm256_add_epi32(sum, _mm256_madd_epi16(a, b));
        i += 16;
    }

    let mut total = horizontal_sum_i32_avx2(sum);
    while i < lhs.len() {
        total += (lhs[i] as i64) * (rhs[i] as i64);
        i += 1;
    }

    total
}

#[inline]
fn clamp_scalar(acc: &[i32], hidden_inputs: &mut [i16; HIDDEN_INPUT_SIZE]) {
    for i in 0..ACCUMULATOR_SIZE {
        hidden_inputs[i] = acc[i].clamp(0, 127) as i16;
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn clamp_avx2(acc: &[i32], hidden_inputs: &mut [i16; HIDDEN_INPUT_SIZE]) {
    let zero = _mm256_setzero_si256();
    let max_val = _mm256_set1_epi32(127);

    for i in (0..ACCUMULATOR_SIZE).step_by(8) {
        let a = _mm256_loadu_si256(acc.as_ptr().add(i) as *const __m256i);
        let clamped = _mm256_max_epi32(_mm256_min_epi32(a, max_val), zero);

        let lo = _mm256_castsi256_si128(clamped);
        let hi = _mm256_extracti128_si256(clamped, 1);
        let packed = _mm_packs_epi32(lo, hi);
        _mm_storeu_si128(hidden_inputs.as_mut_ptr().add(i) as *mut __m128i, packed);
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse2")]
unsafe fn clamp_sse2(acc: &[i32], hidden_inputs: &mut [i16; HIDDEN_INPUT_SIZE]) {
    let zero = _mm_setzero_si128();
    let max_val = _mm_set1_epi32(127);

    for i in (0..ACCUMULATOR_SIZE).step_by(8) {
        let a0 = _mm_loadu_si128(acc.as_ptr().add(i) as *const __m128i);
        let a1 = _mm_loadu_si128(acc.as_ptr().add(i + 4) as *const __m128i);

        let clamped0 = _mm_max_epi32(_mm_min_epi32(a0, max_val), zero);
        let clamped1 = _mm_max_epi32(_mm_min_epi32(a1, max_val), zero);

        let packed = _mm_packs_epi32(clamped0, clamped1);
        _mm_storeu_si128(hidden_inputs.as_mut_ptr().add(i) as *mut __m128i, packed);
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse2")]
unsafe fn horizontal_sum_i32_sse2(v: __m128i) -> i64 {
    let pairwise = _mm_add_epi32(v, _mm_unpackhi_epi64(v, v));
    let pairwise = _mm_add_epi32(pairwise, _mm_shuffle_epi32(pairwise, 0x1B));
    _mm_cvtsi128_si32(pairwise) as i64
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn horizontal_sum_i32_avx2(v: __m256i) -> i64 {
    let lo = _mm256_castsi256_si128(v);
    let hi = _mm256_extracti128_si256(v, 1);
    horizontal_sum_i32_sse2(_mm_add_epi32(lo, hi))
}

#[inline]
fn compute_hidden_layer_scalar(weights: &NnueWeights, hidden_inputs: &[i16; HIDDEN_INPUT_SIZE], hidden: &mut [i16; HIDDEN_SIZE]) {
    const BIAS_SCALE: i64 = 127;
    const BIAS_OFFSET: i64 = 4064;
    const DIVISOR: i64 = 8128;
    
    for neuron in 0..HIDDEN_SIZE {
        let w = weights.hidden_weights_for_neuron(neuron);
        let mut dot = 0i64;

        for i in 0..HIDDEN_INPUT_SIZE {
            dot += (hidden_inputs[i] as i64) * (w[i] as i64);
        }
        
        let bias = (weights.hidden_bias[neuron] as i64) * BIAS_SCALE + BIAS_OFFSET;
        let value = ((dot + bias) / DIVISOR) as i32;
        hidden[neuron] = value.clamp(0, 127) as i16;
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse2")]
unsafe fn compute_hidden_layer_sse2(weights: &NnueWeights, hidden_inputs: &[i16; HIDDEN_INPUT_SIZE], hidden: &mut [i16; HIDDEN_SIZE]) {
    const BIAS_SCALE: i64 = 127;
    const BIAS_OFFSET: i64 = 4064;
    const DIVISOR: i64 = 8128;
    
    for neuron in 0..HIDDEN_SIZE {
        let w = weights.hidden_weights_for_neuron(neuron);
        let mut sum = _mm_setzero_si128();

        for i in (0..HIDDEN_INPUT_SIZE).step_by(8) {
            let a = _mm_loadu_si128(hidden_inputs.as_ptr().add(i) as *const __m128i);
            let b = _mm_loadu_si128(w.as_ptr().add(i) as *const __m128i);
            sum = _mm_add_epi32(sum, _mm_madd_epi16(a, b));
        }
        let dot = horizontal_sum_i32_sse2(sum);
        
        let bias = (weights.hidden_bias[neuron] as i64) * BIAS_SCALE + BIAS_OFFSET;
        let value = ((dot + bias) / DIVISOR) as i32;
        hidden[neuron] = value.clamp(0, 127) as i16;
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn compute_hidden_layer_avx2(weights: &NnueWeights, hidden_inputs: &[i16; HIDDEN_INPUT_SIZE], hidden: &mut [i16; HIDDEN_SIZE]) {
    const BIAS_SCALE: i64 = 127;
    const BIAS_OFFSET: i64 = 4064;
    const DIVISOR: i64 = 8128;

    for neuron in 0..HIDDEN_SIZE {
        let w = weights.hidden_weights_for_neuron(neuron);
        let mut sum = _mm256_setzero_si256();

        for i in (0..HIDDEN_INPUT_SIZE).step_by(16) {
            let a = _mm256_loadu_si256(hidden_inputs.as_ptr().add(i) as *const __m256i);
            let b = _mm256_loadu_si256(w.as_ptr().add(i) as *const __m256i);
            sum = _mm256_add_epi32(sum, _mm256_madd_epi16(a, b));
        }
        let dot = horizontal_sum_i32_avx2(sum);
        
        let bias = (weights.hidden_bias[neuron] as i64) * BIAS_SCALE + BIAS_OFFSET;
        let value = ((dot + bias) / DIVISOR) as i32;
        hidden[neuron] = value.clamp(0, 127) as i16;
    }
}
