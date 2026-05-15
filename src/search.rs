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
use crate::board::{Board, PieceType, Undo};
use crate::evaluation;
use crate::movegen::SEE_VALUE;
use crate::movegen::{self, Move};
use crate::nnue::NnueEvaluator;
use std::mem::size_of;
use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc,
};
use std::time::{Duration, Instant};

const INF: i32 = 1_000_000_000;
const NEG_INF: i32 = -INF;
pub const CHECKMATE_SCORE: i32 = 100_000;
pub const DEFAULT_HASH_MB: usize = 64;

// Late Move Reduction thresholds and formulas
const LMR_DEPTH_THRESHOLD: u8 = 4;
const LMR_MOVE_THRESHOLD: usize = 3;

// Late Move Pruning thresholds (prune quiet moves at shallow depths)
const LMP_MOVE_COUNTS: &[usize] = &[0, 0, 3, 8, 16, 24, 32];

// Null Move Pruning parameters - tuned more aggressively
const NMP_MIN_DEPTH: u8 = 3;
const NMP_BASE_REDUCTION: u8 = 2;

// Delta pruning in quiescence
const DELTA_PRUNING_MARGIN: i32 = 120;
// Razoring margin
const RAZOR_MARGIN: i32 = 200;
// Reverse Futility margin (depth 1 and 2)
const REVERSE_FUTILITY_MARGIN: &[i32] = &[0, 120, 240];
// Probcut tuning: reduced-depth pre-search for quiet candidate moves
const PROCUT_MIN_DEPTH: u8 = 10;
const PROCUT_REDUCTION: u8 = 4;
const PROCUT_MOVE_THRESHOLD: usize = 3;
// Singular extension threshold
const SINGULAR_EXTENSION_THRESHOLD: i32 = 75;

// ============================================================================
// Static Exchange Evaluation (SEE)
// ============================================================================
//
// Lightweight evaluation of a capture sequence. Used to prune bad captures
// and for move ordering refinement.

#[derive(Copy, Clone, Debug)]
pub struct TTEntry {
    pub depth: u8,
    pub flag: TTFlag,
    pub score: i32,
    pub best_move: Option<Move>,
    pub age: u8,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum TTFlag {
    Exact,
    LowerBound,
    UpperBound,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum EvalMode {
    Hce,
    Nnue,
    Hybrid,
}

pub struct Searcher {
    pub nodes: u64,
    pub tt: Vec<TTSlot>,
    tt_mask: usize,
    pub nnue: NnueEvaluator,
    eval_mode: EvalMode,
    nnue_active: bool,
    pub nnue_path: String,
    pub start_time: Instant,
    pub time_limit: Duration,
    pub stop: Arc<AtomicBool>,
    pub killer_moves: [[Option<Move>; 2]; 64],
    pub history_moves: [[i32; 64]; 64],
    pub history_piece: [[[i32; 64]; 64]; 6], // [piece][from][to]
    pub counter_move: [[Option<Move>; 64]; 64], // [from][to] of opponent's previous move
    pub counter_move_history: [[i32; 64]; 6], // [piece][to_sq] counter-move history bonus
    pub uci_chess960: bool,
    position_stack: Vec<u64>,
    pub seldepth: u8,
    max_seldepth: u8,
    prev_iteration_score: i32,
    age: u8, // for TT replacement
    history_aging_counter: u8,
}

#[derive(Copy, Clone, Debug)]
pub struct TTSlot {
    pub key: u64,
    pub entry: TTEntry,
}

impl TTSlot {
    fn empty() -> Self {
        Self {
            key: 0,
            entry: TTEntry {
                depth: 0,
                flag: TTFlag::Exact,
                score: 0,
                best_move: None,
                age: 0,
            },
        }
    }
}

impl Searcher {
    #[inline(always)]
    fn aspiration_delta(&self, depth: u8) -> i32 {
        let base = if self.nnue_active { 180 } else { 150 };
        base + (depth as i32 * 8).min(96)
    }

    fn see(&self, board: &Board, mv: Move) -> i32 {
        let from = mv.from as u8;
        let captured = mv.capture;
        if captured.is_none() {
            return 0;
        }
        let captured_val = SEE_VALUE[captured.unwrap() as usize];
        let attacker = board.get_piece_at(from).map(|(p, _)| p);
        if attacker.is_none() {
            return 0;
        }
        let attacker_val = SEE_VALUE[attacker.unwrap() as usize];
        // Simple SEE: if we capture a more valuable piece than our attacker, it's good.
        // Full SEE would simulate recaptures – this simplified version is enough for pruning.
        captured_val - attacker_val / 2
    }
    fn age_history(&mut self) {
        // Gradual history aging: divide by smaller factors or use halving more frequently
        // This allows good moves to decay slower and prevents history overflow
        for piece in 0..6 {
            for from in 0..64 {
                for to in 0..64 {
                    // Saturating integer division by 2 (faster than by 4)
                    self.history_piece[piece][from][to] >>= 1;
                }
            }
        }
        for from in 0..64 {
            for to in 0..64 {
                self.history_moves[from][to] >>= 1;
            }
        }
        // Also age counter-move history
        for piece in 0..6 {
            for to in 0..64 {
                self.counter_move_history[piece][to] >>= 1;
            }
        }
    }
    pub fn new(stop: Arc<AtomicBool>, hash_mb: usize) -> Self {
        let (tt, tt_mask) = Self::make_tt(hash_mb);
        let default_path = crate::nnue::DEFAULT_WEIGHTS_PATH.to_string();
        let nnue = NnueEvaluator::new(default_path.clone());
        Self {
            nodes: 0,
            tt,
            tt_mask,
            nnue,
            eval_mode: EvalMode::Hce,
            nnue_active: false,
            nnue_path: default_path,
            start_time: Instant::now(),
            time_limit: Duration::from_secs(0),
            stop,
            killer_moves: [[None; 2]; 64],
            history_moves: [[0; 64]; 64],
            history_piece: [[[0; 64]; 64]; 6],
            counter_move: [[None; 64]; 64],
            counter_move_history: [[0; 64]; 6],
            uci_chess960: false,
            position_stack: Vec::with_capacity(128),
            seldepth: 0,
            max_seldepth: 0,
            prev_iteration_score: 0,
            age: 1,
            history_aging_counter: 0,
        }
    }

    pub fn set_eval_mode(&mut self, mode: EvalMode) {
        self.eval_mode = mode;
    }

    pub fn set_nnue_path(&mut self, path: String) {
        eprintln!("Setting NNUE path to: {}", path);
        self.nnue_path = path.clone();
        self.nnue.set_weights_path(path);
        self.nnue.reload();
        eprintln!("NNUE weights loaded: {}", self.nnue.weights_loaded);
        // If weights loaded successfully and we're in NNUE/Hybrid mode, mark as ready
        if self.nnue.weights_loaded && matches!(self.eval_mode, EvalMode::Nnue | EvalMode::Hybrid) {
            // Will be properly initialized in reset_search_state
        }
    }

    pub fn set_uci_chess960(&mut self, enabled: bool) {
        self.uci_chess960 = enabled;
    }

    fn make_tt(hash_mb: usize) -> (Vec<TTSlot>, usize) {
        let hash_mb = hash_mb.max(1);
        let bytes = hash_mb
            .saturating_mul(1024)
            .saturating_mul(1024)
            .max(size_of::<TTSlot>());
        let mut entries = bytes / size_of::<TTSlot>();
        if entries == 0 {
            entries = 1;
        }
        entries = entries.next_power_of_two();
        let mask = entries - 1;
        (vec![TTSlot::empty(); entries], mask)
    }

    fn evaluate_board(&self, board: &Board) -> i32 {
        match self.eval_mode {
            EvalMode::Hce => evaluation::evaluate(board),
            EvalMode::Nnue => {
                if self.nnue_active {
                    self.nnue.evaluate(board)
                } else {
                    evaluation::evaluate(board)
                }
            }
            EvalMode::Hybrid => {
                if self.nnue_active {
                    let hce = evaluation::evaluate(board);
                    let nnue = self.nnue.evaluate(board);
                    (nnue * 7 + hce * 3) / 10
                } else {
                    evaluation::evaluate(board)
                }
            }
        }
    }

    fn make_move_with_nnue(&mut self, board: &mut Board, mv: &Move) -> Undo {
        if self.nnue_active {
            self.nnue.apply_move(board, mv);
        }
        board.make_move(mv)
    }

    fn unmake_move_with_nnue(&mut self, board: &mut Board, mv: &Move, undo: Undo) {
        board.unmake_move(undo);
        if self.nnue_active {
            self.nnue.unapply_move(board, mv);
        }
    }

    pub fn search(&mut self, board: &mut Board, max_depth: u8, time_limit_ms: u64) -> Option<Move> {
        self.age = self.age.wrapping_add(1);
        self.start_time = Instant::now();
        self.time_limit = Duration::from_millis(time_limit_ms.max(1));
        self.nodes = 0;
        self.max_seldepth = 0;
        // clear transposition table quickly by resetting keys
        for slot in self.tt.iter_mut() {
            *slot = TTSlot::empty();
        }
        self.killer_moves = [[None; 2]; 64];
        self.history_moves = [[0; 64]; 64];
        self.reset_search_state(board);
        self.prev_iteration_score = 0;

        let mut best_move = None;
        let mut best_score = 0;
        let mut pv_line = Vec::new();

        for depth in 1..=max_depth {
            self.seldepth = 0;
            self.max_seldepth = 0;
            let mut local_pv = Vec::new();

            let mut window_delta = if depth == 1 {
                INF
            } else {
                self.aspiration_delta(depth)
            };
            let mut local_alpha = if depth == 1 {
                NEG_INF
            } else {
                (self.prev_iteration_score - window_delta).max(NEG_INF + 1)
            };
            let mut local_beta = if depth == 1 {
                INF
            } else {
                (self.prev_iteration_score + window_delta).min(INF - 1)
            };

            let score = loop {
                local_pv.clear();
                let score = self.negamax_root(board, depth, local_alpha, local_beta, &mut local_pv);
                if self.should_stop() {
                    break score;
                }

                if depth == 1 || (score > local_alpha && score < local_beta) {
                    break score;
                }

                window_delta = (window_delta.saturating_mul(2)).min(10_000);
                if score <= local_alpha {
                    local_alpha = (local_alpha - window_delta).max(NEG_INF + 1);
                } else {
                    local_beta = (local_beta + window_delta).min(INF - 1);
                }
            };

            if self.should_stop() {
                break;
            }
            if let Some(first) = local_pv.first().copied() {
                best_move = Some(first);
                best_score = score;
                pv_line = local_pv;
                self.prev_iteration_score = score;
            }
            self.print_info(board, depth, best_score, &pv_line);
        }
        self.history_aging_counter += 1;
        if self.history_aging_counter >= 8 {
            self.age_history();
            self.history_aging_counter = 0;
        }

        // Validate best_move is legal
        if let Some(bm) = best_move {
            let legal_moves = crate::movegen::generate_moves(board);
            if !legal_moves.contains(&bm) {
                eprintln!("ILLEGAL MOVE DETECTED: {:?}", bm);
                eprintln!("Board:\n{}", board.to_string());
                eprintln!("Legal moves: {:?}", legal_moves);
                // Return first legal move instead
                best_move = legal_moves.first().copied();
            }
        }

        best_move
    }

    pub fn negamax_root(
        &mut self,
        board: &mut Board,
        depth: u8,
        alpha: i32,
        beta: i32,
        pv_line: &mut Vec<Move>,
    ) -> i32 {
        let mut moves = movegen::generate_moves(board);
        if moves.is_empty() {
            return if board.is_in_check() {
                -CHECKMATE_SCORE
            } else {
                0
            };
        }

        let tt_move = self.tt_get(board.hash).and_then(|entry| entry.best_move);
        self.order_moves_inline(board, 0, &mut moves, tt_move);

        let mut local_alpha = alpha;
        let mut best_score = NEG_INF;
        let mut best_move = moves[0];
        let mut best_child_pv = Vec::new();

        // Debug: validate TT move
        if let Some(tt_mv) = tt_move {
            if !moves.contains(&tt_mv) {
                eprintln!("WARNING: TT move {:?} not in legal moves!", tt_mv);
                eprintln!("Board state:\n{}", board.to_string());
                eprintln!("TT move from hash: {:?}", board.hash);
            }
        }

        for mv in moves {
            let undo = self.make_move_with_nnue(board, &mv);
            self.position_stack.push(board.hash);
            let mut child_pv = Vec::new();

            let score = -self.negamax(
                board,
                depth - 1,
                -beta,
                -local_alpha,
                1,
                &mut child_pv,
                None,
            );

            self.position_stack.pop();
            if self.should_stop() {
                self.unmake_move_with_nnue(board, &mv, undo);
                break;
            }

            if score > best_score {
                best_score = score;
                best_move = mv;
                best_child_pv = child_pv;
            }

            local_alpha = local_alpha.max(score);
            if local_alpha >= beta {
                self.unmake_move_with_nnue(board, &mv, undo);
                break;
            }
            self.unmake_move_with_nnue(board, &mv, undo);
        }

        pv_line.clear();
        pv_line.push(best_move);
        pv_line.extend(best_child_pv);
        self.tt_put(
            board.hash,
            TTEntry {
                depth,
                flag: TTFlag::Exact,
                score: best_score,
                best_move: Some(best_move),
                age: self.age,
            },
        );
        best_score
    }

    fn negamax(
        &mut self,
        board: &mut Board,
        depth: u8,
        mut alpha: i32,
        beta: i32,
        ply: usize,
        pv_line: &mut Vec<Move>,
        prev_move: Option<Move>,
    ) -> i32 {
        let is_pv = beta - alpha > 1;
        self.update_seldepth(ply);

        if self.should_stop() {
            return 0;
        }
        self.nodes += 1;
        if self.nodes & 2047 == 0 && self.start_time.elapsed() >= self.time_limit {
            self.stop.store(true, Ordering::Relaxed);
            return 0;
        }

        if ply >= 63 {
            return self.evaluate_board(board);
        }

        if self.is_repetition(board.hash) {
            return 0;
        }

        // TT Lookup (more aggressive before move generation)
        if let Some(entry) = self.tt_get(board.hash) {
            if entry.depth >= depth {
                match entry.flag {
                    TTFlag::Exact => return entry.score,
                    TTFlag::LowerBound => alpha = alpha.max(entry.score),
                    TTFlag::UpperBound => {}
                }
                if alpha >= beta {
                    return entry.score;
                }
            }
        }

        if depth == 0 {
            return self.quiescence(board, alpha, beta, ply);
        }

        let in_check = board.is_in_check();
        let tt_move = self.tt_get(board.hash).and_then(|entry| entry.best_move);

        let mut singular = false;
        if depth >= 4 && !in_check && is_pv {
            if let Some(entry) = self.tt_get(board.hash) {
                if entry.flag == TTFlag::Exact
                    && entry.score > alpha + SINGULAR_EXTENSION_THRESHOLD
                    && entry.score < beta
                    && entry.best_move == Some(tt_move.unwrap())
                {
                    singular = true;
                }
            }
        }
        let static_eval = self.evaluate_board(board);

        // ======================================================================
        // Pruning Techniques
        // ======================================================================

        // Futility pruning (depth <= 3, quiet moves, not in check)
        if depth <= 3 && !in_check && !is_pv {
            let margin = 100 * depth as i32 + 50;
            if static_eval + margin < alpha {
                return static_eval + margin;
            }
        }

        if depth <= 2 && !in_check && !is_pv && static_eval + RAZOR_MARGIN < alpha {
            let qscore = self.quiescence(board, alpha, beta, ply);
            if qscore <= alpha {
                return qscore;
            }
        }

        // Null Move Pruning (depth >= 3, not in check, not PV, sufficient material)
        if depth >= NMP_MIN_DEPTH
            && !in_check
            && !is_pv
            && alpha < CHECKMATE_SCORE - 1000
            && beta > -CHECKMATE_SCORE + 1000
            && static_eval >= beta
        {
            // Check for sufficient non‑pawn material
            let has_major_pieces = {
                let our_pieces = board.color_bitboards[board.side_to_move as usize];
                let pawns = board.bitboards[PieceType::Pawn as usize];
                let kings = board.bitboards[PieceType::King as usize];
                (our_pieces & !(pawns | kings)) != 0
            };
            if has_major_pieces {
                let reduction = NMP_BASE_REDUCTION + (depth / 3).max(1);
                let reduced_depth = depth.saturating_sub(reduction).max(1);
                let undo = board.make_move_null();
                self.position_stack.push(board.hash);
                let nmp_score = -self.negamax(
                    board,
                    reduced_depth,
                    -beta,
                    -beta + 1,
                    ply + 1,
                    pv_line,
                    None,
                );
                self.position_stack.pop();
                board.unmake_move_null(undo);
                if nmp_score >= beta {
                    return beta;
                }
            }
        }

        // Reverse Futility Pruning (depth 1-2)
        if depth <= 2
            && !in_check
            && !is_pv
            && static_eval - REVERSE_FUTILITY_MARGIN[depth as usize] >= beta
        {
            return static_eval;
        }
        // ======================================================================
        // Move Generation and Ordering
        // ======================================================================

        let mut moves = movegen::generate_moves(board);
        if moves.is_empty() {
            return if in_check {
                -CHECKMATE_SCORE + ply as i32
            } else {
                0
            };
        }

        self.order_moves_inline(board, ply, &mut moves, tt_move);

        let original_alpha = alpha;
        let mut best_score = NEG_INF;
        let mut best_move = None;
        let mut best_child_pv = Vec::new();
        let mut move_count = 0;
        let mut first_move = true;

        // ======================================================================
        // Main Search Loop with PVS (first move full search, others zero-window)
        // ======================================================================

        for mv in moves {
            // Late Move Pruning (LMP): hard prune quiet moves at shallow depths
            // Always search at least one move at a node before pruning quiet moves.
            if !is_pv && !in_check && depth <= 6 && !first_move {
                let lmp_threshold = LMP_MOVE_COUNTS.get(depth as usize).copied().unwrap_or(16);
                if move_count >= lmp_threshold && mv.capture.is_none() {
                    break; // Skip remaining quiet moves
                }
            }

            let undo = self.make_move_with_nnue(board, &mv);
            self.position_stack.push(board.hash);
            let mut child_pv = Vec::new();

            // Probcut: shallow reduction search to quickly cutoff strong quiet moves.
            // This is only safe for deeper non-capture moves in non-PV nodes.
            if !first_move
                && depth >= PROCUT_MIN_DEPTH
                && !in_check
                && !is_pv
                && mv.capture.is_none()
                && mv.promotion.is_none()
                && move_count >= PROCUT_MOVE_THRESHOLD
            {
                let reduced_depth = depth.saturating_sub(PROCUT_REDUCTION).max(1);
                let mut probe_pv = Vec::new();
                let probe_score = -self.negamax(
                    board,
                    reduced_depth,
                    -beta,
                    -beta + 1,
                    ply + 1,
                    &mut probe_pv,
                    Some(mv),
                );
                if probe_score >= beta {
                    self.position_stack.pop();
                    self.unmake_move_with_nnue(board, &mv, undo);
                    return beta;
                }
            }

            // ================================================================
            // PVS only: first move full search, others zero-window
            // ================================================================
            let ext_depth = if singular && Some(mv) == tt_move {
                depth
            } else {
                depth - 1
            };
            let score = if first_move {
                -self.negamax(
                    board,
                    ext_depth,
                    -beta,
                    -alpha,
                    ply + 1,
                    &mut child_pv,
                    Some(mv),
                )
            } else {
                let use_lmr = depth >= LMR_DEPTH_THRESHOLD
                    && move_count >= LMR_MOVE_THRESHOLD
                    && mv.capture.is_none()
                    && mv.promotion.is_none()
                    && !in_check
                    && !is_pv;
                let reduction = if use_lmr {
                    // Increased aggressiveness: base formula + 1 reduction for moves after 6th
                    let base_reduction =
                        ((depth as u16 / 3) + (move_count as u16 / 6)).min(3) as u8;
                    let additional = if move_count > 6 { 1 } else { 0 };
                    (base_reduction + additional).min(depth - 1)
                } else {
                    0
                };
                let reduced_depth = depth - 1 - reduction;

                let mut score = -self.negamax(
                    board,
                    reduced_depth,
                    -alpha - 1,
                    -alpha,
                    ply + 1,
                    &mut child_pv,
                    Some(mv),
                );
                if score > alpha && score < beta {
                    // re‑search with full depth
                    child_pv.clear();
                    score = -self.negamax(
                        board,
                        depth - 1,
                        -beta,
                        -alpha,
                        ply + 1,
                        &mut child_pv,
                        Some(mv),
                    );
                }
                score
            };

            first_move = false;

            if self.should_stop() {
                self.position_stack.pop();
                self.unmake_move_with_nnue(board, &mv, undo);
                return 0;
            }

            if score > best_score {
                best_score = score;
                best_move = Some(mv);
                best_child_pv = child_pv;
            }

            alpha = alpha.max(score);
            if alpha >= beta {
                // Beta cutoff: update killers and history
                if mv.capture.is_none() {
                    self.store_killer(ply, mv);
                    self.history_moves[mv.from as usize][mv.to as usize] +=
                        depth as i32 * depth as i32;
                    // Update counter-move history for the beta cutoff move
                    if let Some((piece, _)) = board.get_piece_at(mv.from as u8) {
                        self.counter_move_history[piece as usize][mv.to as usize] +=
                            depth as i32 * depth as i32;
                    }
                }
                self.position_stack.pop();
                self.unmake_move_with_nnue(board, &mv, undo);
                if let Some(prev) = prev_move {
                    self.counter_move[prev.from as usize][prev.to as usize] = Some(mv);
                }
                break;
            }
            self.position_stack.pop();
            self.unmake_move_with_nnue(board, &mv, undo);
            move_count += 1;
        }

        if best_move.is_none() {
            return static_eval.max(alpha);
        }

        pv_line.clear();
        if let Some(mv) = best_move {
            pv_line.push(mv);
            pv_line.extend(best_child_pv);
        }

        let flag = if best_score <= original_alpha {
            TTFlag::UpperBound
        } else if best_score >= beta {
            TTFlag::LowerBound
        } else {
            TTFlag::Exact
        };
        self.tt_put(
            board.hash,
            TTEntry {
                depth,
                flag,
                score: best_score,
                best_move,
                age: self.age,
            },
        );

        best_score
    }

    pub fn reset_search_state(&mut self, board: &Board) {
        self.nodes = 0;
        self.position_stack.clear();
        self.position_stack.push(board.hash);
        self.stop.store(false, Ordering::Relaxed);
        self.start_time = Instant::now();
        self.seldepth = 0;
        self.max_seldepth = 0;
        self.nnue_active =
            self.nnue.weights_loaded && matches!(self.eval_mode, EvalMode::Nnue | EvalMode::Hybrid);
        if self.nnue_active {
            self.nnue.reset(board);
        } else {
            self.nnue.disable();
        }
    }

    fn quiescence(&mut self, board: &mut Board, mut alpha: i32, beta: i32, ply: usize) -> i32 {
        self.update_seldepth(ply);
        if self.should_stop() {
            return 0;
        }
        self.nodes += 1;
        if self.nodes & 2047 == 0 && self.start_time.elapsed() >= self.time_limit {
            self.stop.store(true, Ordering::Relaxed);
            return 0;
        }
        if self.is_repetition(board.hash) {
            return 0;
        }

        let in_check = board.is_in_check();
        let stand_pat = self.evaluate_board(board);

        if !in_check {
            if stand_pat >= beta {
                return beta;
            }
            // Delta pruning with dynamic margin
            let delta_margin = DELTA_PRUNING_MARGIN
                + (stand_pat.abs() / 10)
                + if self.nnue_active { 40 } else { 0 };
            if stand_pat + delta_margin < alpha {
                return stand_pat + delta_margin;
            }
            alpha = alpha.max(stand_pat);
        }

        let mut moves = if in_check {
            movegen::generate_moves(board)
        } else {
            // In QS, normally only generate captures
            // But in first ply of QS, also generate quiet checks to reduce horizon effect
            let mut all_moves = movegen::generate_all_captures(board);

            // Add quiet checks if this is early in QS (ply indicates depth in QS)
            let quiet_check_depth = if self.nnue_active { 2 } else { 3 };
            if ply < quiet_check_depth {
                let enemy_king_sq = {
                    let enemy_color = board.side_to_move.opponent();
                    let king_bb = board.bitboards[PieceType::King as usize]
                        & board.color_bitboards[enemy_color as usize];
                    if king_bb == 0 {
                        0
                    } else {
                        king_bb.trailing_zeros() as u8
                    }
                };
                let all_possible = movegen::generate_moves(board);
                for mv in all_possible {
                    if mv.capture.is_none()
                        && mv.promotion.is_none()
                        && mv.move_type == movegen::MoveType::Normal
                    {
                        // Fast check test without cloning board
                        // Get piece from board using from square
                        let from_sq = mv.from as u8;
                        let from_bb = 1u64 << from_sq;
                        let piece = if (board.bitboards[PieceType::Pawn as usize] & from_bb) != 0 {
                            PieceType::Pawn
                        } else if (board.bitboards[PieceType::Knight as usize] & from_bb) != 0 {
                            PieceType::Knight
                        } else if (board.bitboards[PieceType::Bishop as usize] & from_bb) != 0 {
                            PieceType::Bishop
                        } else if (board.bitboards[PieceType::Rook as usize] & from_bb) != 0 {
                            PieceType::Rook
                        } else if (board.bitboards[PieceType::Queen as usize] & from_bb) != 0 {
                            PieceType::Queen
                        } else {
                            PieceType::King
                        };
                        let to_sq = mv.to as u8;
                        let gives_check = match piece {
                            PieceType::Pawn => {
                                (attacks::get_pawn_attacks(to_sq, board.side_to_move)
                                    & (1u64 << enemy_king_sq))
                                    != 0
                            }
                            PieceType::Knight => {
                                (attacks::get_knight_attacks(to_sq) & (1u64 << enemy_king_sq)) != 0
                            }
                            PieceType::Bishop => {
                                (attacks::get_bishop_attacks(
                                    to_sq,
                                    board.color_bitboards[0] | board.color_bitboards[1],
                                ) & (1u64 << enemy_king_sq))
                                    != 0
                            }
                            PieceType::Rook => {
                                (attacks::get_rook_attacks(
                                    to_sq,
                                    board.color_bitboards[0] | board.color_bitboards[1],
                                ) & (1u64 << enemy_king_sq))
                                    != 0
                            }
                            PieceType::Queen => {
                                ((attacks::get_bishop_attacks(
                                    to_sq,
                                    board.color_bitboards[0] | board.color_bitboards[1],
                                ) | attacks::get_rook_attacks(
                                    to_sq,
                                    board.color_bitboards[0] | board.color_bitboards[1],
                                )) & (1u64 << enemy_king_sq))
                                    != 0
                            }
                            _ => false,
                        };
                        if gives_check {
                            all_moves.push(mv);
                        }
                    }
                }
            }

            all_moves
        };
        if moves.is_empty() {
            return if in_check {
                -CHECKMATE_SCORE + ply as i32
            } else {
                alpha
            };
        }

        // Order captures with MVV‑LVA + SEE
        let tt_move = self.tt_get(board.hash).and_then(|entry| entry.best_move);
        self.order_moves_inline(board, ply, &mut moves, tt_move);

        for mv in moves {
            if !in_check && mv.capture.is_none() && mv.promotion.is_none() {
                continue;
            }
            // SEE pruning: skip bad captures (losing material)
            if !in_check && mv.capture.is_some() {
                if self.see(board, mv) < -50 {
                    continue;
                }
            }

            let undo = self.make_move_with_nnue(board, &mv);
            self.position_stack.push(board.hash);
            let score = -self.quiescence(board, -beta, -alpha, ply + 1);
            self.position_stack.pop();
            if self.should_stop() {
                self.unmake_move_with_nnue(board, &mv, undo);
                return 0;
            }
            if score >= beta {
                self.unmake_move_with_nnue(board, &mv, undo);
                return beta;
            }
            alpha = alpha.max(score);
            self.unmake_move_with_nnue(board, &mv, undo);
        }
        alpha
    }

    fn is_repetition(&self, hash: u64) -> bool {
        let stack = &self.position_stack;
        let len = stack.len();
        if len <= 1 {
            return false;
        }
        let mut count = 0;
        for &h in &stack[..len - 1] {
            if h == hash {
                count += 1;
                if count >= 2 {
                    return true;
                }
            }
        }
        false
    }

    fn order_moves_inline(
        &self,
        board: &Board,
        ply: usize,
        moves: &mut [Move],
        tt_move: Option<Move>,
    ) {
        if moves.len() <= 1 {
            return;
        }

        let mut scored_moves: Vec<(i32, Move)> = moves
            .iter()
            .copied()
            .map(|mv| (self.score_move(board, ply, mv, tt_move), mv))
            .collect();

        scored_moves.sort_unstable_by(|a, b| b.0.cmp(&a.0));

        for (i, (_, mv)) in scored_moves.into_iter().enumerate() {
            moves[i] = mv;
        }
    }

    #[inline]
    fn score_move(&self, board: &Board, ply: usize, mv: Move, tt_move: Option<Move>) -> i32 {
        if Some(mv) == tt_move {
            return 2_000_000;
        }
        // Captures: MVV‑LVA
        if let Some(victim) = mv.capture {
            let attacker_score = if let Some((attacker, _)) = board.get_piece_at(mv.from as u8) {
                self.mvv_lva_score(victim, attacker)
            } else {
                0
            };
            return 1_000_000 + attacker_score;
        }
        // Quiet moves: killer + history + counter + counter-move history
        let mut score = 0;
        if Some(mv) == self.killer_moves[ply][0] {
            score += 900_000;
        } else if Some(mv) == self.killer_moves[ply][1] {
            score += 800_000;
        }
        // Piece‑specific history
        if let Some((piece, _)) = board.get_piece_at(mv.from as u8) {
            score += self.history_piece[piece as usize][mv.from as usize][mv.to as usize];
            // Counter-move history bonus: based on piece type and destination square
            score += self.counter_move_history[piece as usize][mv.to as usize] / 2;
        }
        // Counter move heuristic
        if let Some(counter) = self.counter_move[mv.from as usize][mv.to as usize] {
            if counter == mv {
                score += 500_000;
            }
        }
        score
    }

    fn mvv_lva_score(&self, victim: PieceType, attacker: PieceType) -> i32 {
        let victim_value = piece_order(victim);
        let attacker_value = piece_order(attacker);
        victim_value * 10 - attacker_value
    }

    fn store_killer(&mut self, ply: usize, mv: Move) {
        if self.killer_moves[ply][0] != Some(mv) {
            self.killer_moves[ply][1] = self.killer_moves[ply][0];
            self.killer_moves[ply][0] = Some(mv);
        }
    }

    fn should_stop(&self) -> bool {
        self.stop.load(Ordering::Relaxed)
    }

    #[inline(always)]
    fn update_seldepth(&mut self, ply: usize) {
        let ply = ply as u8;
        if ply > self.seldepth {
            self.seldepth = ply;
        }
        if ply > self.max_seldepth {
            self.max_seldepth = ply;
        }
    }

    fn tt_get(&self, key: u64) -> Option<TTEntry> {
        let idx = (key as usize) & self.tt_mask;
        let slot = &self.tt[idx];
        // Only check key match. Depth 0 (quiescence) entries are valid!
        if slot.key == key {
            Some(slot.entry)
        } else {
            None
        }
    }

    fn tt_put(&mut self, key: u64, entry: TTEntry) {
        let idx = (key as usize) & self.tt_mask;
        let slot = &mut self.tt[idx];

        // If same key, always replace (depth may be higher or exact flag better)
        if slot.key == key {
            slot.entry = entry;
        } else {
            // Different key: use depth-preferred replacement for PV nodes,
            // but always replace if new entry is deeper or old entry is from a previous search age
            let should_replace = entry.depth >= slot.entry.depth ||  // New entry is deeper
                slot.entry.age < self.age; // Old entry is from previous search iteration(s)

            if should_replace {
                slot.key = key;
                slot.entry = entry;
            }
        }
    }

    fn print_info(&self, board: &Board, depth: u8, score: i32, pv_line: &[Move]) {
        let elapsed_ms = self.start_time.elapsed().as_millis().max(1);
        let nps = (self.nodes as u128 * 1000 / elapsed_ms) as u64;
        let score_string = format_score(score);
        let pv_string = pv_line
            .iter()
            .map(|mv| mv.to_uci_string_for_board(board, self.uci_chess960))
            .collect::<Vec<_>>()
            .join(" ");
        println!(
            "info depth {} seldepth {} score {} raw {} nodes {} nps {} time {} pv {}",
            depth, self.seldepth, score_string, score, self.nodes, nps, elapsed_ms, pv_string
        );
    }
}

fn piece_order(piece: PieceType) -> i32 {
    match piece {
        PieceType::Pawn => 100,
        PieceType::Knight => 300,
        PieceType::Bishop => 325,
        PieceType::Rook => 500,
        PieceType::Queen => 900,
        PieceType::King => 10_000,
    }
}

fn format_score(score: i32) -> String {
    if score.abs() >= CHECKMATE_SCORE - 1000 {
        let mate_in = if score > 0 {
            (CHECKMATE_SCORE - score + 1) / 2
        } else {
            -((CHECKMATE_SCORE + score + 1) / 2)
        };
        format!("mate {}", mate_in)
    } else {
        format!("cp {}", score)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::board::Board;

    #[test]
    fn quiescence_searches_evasions_when_in_check() {
        let mut board =
            Board::from_fen("3qk3/8/8/8/8/8/4R3/4K3 b - - 0 1").expect("valid in-check FEN");
        assert!(board.is_in_check());

        let stop = Arc::new(AtomicBool::new(false));
        let mut searcher = Searcher::new(stop, DEFAULT_HASH_MB);
        searcher.position_stack.push(board.hash);

        let _ = searcher.quiescence(&mut board, NEG_INF, NEG_INF, 0);
        assert!(searcher.nodes > 1);
    }

    #[test]
    fn make_and_unmake_restore_start_position() {
        let mut board = Board::new();
        let original_fen = board.to_fen();
        let original_hash = board.hash;

        let mv = movegen::find_legal_move(&board, "e2e4").expect("e2e4 should be legal");
        let undo = board.make_move(&mv);
        board.unmake_move(undo);

        assert_eq!(board.to_fen(), original_fen);
        assert_eq!(board.hash, original_hash);
    }

    #[test]
    fn negamax_root_returns_finite_score_on_quiet_positions() {
        let mut board = Board::new();
        let stop = Arc::new(AtomicBool::new(false));
        let mut searcher = Searcher::new(stop, DEFAULT_HASH_MB);
        searcher.start_time = Instant::now();
        searcher.time_limit = Duration::from_secs(1);

        let mut pv_line = Vec::new();
        let score = searcher.negamax_root(&mut board, 2, NEG_INF, INF, &mut pv_line);

        assert!(score > NEG_INF / 2, "expected a finite score, got {score}");
        assert!(score < INF / 2, "expected a finite score, got {score}");
    }

    #[test]
    fn repetition_scan_reaches_beyond_twelve_positions() {
        let stop = Arc::new(AtomicBool::new(false));
        let searcher = Searcher::new(stop, DEFAULT_HASH_MB);

        let repeated_hash = 0xCAFE_BABE_u64;
        let mut stack = Vec::new();
        stack.push(repeated_hash);
        for value in 1..=12u64 {
            stack.push(value);
        }
        stack.push(repeated_hash);
        stack.push(repeated_hash);

        let mut searcher = searcher;
        searcher.position_stack = stack;

        assert!(searcher.is_repetition(repeated_hash));
    }
}
