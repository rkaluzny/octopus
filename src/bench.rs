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

// Benchmarking utilities for evaluation and search performance.

use crate::board::Board;
use crate::build_info;
use crate::nnue::NnueEvaluator;
use crate::search::{EvalMode, Searcher, DEFAULT_HASH_MB};

use std::sync::{atomic::AtomicBool, Arc};
use std::time::Instant;

pub fn run_bench_cli(args: Vec<String>) {
    let mode = args.get(0).map(|s| s.as_str()).unwrap_or("nnue");
    match mode {
        "nnue" => {
            let iters = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(250_000);
            run_eval_bench(iters);
        }
        "search" => {
            let depth = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(6);
            let time_ms = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(2_000);
            run_search_bench(depth, time_ms);
        }
        "compare" => {
            let depth = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(6);
            let time_ms = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(2_000);
            run_compare_bench(depth, time_ms);
        }
        _ => {
            eprintln!("usage:");
            eprintln!("  bench nnue [iterations]");
            eprintln!("  bench search [depth] [time_ms]");
            eprintln!("  bench compare [depth] [time_ms]");
        }
    }
}

pub fn run_eval_bench_from_args(args: Vec<String>) {
    let iters = args.get(0).and_then(|s| s.parse().ok()).unwrap_or(250_000);
    run_eval_bench(iters);
}

pub fn run_search_bench_from_args(args: Vec<String>) {
    let depth = args.get(0).and_then(|s| s.parse().ok()).unwrap_or(6);
    let time_ms = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(2_000);
    run_search_bench(depth, time_ms);
}

pub fn run_eval_bench(iterations: usize) {
    let board = Board::new();
    let mut nnue = NnueEvaluator::new(crate::nnue::DEFAULT_WEIGHTS_PATH);
    nnue.reset(&board);

    let warmup = 1_000.min(iterations.max(1));
    for _ in 0..warmup {
        let _ = nnue.evaluate(&board);
    }

    let start = Instant::now();
    let mut checksum = 0i64;
    for _ in 0..iterations.max(1) {
        checksum += nnue.evaluate(&board) as i64;
    }
    let elapsed = start.elapsed();
    let per_eval = elapsed.as_secs_f64() * 1e9 / iterations.max(1) as f64;

    println!("bench nnue");
    println!("  build level: {} ({})", build_info::NNUE_LEVEL, build_info::MICROARCH);
    println!("  iterations: {}", iterations.max(1));
    println!("  elapsed_ms: {:.3}", elapsed.as_secs_f64() * 1000.0);
    println!("  ns_per_eval: {:.2}", per_eval);
    println!("  evals_per_sec: {:.2}", 1e9 / per_eval.max(1.0));
    println!("  checksum: {}", checksum);
}

pub fn run_search_bench(depth: u8, time_ms: u64) {
    let mut board = Board::new();
    let stop = Arc::new(AtomicBool::new(false));
    let mut searcher = Searcher::new(stop, DEFAULT_HASH_MB);
    searcher.set_eval_mode(EvalMode::Nnue);
    searcher.reset_search_state(&board);
    searcher.time_limit = std::time::Duration::from_millis(time_ms);

    let start = Instant::now();
    let best_move = searcher.search(&mut board, depth, time_ms);
    let elapsed = start.elapsed();

    println!("bench search");
    println!("  build level: {} ({})", build_info::NNUE_LEVEL, build_info::MICROARCH);
    println!("  depth: {}", depth);
    println!("  time_limit_ms: {}", time_ms);
    println!("  elapsed_ms: {:.3}", elapsed.as_secs_f64() * 1000.0);
    println!("  nodes: {}", searcher.nodes);
    println!("  nps: {:.2}", searcher.nodes as f64 / elapsed.as_secs_f64().max(0.001));
    println!("  seldepth: {}", searcher.seldepth);
    println!("  bestmove: {}", best_move.map(|m| m.to_uci_string()).unwrap_or_else(|| "0000".to_string()));
}

pub fn run_compare_bench(depth: u8, time_ms: u64) {
    let mut hce_board = Board::new();
    let mut nnue_board = Board::new();
    let stop_hce = Arc::new(AtomicBool::new(false));
    let stop_nnue = Arc::new(AtomicBool::new(false));
    let mut hce = Searcher::new(stop_hce, DEFAULT_HASH_MB);
    let mut nnue = Searcher::new(stop_nnue, DEFAULT_HASH_MB);
    hce.set_eval_mode(EvalMode::Hce);
    nnue.set_eval_mode(EvalMode::Nnue);
    hce.reset_search_state(&hce_board);
    nnue.reset_search_state(&nnue_board);
    hce.time_limit = std::time::Duration::from_millis(time_ms);
    nnue.time_limit = std::time::Duration::from_millis(time_ms);

    let start_hce = Instant::now();
    let hce_best = hce.search(&mut hce_board, depth, time_ms);
    let hce_elapsed = start_hce.elapsed();

    let start_nnue = Instant::now();
    let nnue_best = nnue.search(&mut nnue_board, depth, time_ms);
    let nnue_elapsed = start_nnue.elapsed();

    println!("bench compare");
    println!("  build level: {} ({})", build_info::NNUE_LEVEL, build_info::MICROARCH);
    println!("  depth: {}", depth);
    println!("  time_limit_ms: {}", time_ms);
    println!(
        "  hce:  nodes={} nps={:.2} bestmove={}",
        hce.nodes,
        hce.nodes as f64 / hce_elapsed.as_secs_f64().max(0.001),
        hce_best.map(|m| m.to_uci_string()).unwrap_or_else(|| "0000".to_string())
    );
    println!(
        "  nnue: nodes={} nps={:.2} bestmove={}",
        nnue.nodes,
        nnue.nodes as f64 / nnue_elapsed.as_secs_f64().max(0.001),
        nnue_best.map(|m| m.to_uci_string()).unwrap_or_else(|| "0000".to_string())
    );
}
