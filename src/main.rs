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

pub mod accumulator;
pub mod attacks;
pub mod bench;
pub mod board;
pub mod build_info;
pub mod evaluation;
pub mod features;
pub mod movegen;
pub mod nnue;
pub mod search;
pub mod uci;
pub mod zobrist;

fn main() {
    let mut args = std::env::args().skip(1);
    match args.next().as_deref() {
        Some("bench") => crate::bench::run_bench_cli(args.collect()),
        Some("bench-nnue") => crate::bench::run_eval_bench_from_args(args.collect()),
        Some("bench-search") => crate::bench::run_search_bench_from_args(args.collect()),
        _ => uci::uci_loop(),
    }
}
