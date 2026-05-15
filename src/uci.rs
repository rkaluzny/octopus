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

use crate::board::{Board, Color, STARTING_FEN};
use crate::build_info;
use crate::movegen;
use crate::search::{EvalMode, Searcher, DEFAULT_HASH_MB};

use std::io::{self, BufRead};
use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc, Mutex,
};
use std::thread::{self, JoinHandle};

struct ActiveSearch {
    stop: Arc<AtomicBool>,
    handle: JoinHandle<()>,
}

const MIN_HASH_MB: usize = 1;
const MAX_HASH_MB: usize = 4096;

pub fn uci_loop() {
    let mut board = Board::new();
    let stdin = io::stdin();
    let mut active_search: Option<ActiveSearch> = None;
    let mut hash_mb = DEFAULT_HASH_MB;
    let mut uci_chess960 = false;
    let searcher = Arc::new(Mutex::new(Searcher::new(
        Arc::new(AtomicBool::new(false)),
        hash_mb,
    )));

    for line in stdin.lock().lines() {
        let line = match line {
            Ok(line) => line,
            Err(_) => break,
        };
        let mut parts = line.split_whitespace();
        let command = parts.next().unwrap_or("");

        match command {
            "uci" => {
                println!(
                    "id name Octopus v0.1 {} ({})",
                    build_info::NNUE_LEVEL,
                    build_info::MICROARCH
                );
                println!("id author Robin Kaluzny");
                println!(
                    "option name Hash type spin default {} min {} max {}",
                    DEFAULT_HASH_MB, MIN_HASH_MB, MAX_HASH_MB
                );
                println!("option name UCI_Chess960 type check default false");
                println!("option name EvalMode type combo default HCE var HCE var NNUE var Hybrid");
                println!("option name NnuePath type string default output/nnue_weights.bin");
                println!("uciok");
            }
            "isready" => {
                println!("readyok");
            }
            "ucinewgame" => {
                stop_active_search(&mut active_search);
                board = Board::new();
            }
            "position" => {
                stop_active_search(&mut active_search);
                handle_position_command(&mut board, &mut parts);
            }
            "go" => {
                stop_active_search(&mut active_search);
                let searcher_clone = Arc::clone(&searcher);
                active_search = Some(start_search(
                    &board,
                    board.side_to_move,
                    searcher_clone,
                    &mut parts,
                ));
            }
            "setoption" => {
                handle_setoption_command(
                    &mut parts,
                    &mut hash_mb,
                    &mut uci_chess960,
                    &mut active_search,
                    &searcher,
                );
            }
            "stop" => {
                stop_active_search(&mut active_search);
            }
            "quit" => {
                stop_active_search(&mut active_search);
                break;
            }
            "d" => {
                println!("{}", board);
            }
            _ => {}
        }
    }
}

fn stop_active_search(active_search: &mut Option<ActiveSearch>) {
    if let Some(search) = active_search.take() {
        search.stop.store(true, Ordering::Relaxed);
        let _ = search.handle.join();
    }
}

fn handle_position_command(board: &mut Board, parts: &mut std::str::SplitWhitespace) {
    let first_part = parts.next().unwrap_or("");
    let mut moves_to_apply: Vec<String> = Vec::new();

    match first_part {
        "startpos" => {
            *board = Board::from_fen(STARTING_FEN).unwrap();
            if let Some("moves") = parts.next() {
                moves_to_apply = parts.map(|mv| mv.to_string()).collect();
            }
        }
        "fen" => {
            let fen_parts: Vec<&str> = parts.collect();
            let moves_index = fen_parts.iter().position(|&token| token == "moves");
            let fen_slice = if let Some(index) = moves_index {
                fen_parts.split_at(index).0
            } else {
                fen_parts.as_slice()
            };

            let fen = fen_slice.join(" ");
            if let Ok(parsed_board) = Board::from_fen(&fen) {
                *board = parsed_board;
            }
            if let Some(index) = moves_index {
                moves_to_apply = fen_parts[index + 1..]
                    .iter()
                    .map(|mv| mv.to_string())
                    .collect();
            }
        }
        _ => {}
    }

    for move_str in &moves_to_apply {
        if let Some(mv) = movegen::find_legal_move(board, move_str) {
            board.apply_move(&mv);
        } else {
            eprintln!("ERROR: Could not find move {}!", move_str);
            break;
        }
    }
}

fn start_search(
    board: &Board,
    side_to_move: Color,
    searcher: Arc<Mutex<Searcher>>,
    parts: &mut std::str::SplitWhitespace,
) -> ActiveSearch {
    let params = parse_go_params(parts, side_to_move);
    let stop = Arc::new(AtomicBool::new(false));
    let _search_stop = stop.clone();

    let mut board = board.clone();
    let handle = thread::spawn(move || {
        let mut searcher = searcher.lock().unwrap();
        searcher.reset_search_state(&board);
        searcher.time_limit = std::time::Duration::from_millis(params.time_limit_ms);
        let best_move = searcher.search(&mut board, params.depth, params.time_limit_ms);

        match best_move {
            Some(mv) => println!(
                "bestmove {}",
                mv.to_uci_string_for_board(&board, searcher.uci_chess960)
            ),
            None => println!("bestmove 0000"),
        }
    });

    ActiveSearch { stop, handle }
}

fn handle_setoption_command(
    parts: &mut std::str::SplitWhitespace,
    hash_mb: &mut usize,
    uci_chess960: &mut bool,
    active_search: &mut Option<ActiveSearch>,
    searcher: &Arc<Mutex<Searcher>>,
) {
    let tokens: Vec<&str> = parts.collect();
    if tokens.is_empty() {
        return;
    }

    let name_pos = tokens
        .iter()
        .position(|&token| token.eq_ignore_ascii_case("name"));
    let Some(name_pos) = name_pos else {
        return;
    };

    let value_pos = tokens
        .iter()
        .position(|&token| token.eq_ignore_ascii_case("value"));
    let name_tokens = match value_pos {
        Some(pos) if pos > name_pos + 1 => &tokens[name_pos + 1..pos],
        Some(_) => &tokens[name_pos + 1..name_pos + 1],
        None => &tokens[name_pos + 1..],
    };
    let option_name = name_tokens.join(" ").to_lowercase();

    if option_name == "hash" {
        if let Some(pos) = value_pos {
            if let Some(value_str) = tokens.get(pos + 1) {
                if let Ok(value) = value_str.parse::<usize>() {
                    *hash_mb = value.clamp(MIN_HASH_MB, MAX_HASH_MB);
                    stop_active_search(active_search);
                }
            }
        }
    } else if option_name == "uci_chess960" || option_name == "uci chess960" {
        if let Some(pos) = value_pos {
            if let Some(value_str) = tokens.get(pos + 1) {
                let value = value_str.eq_ignore_ascii_case("true") || *value_str == "1";
                *uci_chess960 = value;
                if let Ok(mut searcher) = searcher.lock() {
                    searcher.set_uci_chess960(value);
                }
            }
        }
    } else if option_name == "evalmode" || option_name == "eval mode" {
        if let Some(pos) = value_pos {
            if let Some(value_str) = tokens.get(pos + 1) {
                let mode = if value_str.eq_ignore_ascii_case("hce") {
                    Some(EvalMode::Hce)
                } else if value_str.eq_ignore_ascii_case("nnue") {
                    Some(EvalMode::Nnue)
                } else if value_str.eq_ignore_ascii_case("hybrid") {
                    Some(EvalMode::Hybrid)
                } else {
                    None
                };
                if let Some(mode) = mode {
                    stop_active_search(active_search);
                    if let Ok(mut searcher) = searcher.lock() {
                        searcher.set_eval_mode(mode);
                    }
                }
            }
        }
    } else if option_name == "nnuepath" || option_name == "nnue_path" {
        if let Some(pos) = value_pos {
            if let Some(value_str) = tokens.get(pos + 1) {
                let path = value_str.to_string();
                stop_active_search(active_search);
                if let Ok(mut searcher) = searcher.lock() {
                    searcher.set_nnue_path(path);
                }
            }
        }
    }
}

struct GoParams {
    depth: u8,
    time_limit_ms: u64,
}

fn parse_go_params(parts: &mut std::str::SplitWhitespace, side_to_move: Color) -> GoParams {
    let mut depth = 64u8;
    let mut movetime_ms: Option<u64> = None;
    let mut wtime: Option<u64> = None;
    let mut btime: Option<u64> = None;
    let mut winc: u64 = 0;
    let mut binc: u64 = 0;
    let mut movestogo: Option<u64> = None;
    let mut infinite = false;

    while let Some(part) = parts.next() {
        match part {
            "depth" => {
                if let Some(value) = parts.next() {
                    depth = value.parse().unwrap_or(depth);
                }
            }
            "movetime" => {
                if let Some(value) = parts.next() {
                    movetime_ms = value.parse().ok();
                }
            }
            "wtime" => wtime = parts.next().and_then(|v| v.parse().ok()),
            "btime" => btime = parts.next().and_then(|v| v.parse().ok()),
            "winc" => winc = parts.next().and_then(|v| v.parse().ok()).unwrap_or(0),
            "binc" => binc = parts.next().and_then(|v| v.parse().ok()).unwrap_or(0),
            "movestogo" => movestogo = parts.next().and_then(|v| v.parse().ok()),
            "infinite" => infinite = true,
            _ => {}
        }
    }

    let time_limit_ms = if infinite {
        24 * 60 * 60 * 1000
    } else if let Some(movetime_ms) = movetime_ms {
        movetime_ms
    } else if let (Some(wtime), Some(btime)) = (wtime, btime) {
        let (side_time, increment) = match side_to_move {
            Color::White => (wtime, winc),
            Color::Black => (btime, binc),
        };
        let moves_to_go = movestogo.unwrap_or(30).max(1);
        let allotted = side_time / moves_to_go + increment;
        allotted.max(20).min(side_time.saturating_sub(50)).max(20)
    } else if depth != 64 {
        // Explicit depth search should not be cut off by the default 5s limit.
        24 * 60 * 60 * 1000
    } else {
        5_000
    };

    GoParams {
        depth,
        time_limit_ms,
    }
}
