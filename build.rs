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

fn main() {
    println!("cargo:rerun-if-env-changed=NNUE_BUILD_LEVEL");
    println!("cargo:rustc-check-cfg=cfg(nnue_level_v2)");
    println!("cargo:rustc-check-cfg=cfg(nnue_level_v3)");

    let level = std::env::var("NNUE_BUILD_LEVEL").unwrap_or_else(|_| "v3".to_string());
    match level.as_str() {
        "v2" => println!("cargo:rustc-cfg=nnue_level_v2"),
        "v3" => println!("cargo:rustc-cfg=nnue_level_v3"),
        other => {
            println!("cargo:warning=Unknown NNUE_BUILD_LEVEL '{other}', defaulting to v3");
            println!("cargo:rustc-cfg=nnue_level_v3");
        }
    }
}
