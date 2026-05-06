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

#[cfg(nnue_level_v2)]
pub const NNUE_LEVEL: &str = "v2";
#[cfg(nnue_level_v3)]
pub const NNUE_LEVEL: &str = "v3";
#[cfg(not(any(nnue_level_v2, nnue_level_v3)))]
pub const NNUE_LEVEL: &str = "runtime";

#[cfg(nnue_level_v2)]
pub const MICROARCH: &str = "x86-64-v2";
#[cfg(nnue_level_v3)]
pub const MICROARCH: &str = "x86-64-v3";
#[cfg(not(any(nnue_level_v2, nnue_level_v3)))]
pub const MICROARCH: &str = "runtime";
