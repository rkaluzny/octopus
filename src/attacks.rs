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

// Precomputed magic bitboard attack tables.

use crate::board::Bitboard;
use lazy_static::lazy_static;

// Magic bitboard parameters for bishops.
const BISHOP_BITS: [u8; 64] = [
    6, 5, 5, 5, 5, 5, 5, 6, 5, 5, 5, 5, 5, 5, 5, 5,
    5, 5, 7, 7, 7, 7, 5, 5, 5, 5, 7, 9, 9, 7, 5, 5,
    5, 5, 7, 9, 9, 7, 5, 5, 5, 5, 7, 7, 7, 7, 5, 5,
    5, 5, 7, 7, 7, 7, 5, 5, 6, 5, 5, 5, 5, 5, 5, 6,
];

const BISHOP_MAGICS: [u64; 64] = [
    0x0814282217e20201, 0x1038628801530084, 0x4f1c03022602a102, 0x1831040084406881,
    0x1184042084980212, 0x0442063220080001, 0x110108b014201222, 0xb026018488085219,
    0x8205212004052ac9, 0xa821d0020e064609, 0x22ea0802114a0061, 0x810b1c4040888221,
    0x4204811040040b2c, 0x8104860806481c31, 0x1412a08c2120b012, 0x61150e0080941001,
    0x2140005070078d01, 0x0490148902080450, 0x60024310038200c4, 0x040d067024018019,
    0x04c5900404200001, 0x880440060100a02c, 0x0480c00092182009, 0x4000450704128c29,
    0x7424120020608509, 0x20329218a8081810, 0x0408060234013202, 0x9112008188008cc0,
    0x024100101101c004, 0x005014800d0084a4, 0x888287005584101a, 0x00e1c90012450842,
    0x0014031809c0908c, 0x060f305802364801, 0xac02020602131801, 0x1d09202020080081,
    0x00820084000600e1, 0x01860eca00a90091, 0x4098190247051829, 0x0146024204a1009a,
    0xe00c02600602b05a, 0x84118e900894500a, 0x4482913804009801, 0x06806016c4000801,
    0x30202821040044c2, 0x31401008cc418281, 0x01680800c4004281, 0x02080e04da010144,
    0x164400ac05202811, 0x5106006202300e04, 0x70512f24120800e0, 0xc1195029c2061131,
    0x202a810830740061, 0x24404803181a0198, 0x8920620606242301, 0xa8140922020201a1,
    0x0432410090012061, 0x042c034402082241, 0x0020065903511012, 0x2c18190420618822,
    0x88c208c431a2020c, 0x0c24401c11820202, 0x83e340050809906c, 0x8044200081160d81,
];

// Magic bitboard parameters for rooks.
const ROOK_BITS: [u8; 64] = [
    12, 11, 11, 11, 11, 11, 11, 12, 11, 10, 10, 10, 10, 10, 10, 11,
    11, 10, 10, 10, 10, 10, 10, 11, 11, 10, 10, 10, 10, 10, 10, 11,
    11, 10, 10, 10, 10, 10, 10, 11, 11, 10, 10, 10, 10, 10, 10, 11,
    11, 10, 10, 10, 10, 10, 10, 11, 12, 11, 11, 11, 11, 11, 11, 12,
];

const ROOK_MAGICS: [u64; 64] = [
    0x8080004002233780, 0x0240002001c0500c, 0x190008c101200151, 0x510009003000210c,
    0x9a00201200085044, 0x050001000c001812, 0xf200013382001804, 0x820005812dc30402,
    0x04088008824002a1, 0x81124010002002c0, 0xc4350040a0050011, 0x8001001001001821,
    0x9002000408502201, 0x4001800a004c0081, 0x8004002428031032, 0x28020000c4090182,
    0xc20b208000400c8c, 0x481240401000a004, 0x4c09220040803201, 0x2070010031090021,
    0xc802920022007208, 0x8338808004006201, 0x409484000a28b001, 0x290242000a410484,
    0x868c4008800c6082, 0xd0601000c000c721, 0x0088924100200101, 0xea0a4122000a0111,
    0x3188008080040049, 0x2402010200086410, 0x0021e81400304209, 0x34a0624200198411,
    0x264006ae81800442, 0xe2a080c003002102, 0xe07004d182802002, 0x1800800801805001,
    0x111d00380100304c, 0xe00e001042000538, 0x4864902104000812, 0x02000c0a820004c9,
    0x80a9c00032848002, 0x0c11201000444001, 0x0120050130410022, 0x0d00385001010021,
    0x00020004500a0020, 0x410600101802008c, 0x2300100322540038, 0x2840088c00d20021,
    0x0180004100806900, 0x4d8908c200238e00, 0x020a89d003e00080, 0x828260300300db00,
    0x2b01027008001500, 0x8304407460100801, 0x001008021005cc00, 0x0c03840102935600,
    0x0847044020811202, 0x0005023580204001, 0x1e30804022024832, 0x0a0e04b900601001,
    0x8885001044024801, 0x8211000694004801, 0xa10600110c982402, 0x9140440518408022,
];

lazy_static! {
    static ref BISHOP_ATTACKS: Vec<Bitboard> = init_magics(&BISHOP_MAGICS, &BISHOP_BITS, true);
    static ref ROOK_ATTACKS: Vec<Bitboard> = init_magics(&ROOK_MAGICS, &ROOK_BITS, false);
    static ref BISHOP_MASKS: [Bitboard; 64] = init_masks(true);
    static ref ROOK_MASKS: [Bitboard; 64] = init_masks(false);
    static ref BISHOP_OFFSETS: [usize; 64] = init_offsets(&BISHOP_BITS);
    static ref ROOK_OFFSETS: [usize; 64] = init_offsets(&ROOK_BITS);
}

#[inline]
pub fn get_rook_attacks(sq: u8, blockers: Bitboard) -> Bitboard {
    let sq_idx = sq as usize;
    let blockers = blockers & ROOK_MASKS[sq_idx];
    let magic_index = blockers.wrapping_mul(ROOK_MAGICS[sq_idx]) >> (64 - ROOK_BITS[sq_idx]);
    ROOK_ATTACKS[ROOK_OFFSETS[sq_idx] + magic_index as usize]
}

#[inline]
pub fn get_bishop_attacks(sq: u8, blockers: Bitboard) -> Bitboard {
    let sq_idx = sq as usize;
    let blockers = blockers & BISHOP_MASKS[sq_idx];
    let magic_index = blockers.wrapping_mul(BISHOP_MAGICS[sq_idx]) >> (64 - BISHOP_BITS[sq_idx]);
    BISHOP_ATTACKS[BISHOP_OFFSETS[sq_idx] + magic_index as usize]
}

fn init_offsets(bits: &[u8; 64]) -> [usize; 64] {
    let mut offsets = [0; 64];
    let mut current = 0;
    for i in 0..64 {
        offsets[i] = current;
        current += 1 << bits[i];
    }
    offsets
}

fn init_masks(is_bishop: bool) -> [Bitboard; 64] {
    let mut masks = [0; 64];
    for sq in 0..64 {
        masks[sq] = mask_sliding_attacks(sq as u8, is_bishop);
    }
    masks
}

fn init_magics(magics: &[u64; 64], bits: &[u8; 64], is_bishop: bool) -> Vec<Bitboard> {
    let total_size: usize = (0..64).map(|i| 1 << bits[i]).sum();
    let mut attacks = vec![0; total_size];
    let offsets = init_offsets(bits);
    let masks = init_masks(is_bishop);

    for sq in 0..64 {
        let mask = masks[sq];
        let num_bits = bits[sq];
        let offset = offsets[sq];
        let mut occupancy = 0u64;

        loop {
            let magic_index = occupancy.wrapping_mul(magics[sq]) >> (64 - num_bits);
            attacks[offset + magic_index as usize] = slow_sliding_attacks(sq as u8, occupancy, is_bishop);
            occupancy = occupancy.wrapping_sub(mask) & mask;
            if occupancy == 0 { break; }
        }
    }
    attacks
}

fn mask_sliding_attacks(sq: u8, is_bishop: bool) -> Bitboard {
    let mut result = 0u64;
    let r = (sq / 8) as i8;
    let f = (sq % 8) as i8;
    let deltas = if is_bishop {
        [(-1,-1), (-1,1), (1,-1), (1,1)]
    } else {
        [(-1,0), (1,0), (0,-1), (0,1)]
    };

    for (dr, df) in deltas {
        let mut nr = r + dr;
        let mut nf = f + df;
        while nr >= 0 && nr <= 7 && nf >= 0 && nf <= 7 {
            let next_r = nr + dr;
            let next_f = nf + df;
            if next_r < 0 || next_r > 7 || next_f < 0 || next_f > 7 {
                break;
            }
            result |= 1 << (nr * 8 + nf);
            nr += dr;
            nf += df;
        }
    }
    result
}

fn slow_sliding_attacks(sq: u8, blockers: Bitboard, is_bishop: bool) -> Bitboard {
    let mut attacks = 0u64;
    let r = (sq / 8) as i8;
    let f = (sq % 8) as i8;
    let deltas = if is_bishop {
        [(-1,-1), (-1,1), (1,-1), (1,1)]
    } else {
        [(-1,0), (1,0), (0,-1), (0,1)]
    };

    for (dr, df) in deltas {
        let mut nr = r + dr;
        let mut nf = f + df;
        while nr >= 0 && nr <= 7 && nf >= 0 && nf <= 7 {
            let bit = 1 << (nr * 8 + nf);
            attacks |= bit;
            if (blockers & bit) != 0 { break; }
            nr += dr;
            nf += df;
        }
    }
    attacks
}

#[inline]
pub fn get_knight_attacks(sq: u8) -> Bitboard {
    let mut attacks = 0u64;
    let file = (sq % 8) as i8;
    let rank = (sq / 8) as i8;
    let knight_moves = [
        (-2, -1), (-2, 1), (-1, -2), (-1, 2),
        (1, -2), (1, 2), (2, -1), (2, 1),
    ];
    for (df, dr) in knight_moves {
        let new_file = file + df;
        let new_rank = rank + dr;
        if new_file >= 0 && new_file < 8 && new_rank >= 0 && new_rank < 8 {
            attacks |= 1u64 << (new_rank * 8 + new_file);
        }
    }
    attacks
}

#[inline]
pub fn get_king_attacks(sq: u8) -> Bitboard {
    let mut attacks = 0u64;
    let file = (sq % 8) as i8;
    let rank = (sq / 8) as i8;
    let deltas = [
        (-1,-1), (-1,0), (-1,1),
        (0,-1),          (0,1),
        (1,-1),  (1,0),  (1,1),
    ];
    for (dr, df) in deltas {
        let nr = rank + dr;
        let nf = file + df;
        if nr >= 0 && nr < 8 && nf >= 0 && nf < 8 {
            attacks |= 1u64 << (nr * 8 + nf);
        }
    }
    attacks
}

#[inline]
pub fn get_pawn_attacks(sq: u8, color: crate::board::Color) -> Bitboard {
    let mut attacks = 0u64;
    let file = (sq % 8) as i8;
    let rank = (sq / 8) as i8;
    if color == crate::board::Color::White {
        if file > 0 && rank < 7 {
            attacks |= 1u64 << ((rank + 1) * 8 + (file - 1));
        }
        if file < 7 && rank < 7 {
            attacks |= 1u64 << ((rank + 1) * 8 + (file + 1));
        }
    } else {
        if file > 0 && rank > 0 {
            attacks |= 1u64 << ((rank - 1) * 8 + (file - 1));
        }
        if file < 7 && rank > 0 {
            attacks |= 1u64 << ((rank - 1) * 8 + (file + 1));
        }
    }
    attacks
}
