# =============================================================================
# Octopus — UCI-compatible chess engine written in Rust
# Copyright (c) 2026 Robin Kaluzny
# SPDX-License-Identifier: MIT
#
# This file is part of the Octopus project.
#
# Licensed under the MIT License; you may not use this file except in
# compliance with the License. See the LICENSE file in the project root
# for full license information.
#
# =============================================================================

import struct
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

PIECE_TYPES = 6  # pawn, knight, bishop, rook, queen, king
SQUARES = 64
INPUT_FEATURES = PIECE_TYPES * SQUARES * 2  # white pieces + black pieces

def piece_square_index(piece_type, square, side):
    """Convert piece+square to feature index (non-king-relative)."""
    return (piece_type - 1) * SQUARES + square + side * (PIECE_TYPES * SQUARES)

def king_relative_index(piece_type, square, king_square, stm, piece_color):
    """
    Convert to king-relative feature index matching Rust implementation.
    piece_type: 1-6 (1=pawn, ..., 6=king)
    square: 0-63
    king_square: 0-63 (king of side to move)
    stm: 0 for white to move, 1 for black to move
    piece_color: 0 for white piece, 1 for black piece
    """
    king_rank = king_square // 8
    king_file = king_square % 8
    sq_rank = square // 8
    sq_file = square % 8
    
    # Always compute relative to side to move's king
    if stm == 0:  # white to move: square - king
        rel_rank = (sq_rank - king_rank + 7) % 8
        rel_file = (sq_file - king_file + 7) % 8
    else:  # black to move: king - square (flipped perspective)
        rel_rank = (king_rank - sq_rank + 7) % 8
        rel_file = (king_file - sq_file + 7) % 8
    
    rel_square = rel_rank * 8 + rel_file
    
    # side bit: 0 if piece is same color as stm (our piece), 1 if opponent's piece
    side = 0 if piece_color == stm else 1
    
    return (piece_type - 1) * 64 + rel_square + side * (PIECE_TYPES * SQUARES)

class NNUEDataset(Dataset):
    """Dataset for NNUE training from binary format (supports single file or folder)."""
    
    def __init__(self, bin_path, clamp_eval=3000, transform_eval='tanh'):
        self.bin_files = []
        self.file_offsets = []  # cumulative record counts
        self.clamp_eval = clamp_eval
        self.transform_eval = transform_eval
        
        import os
        if os.path.isdir(bin_path):
            # Scan directory for .bin files
            files = sorted([os.path.join(bin_path, f) for f in os.listdir(bin_path) 
                          if f.endswith('.bin')])
            self.bin_files = files
        else:
            self.bin_files = [bin_path]
        
        if not self.bin_files:
            raise ValueError(f"No .bin files found in {bin_path}")
        
        # Detect format: check first file
        with open(self.bin_files[0], 'rb') as fp:
            fp.seek(0, 2)
            file_size = fp.tell()
            fp.seek(0)
            test_bytes = fp.read(112)
        
        # Detect format based on file size divisibility
        if file_size % 112 == 0:
            self.record_size = 112
            self.format = 'new'  # 12 piece bitboards
        elif file_size % 32 == 0:
            self.record_size = 32
            self.format = 'old'  # legacy format with occupancy bitboards only
        else:
            raise ValueError(f"Unknown file format: file size {file_size} not divisible by 32 or 112")
        
        print(f"Detected format: {self.format} ({self.record_size} bytes per record)")
        
        # Count records per file and build offset table
        self.file_record_counts = []
        total = 0
        for f in self.bin_files:
            with open(f, 'rb') as fp:
                fp.seek(0, 2)
                num = fp.tell() // self.record_size
            self.file_record_counts.append(num)
            self.file_offsets.append(total)
            total += num
        
        self.num_records = total
        self.file_offsets.append(total)  # sentinel
        
        print(f"Loaded {len(self.bin_files)} files, {self.num_records:,} total positions")
        
        # Mmap all files
        self.data_files = [np.memmap(f, dtype=np.uint8, mode='r') for f in self.bin_files]
    
    def __len__(self):
        return self.num_records
    
    def __getitem__(self, idx):
        # Find which file contains this index
        file_idx = 0
        for i in range(len(self.file_offsets) - 1):
            if idx < self.file_offsets[i + 1]:
                file_idx = i
                break
        
        local_idx = idx - self.file_offsets[file_idx]
        record = self.data_files[file_idx][local_idx * self.record_size:(local_idx + 1) * self.record_size]
        
        if self.format == 'old':
            raise RuntimeError(
                f"Old dataset format detected (32 bytes). Please regenerate your dataset "
                f"using the updated bin_converter.py to include per-piece bitboards. "
                f"Run: python python_scripts/bin_converter.py"
            )
        
        # New format: 112 bytes with per-piece bitboards
        hash_key = int.from_bytes(record[0:8], 'little')
        
        # Read 6 white piece bitboards (pawn, knight, bishop, rook, queen, king)
        white_pieces = []
        for i in range(6):
            offset = 8 + i * 8
            bb = int.from_bytes(record[offset:offset+8], 'little')
            white_pieces.append(bb)
        
        # Read 6 black piece bitboards
        black_pieces = []
        for i in range(6):
            offset = 56 + i * 8  # 8 + 6*8 = 56
            bb = int.from_bytes(record[offset:offset+8], 'little')
            black_pieces.append(bb)
        
        eval_cp = int.from_bytes(record[104:108], 'little', signed=True)
        stm = record[108]
        
        # Clamp eval
        eval_cp = np.clip(eval_cp, -self.clamp_eval, self.clamp_eval)
        
        # Transform eval (tanh scaling to [-1, 1])
        if self.transform_eval == 'tanh':
            target = np.tanh(eval_cp / 600.0)
        else:
            target = eval_cp / 100.0
        
        # Convert bitboards to feature indices
        features = self._bitboards_to_features(white_pieces, black_pieces, stm)
        
        return features, np.float32(target), stm
    
    def _bitboards_to_features(self, white_pieces, black_pieces, stm):
        """Convert per-piece bitboards to NNUE feature indices (king-relative)."""
        indices = []
        
        # Get king squares (piece index 5 = king)
        white_king_sq = -1
        if white_pieces[5] != 0:
            white_king_sq = white_pieces[5].bit_length() - 1
        
        black_king_sq = -1
        if black_pieces[5] != 0:
            black_king_sq = black_pieces[5].bit_length() - 1
        
        # King square for side to move
        king_sq = white_king_sq if stm == 0 else black_king_sq
        
        # Process white pieces (piece_color=0)
        for piece_idx in range(6):
            if piece_idx == 5:  # skip king (reference piece, not a feature)
                continue
            bb = white_pieces[piece_idx]
            sq = 0
            temp_bb = bb
            while temp_bb:
                if temp_bb & 1:
                    piece_type = piece_idx + 1  # convert to 1-6
                    if king_sq >= 0:
                        idx = king_relative_index(piece_type, sq, king_sq, stm, 0)
                        indices.append(idx)
                temp_bb >>= 1
                sq += 1
        
        # Process black pieces (piece_color=1)
        for piece_idx in range(6):
            if piece_idx == 5:  # skip king
                continue
            bb = black_pieces[piece_idx]
            sq = 0
            temp_bb = bb
            while temp_bb:
                if temp_bb & 1:
                    piece_type = piece_idx + 1
                    if king_sq >= 0:
                        idx = king_relative_index(piece_type, sq, king_sq, stm, 1)
                        indices.append(idx)
                temp_bb >>= 1
                sq += 1
        
        return np.array(indices, dtype=np.int64)
    
    def get_feature_indices(self, idx):
        """Get raw feature indices for sparse representation."""
        # Find which file contains this index
        file_idx = 0
        for i in range(len(self.file_offsets) - 1):
            if idx < self.file_offsets[i + 1]:
                file_idx = i
                break
        
        local_idx = idx - self.file_offsets[file_idx]
        record = self.data_files[file_idx][local_idx * self.record_size:(local_idx + 1) * self.record_size]
        
        if self.format == 'old':
            raise RuntimeError("Old dataset format detected. Please regenerate with updated bin_converter.py")
        
        # Read white pieces
        white_pieces = []
        for i in range(6):
            ob = 8 + i * 8
            bb = int.from_bytes(record[ob:ob+8], 'little')
            white_pieces.append(bb)
        
        # Read black pieces
        black_pieces = []
        for i in range(6):
            ob = 56 + i * 8
            bb = int.from_bytes(record[ob:ob+8], 'little')
            black_pieces.append(bb)
        
        stm = record[108]
        
        return self._bitboards_to_features(white_pieces, black_pieces, stm)

def collate_fn(batch):
    """Custom collate for sparse features."""
    features, targets, stms = zip(*batch)
    
    # Compute lengths of each sample's features
    lengths = [len(f) for f in features]
    
    # Concatenate all features into single tensor
    features = torch.tensor(np.concatenate(features), dtype=torch.long)
    
    # Compute offsets (cumulative sum of lengths, starting at 0)
    offsets = torch.zeros(len(lengths), dtype=torch.long)
    offsets[1:] = torch.tensor(lengths[:-1], dtype=torch.long).cumsum(0)
    
    targets = torch.tensor(targets, dtype=torch.float32)
    stms = torch.tensor(stms, dtype=torch.long)
    
    return features, offsets, targets, stms
