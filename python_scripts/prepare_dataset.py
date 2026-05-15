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

# Prepare dataset: combine bin files, deduplicate, shuffle, split train/val
import os
import struct
import random

# CONFIG

INPUT_FOLDER = "../data"
TRAIN_FILE = "plankton_train.bin"
VAL_FILE = "plankton_val.bin"

VAL_SPLIT = 0.01
RANDOM_SEED = 2372

# Format: hash(8) + 6 white piece BBs(8*6) + 6 black piece BBs(8*6) + eval(4) + stm(1) + castling(1) + ep(1) + pad(1)
STRUCT_FORMAT = "<QQQQQQQQQQQQQiBBBB"  # 13 Qs + i + 4 Bs = 112 bytes
RECORD_SIZE = 112

def read_bin_files(folder):
    records = []
    seen_hashes = set()

    bin_files = [f for f in os.listdir(folder) if f.endswith(".bin")]
    print(f"Found {len(bin_files)} bin files")

    for filename in bin_files:
        filepath = os.path.join(folder, filename)
        print(f"Reading {filename}...")

        with open(filepath, "rb") as f:
            while True:
                data = f.read(RECORD_SIZE)
                if not data or len(data) < RECORD_SIZE:
                    break

                unpacked = struct.unpack(STRUCT_FORMAT, data)
                hash_key = unpacked[0]

                if hash_key in seen_hashes:
                    continue

                seen_hashes.add(hash_key)
                records.append(data)

        print(f"  Loaded {len(records)} unique records so far")

    return records


def main():
    random.seed(RANDOM_SEED)

    print("Loading and deduplicating positions...")
    records = read_bin_files(INPUT_FOLDER)
    print(f"Total unique positions: {len(records)}")

    print("Shuffling...")
    random.shuffle(records)

    split_idx = int(len(records) * (1 - VAL_SPLIT))
    train_records = records[:split_idx]
    val_records = records[split_idx:]

    print(f"Train: {len(train_records)}")
    print(f"Val: {len(val_records)}")

    print(f"Writing {TRAIN_FILE}...")
    with open(TRAIN_FILE, "wb") as f:
        for record in train_records:
            f.write(record)

    print(f"Writing {VAL_FILE}...")
    with open(VAL_FILE, "wb") as f:
        for record in val_records:
            f.write(record)

    print("\n=== DONE ===")


if __name__ == "__main__":
    main()
