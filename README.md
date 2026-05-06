<p align="center">
  <img src="docs/logo.png" alt="Octopus Engine Logo" width="200"/>
</p>

<h1 align="center">Octopus</h1>

<p align="center">
  A UCI-compatible chess engine written in Rust.
</p>

<p align="center">
  <a href="#features">Features</a> &middot;
  <a href="#search">Search</a> &middot;
  <a href="#evaluation">Evaluation</a> &middot;
  <a href="#nnue">NNUE</a> &middot;
  <a href="#data-generation">Data Generation</a> &middot;
  <a href="#training">Training</a> &middot;
  <a href="#build-and-run">Build</a> &middot;
  <a href="#license">License</a>
</p>

---

## About

Octopus is a chess engine implemented in Rust with a focus on speed, correctness, and modern engine design. It uses a bitboard-based representation, an alpha-beta search with a wide array of pruning and reduction techniques, and a handcrafted evaluation derived from the PeSTO framework. An NNUE evaluation path is fully integrated with SIMD acceleration, but no trained network weights are currently available.

---

## Search

The engine uses a Principal Variation Search (PVS) framework with iterative deepening and aspiration windows. The following techniques are implemented:

### Core Algorithm

| Technique | Description |
|---|---|
| **PVS** | Principal Variation Search; first move at full window, remaining moves with zero-width searches |
| **Iterative Deepening** | Depth increases from 1 to the target; best move and score carried forward |
| **Aspiration Windows** | Window centered on previous iteration's score; re-searches on fail-high/fail-low |

### Pruning & Reductions

| Technique | Condition | Effect |
|---|---|---|
| **Late Move Reductions (LMR)** | Quiet moves after the first few at depth >= 4 | Reduced-depth zero-window search |
| **Late Move Pruning (LMP)** | Quiet moves beyond a depth-dependent count threshold at shallow depths | Pruned without search |
| **Null Move Pruning** | Not in check, sufficient material, static eval >= beta at depth >= 3 | Pass the turn; if score >= beta, prune |
| **Futility Pruning** | Depth <= 3, not in check, not PV, static eval + margin < alpha | Pruned without search |
| **Razoring** | Depth <= 2, static eval + margin < alpha | Quick quiescence check to prune |
| **Reverse Futility Pruning** | Depth 1-2, static eval - margin >= beta | Pruned and return static eval |
| **Probcut** | Depth >= 10, quiet moves beyond move threshold | Reduced-depth search for fast cutoffs |
| **Delta Pruning** | Quiescence search, stand_pat + margin < alpha | Skip captures unlikely to raise alpha |

### Extensions

| Technique | Description |
|---|---|
| **Singular Extension** | TT exact score significantly above threshold at PV nodes; tested move is singular | Extension by one ply |

### Move Ordering

Moves are scored and sorted before search:

1. TT move (highest priority)
2. Captures (MVV-LVA: Most Valuable Victim -- Least Valuable Attacker)
3. Killer moves (slot 0, then slot 1)
4. History heuristic (piece-specific from/to history)
5. Counter-move history
6. Counter-move match

### Other Features

- **Transposition Table** with depth-preferred replacement and age-based aging
- **Killer Heuristic** per ply (two slots)
- **History Heuristic** (from/to and piece-specific tables)
- **Counter-Move Heuristic**
- **Quiescence Search** with captures, promotions, and early-ply quiet checks

---

## Evaluation

The default evaluation mode is a **Handcrafted Evaluation (HCE)** based on the PeSTO framework. It uses a tapered evaluation that blends a middlegame score and an endgame score based on the remaining material phase.

### Components

| Component | Description |
|---|---|
| **Material** | Piece values for pawn, knight, bishop, rook, queen |
| **Piece-Square Tables (PSTs)** | Positional bonuses derived from PeSTO for all piece types |
| **Pawn Structure** | Penalties for isolated and doubled pawns |
| **Passed Pawns** | Bonus scaling with proximity to promotion; king support bonus |
| **Mobility** | Safe-move counting for knights, bishops, rooks, and queens (excludes squares attacked by enemy pawns) |
| **King Safety** | Enemy attacker counting, pawn shield, open-file penalty near king |
| **Outposts** | Knight and bishop outposts on pawn-safe squares in enemy territory |
| **Rook on Open File** | Bonus for rooks on fully or semi-open files |
| **Trapped Bishops** | Penalty for bishops blocked by own pawns in the opening |
| **Bishop Pair** | Bonus when both bishops are present |
| **Castling Bonus** | Encourages king safety in the middlegame |
| **Development** | Minor piece development and center control bonuses scaled by opening factor |
| **Tempo** | Small bonus for the side to move, fading toward the endgame |

The final score is computed as:

```
score = (mg_score * phase + eg_score * (24 - phase)) / 24
```

where `phase` is derived from remaining pieces (total max phase = 24).

---

## NNUE

Octopus includes a fully functional **Efficiently Updatable Neural Network (NNUE)** evaluation path.

### Architecture

```
Input (768 features, king-relative)
    |
Accumulator (768 -> 512, int8 weights, ReLU)
    |
Hidden Layer (1024 -> 256, int16 weights, ReLU)
    |
Output (256 -> 1, int16 weights)
    |
Centipawn score (clamped to +/- 30,000)
```

### Implementation Details

- **Incremental accumulator updates** during search; full rebuild only on king moves
- **King-relative feature indexing** for compact representation
- **SIMD acceleration**: SSE2 (v2) and AVX2 (v3) backends with scalar fallback
- **Quantized weights** (int8/int16) loaded from a binary file with a custom ONUE format

### Current Status

**NNUE is fully implemented and integrated, but no trained network weights are currently distributed.** The engine ships with a zeroed-out weight file as a placeholder. The Python training pipeline (in `training/`) can be used to train a network from game data, but a pre-trained `nnue_weights.bin` file is not yet available.

When weights are loaded, the engine can be switched between evaluation modes via the UCI `EvalMode` option:

- `HCE` -- Handcrafted evaluation only (default)
- `NNUE` -- Neural network evaluation
- `Hybrid` -- 70% NNUE + 30% HCE blend

---

## Board Representation

- **Bitboard-based** board with per-piece-type and per-color bitboards
- **Zobrist hashing** with incremental update for fast transposition table lookups
- **Magic bitboard** attack tables for rooks and bishops
- Full **FEN parsing** and **UCI move conversion**
- Legal move generation with pseudo-legal filtering and pin/check verification

---

## Data Generation

Octopus includes a complete Python-based data generation pipeline in `python_scripts/`. The pipeline converts raw chess data (PGN files or Lichess games) into labeled positions, then serializes them into a binary format for NNUE training.

### Pipeline Overview

```
PGN / Lichess Games
        |
        v
  pgn_processor.py          # Extract positions from PGN
        |
        v
  lichess_eval.py           # Evaluate positions with the engine
        |
        v
  selfplay.py               # Generate labeled positions via self-play
        |
        v
  bin_converter.py          # Convert text output to binary format
        |
        v
  .bin files (training data)
```

### Step 1: Extract Positions from PGN

`pgn_processor.py` reads a PGN file and extracts unique positions, skipping the opening phase and sampling at regular intervals:

```python
# Configuration
INPUT_PGN = "input.pgn"       # Source PGN file
OUTPUT_FILE = "positions.txt" # Output FEN list

SKIP_FULL_MOVES = 4           # Skip first N full moves (opening book)
SAMPLE_INTERVAL = 3           # Take every Nth position
```

### Step 2: Evaluate Positions

Two parallel evaluation scripts generate labeled data:

**`lichess_eval.py`** -- Evaluates positions from a FEN list by running the engine at shallow depth with a stability check (double-eval, triple-eval on disagreement). Positions with extreme evaluations are downsampled to maintain a balanced dataset.

**`selfplay.py`** -- The engine plays games against itself, evaluating each position with stability verification. Random moves are occasionally played (default 8%) to increase diversity. Positions are deduplicated via MD5 hash.

Both scripts use:
- Multi-threaded workers (configurable `NUM_THREADS`)
- Engine restart on failure
- Batched writing to disk
- Evaluation clamping (default +/- 1500-1700 cp)
- Stability checks: two quick evals, third deeper eval if disagreement exceeds threshold

### Step 3: Convert to Binary Format

`bin_converter.py` converts the text output (`fen | eval_cp` format) into a compact binary format (112 bytes per position):

| Offset | Size | Field |
|---|---|---|
| 0 | 8 | Zobrist hash key |
| 8-55 | 48 | 6 white piece bitboards (8 bytes each) |
| 56-103 | 48 | 6 black piece bitboards (8 bytes each) |
| 104 | 4 | Evaluation in centipawns (signed int32) |
| 108 | 1 | Side to move (0=white, 1=black) |
| 109 | 1 | Castling rights bitmask |
| 110 | 1 | En passant square (255=none) |
| 111 | 1 | Padding (always 0) |

### Step 4: Compare Against Reference

`compare.py` compares your engine's evaluation against Stockfish on a sample of positions, reporting:
- Average raw difference in centipawns
- RMSE
- Classification agreement (who is winning, equal)
- Mismatch analysis (critical: sign disagreements)

### Quick Start

On Windows, run `datagen.bat` to launch both evaluation pipelines in parallel:

```bat
python pgn_processor.py
start "" /B python lichess_eval.py
start "" /B python selfplay.py
```

---

## Training

The NNUE training pipeline is located in `training/` and uses PyTorch. It reads binary dataset files, trains the network, and exports quantized weights compatible with the Rust engine.

### Architecture

```
Sparse Features (king-relative, 768 total)
        |
Accumulator (768 -> 512, int8, ReLU)
        |
Hidden Layer (1024 -> 256, int16, ReLU)
        |
Output (256 -> 1, int16)
        |
Centipawn score
```

### Files

| File | Purpose |
|---|---|
| `nnue_dataset.py` | Binary dataset loader with king-relative feature extraction, memory-mapped file support |
| `nnue_model.py` | PyTorch NNUE model with quantized weight export in ONUE binary format |
| `train_nnue.py` | Training script with MAE/correlation metrics, checkpoints, and weight export |

### Dataset Format

The dataset loader supports both single files and directories of `.bin` files. It auto-detects the format (112-byte new format or 32-byte legacy format) and raises an error if legacy data is detected.

Features are extracted as king-relative indices matching the Rust engine's `features.rs` implementation. The side-to-move's king position determines the coordinate system for all pieces.

### Usage

```bash
# Single file
python train_nnue.py \
    --train-bin dataset.bin \
    --val-bin dataset_val.bin \
    --output-dir ./output \
    --epochs 50 \
    --batch-size 1024 \
    --accumulator-size 512 \
    --hidden-size 256

# Multiple files (recommended)
python train_nnue.py \
    --train-dir ./datasets/train/ \
    --val-dir ./datasets/val/ \
    --output-dir ./output \
    --epochs 50 \
    --batch-size 1024 \
    --hidden-size 256
```

### Arguments

| Argument | Default | Description |
|---|---|---|
| `--train-bin` | - | Path to training binary file |
| `--train-dir` | - | Path to folder with training `.bin` files |
| `--val-bin` | - | Path to validation binary file |
| `--val-dir` | - | Path to folder with validation `.bin` files |
| `--output-dir` | `./output` | Output directory for models and weights |
| `--epochs` | 50 | Number of training epochs |
| `--batch-size` | 1024 | Batch size |
| `--lr` | 1e-3 | Learning rate |
| `--weight-decay` | 1e-4 | AdamW weight decay |
| `--accumulator-size` | 512 | NNUE accumulator size |
| `--hidden-size` | 256 | Hidden layer size |
| `--clamp-eval` | 3000 | Clamp eval to +/- N centipawns |
| `--gpu` | false | Use GPU if available |

### Output Files

After training, the following files are produced:

| File | Description |
|---|---|
| `best_model.pt` | Best model by validation correlation |
| `final_model.pt` | Final model weights (PyTorch) |
| `nnue_weights.bin` | Quantized weights (int8/int16) in ONUE format for the Rust engine |
| `checkpoint_epoch_N.pt` | Periodic checkpoints with optimizer state |

### Evaluation Metrics

The training script reports:
- **MAE** (Mean Absolute Error) on validation set -- lower is better
- **Correlation** (Pearson) between predictions and tanh-scaled targets -- higher is better

Typical targets: MAE < 50 cp, Correlation > 0.90.

### Weight Export

Weights are exported in the custom ONUE binary format:

```
Header: "ONUE" (4 bytes) + version (4 bytes)
Dimensions: input_features, accumulator_size, hidden_size, output_size
Scales: feature_scale, hidden_scale, output_scale, cp_scale (f32 each)
Feature weights: int8[input_features][accumulator_size]
Hidden weights: int16[hidden_size][accumulator_size * 2]
Hidden bias: int16[hidden_size]
Output weights: int16[1][hidden_size]
Output bias: int16[1]
```

Place the resulting `nnue_weights.bin` in the engine's `output/` directory (or set the path via `setoption name NnuePath`).

---

## Benchmarks

The engine includes built-in benchmarking commands:

```bash
# Evaluate performance
cargo run --release -- bench nnue 250000

# Search performance
cargo run --release -- bench search 6 2000

# Compare HCE vs NNUE
cargo run --release -- bench compare 6 2000
```

---

## Build and Run

```bash
cargo build --release
./target/release/octopus
```

The engine communicates via the UCI protocol and can be used with any compatible chess GUI (Arena, CuteChess, BanksiaGUI, etc.).

### NNUE Build Levels

The engine supports two SIMD build levels controlled by the `NNUE_BUILD_LEVEL` environment variable:

| Level | Description |
|---|---|
| `v2` | SSE2 acceleration (x86-64-v2) |
| `v3` | AVX2 acceleration (x86-64-v3, default) |

```bash
NNUE_BUILD_LEVEL=v3 cargo build --release
```

---

## Development Note

Parts of this project were developed using AI-assisted workflows. However, the engine itself is fully self-contained and does not depend on any external AI systems at runtime.

All code is transparent, auditable, and designed to be understood and modified by developers.

---

## Contributing

Contributions are welcome. Areas of interest include:

- Search improvements (move ordering, pruning, extensions)
- Evaluation tuning (HCE parameters, NNUE training data and architecture)
- Performance optimizations (SIMD, cache locality, memory layout)
- Bug fixes and testing
- Tooling (data generation, training pipelines, benchmarking)

---

## License

This project is licensed under the MIT License.

See the [LICENSE](LICENSE) file for full details.
