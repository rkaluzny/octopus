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

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import argparse
import os
import sys
import time

# Allow imports from the training directory regardless of working directory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from nnue_dataset import NNUEDataset, collate_fn
from nnue_model import NNUE

def evaluate(model, dataloader, device):
    """Evaluate model and return MAE and correlation."""
    model.eval()
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for features, offsets, targets, stms in dataloader:
            features = features.to(device)
            offsets = offsets.to(device)
            targets = targets.to(device)
            
            preds = model(features, offsets, stms)
            all_preds.append(preds.cpu().numpy())
            all_targets.append(targets.cpu().numpy())
    
    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)
    
    mae = np.mean(np.abs(all_preds - all_targets))
    correlation = np.corrcoef(all_preds, all_targets)[0, 1]
    
    return mae, correlation

def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() and args.gpu else 'cpu')
    print(f"Using device: {device}")
    
    # Create datasets (accept file or folder)
    train_path = args.train_dir if args.train_dir else args.train_bin
    train_dataset = NNUEDataset(train_path, clamp_eval=args.clamp_eval)
    
    val_dataset = None
    if args.val_dir or args.val_bin:
        val_path = args.val_dir if args.val_dir else args.val_bin
        val_dataset = NNUEDataset(val_path, clamp_eval=args.clamp_eval)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    if val_dataset:
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=collate_fn,
            pin_memory=True
        )
    
    # Create model
    model = NNUE(accumulator_size=args.accumulator_size, hidden_size=args.hidden_size)
    model = model.to(device)
    
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.MSELoss()
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Training samples: {len(train_dataset):,}")
    if val_dataset:
        print(f"Validation samples: {len(val_dataset):,}")
    
    best_corr = -1.0
    
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        num_batches = 0
        start_time = time.time()
        
        for batch_idx, (features, offsets, targets, stms) in enumerate(train_loader):
            features = features.to(device)
            offsets = offsets.to(device)
            targets = targets.to(device)
            
            optimizer.zero_grad()
            preds = model(features, offsets, stms)
            loss = criterion(preds, targets)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            if batch_idx % args.log_interval == 0:
                print(f"Epoch {epoch+1}/{args.epochs} | Batch {batch_idx}/{len(train_loader)} | Loss: {loss.item():.6f}")
        
        scheduler.step()
        
        avg_loss = total_loss / num_batches
        epoch_time = time.time() - start_time
        
        print(f"\nEpoch {epoch+1} | Avg Loss: {avg_loss:.6f} | Time: {epoch_time:.1f}s")
        
        # Validation
        if val_dataset:
            val_mae, val_corr = evaluate(model, val_loader, device)
            print(f"Validation | MAE: {val_mae:.4f} | Correlation: {val_corr:.4f}")
            
            if val_corr > best_corr:
                best_corr = val_corr
                torch.save(model.state_dict(), os.path.join(args.output_dir, 'best_model.pt'))
                print(f"Saved best model (correlation: {val_corr:.4f})")
        
        # Save checkpoint
        if (epoch + 1) % args.save_interval == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }, os.path.join(args.output_dir, f'checkpoint_epoch_{epoch+1}.pt'))
    
    # Save final model and export quantized weights
    torch.save(model.state_dict(), os.path.join(args.output_dir, 'final_model.pt'))
    model.save_quantized(os.path.join(args.output_dir, 'nnue_weights.bin'))
    print(f"\nTraining complete! Weights saved to {args.output_dir}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train NNUE for chess engine')
    
    parser.add_argument('--train-bin', type=str, default=None, help='Path to training binary file (or use --train-dir)')
    parser.add_argument('--val-bin', type=str, default=None, help='Path to validation binary file (or use --val-dir)')
    parser.add_argument('--train-dir', type=str, default=None, help='Path to folder with training .bin files')
    parser.add_argument('--val-dir', type=str, default=None, help='Path to folder with validation .bin files')
    parser.add_argument('--output-dir', type=str, default='./output', help='Output directory')
    
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=1024, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=1e-4, help='Weight decay')
    
    parser.add_argument('--accumulator-size', type=int, default=512, help='NNUE accumulator size')
    parser.add_argument('--hidden-size', type=int, default=256, help='Hidden layer size')
    
    parser.add_argument('--clamp-eval', type=int, default=3000, help='Clamp eval range')
    parser.add_argument('--num-workers', type=int, default=4, help='DataLoader workers')
    parser.add_argument('--log-interval', type=int, default=100, help='Log interval')
    parser.add_argument('--save-interval', type=int, default=10, help='Save interval')
    parser.add_argument('--gpu', action='store_true', help='Use GPU if available')
    
    args = parser.parse_args()
    
    if not args.train_bin and not args.train_dir:
        parser.error("Must specify either --train-bin or --train-dir")
    
    os.makedirs(args.output_dir, exist_ok=True)
    train(args)
