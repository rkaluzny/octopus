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
import torch.nn.functional as F

INPUT_FEATURES = 6 * 64 * 2  # piece types * squares * sides

class NNUE(nn.Module):
    """
    NNUE with incremental accumulator.
    Architecture: sparse features -> accumulator (ReLU) -> hidden (ReLU) -> output
    """
    
    def __init__(self, accumulator_size=512, hidden_size=32):
        super().__init__()
        
        self.accumulator_size = accumulator_size
        self.hidden_size = hidden_size
        
        # Feature -> accumulator weights (sparse, one-hot like)
        self.fc_features = nn.Linear(INPUT_FEATURES, accumulator_size, bias=False)
        
        # King-relative accumulator (separate for white/black to move)
        # In practice, we maintain two accumulators and update incrementally
        
        # Hidden layer
        self.fc_hidden = nn.Linear(accumulator_size * 2, hidden_size)
        self.fc_output = nn.Linear(hidden_size, 1)
        
        self._init_weights()
    
    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc_features.weight, gain=1.0)
        nn.init.xavier_uniform_(self.fc_hidden.weight, gain=1.0)
        nn.init.zeros_(self.fc_output.weight)
        nn.init.zeros_(self.fc_output.bias)
    
    def forward(self, features, offsets, stms):
        """
        Forward pass with sparse features.
        features: concatenated feature indices
        offsets: starting index for each sample in features
        stms: side to move (0=white, 1=black)
        """
        batch_size = len(offsets)
        
        # Compute accumulator for each position
        accumulators = []
        for i in range(batch_size):
            start = offsets[i].item()
            end = offsets[i + 1].item() if i + 1 < batch_size else len(features)
            idx = features[start:end]
            
            # Sparse feature lookup
            acc = self.fc_features.weight[:, idx].sum(dim=1)
            acc = F.relu(acc)
            accumulators.append(acc)
        
        # Combine white/black perspective based on side to move
        # For NNUE, we typically concatenate both king-relative accumulators
        # Here we simplify: use stm to select perspective
        acc_tensor = torch.stack(accumulators)
        
        # In full NNUE, we'd concatenate both perspectives
        # Simplified: just use mirrored for opponent perspective
        acc_mirrored = torch.flip(acc_tensor, dims=[1])  # simplified mirroring
        
        # Concatenate both perspectives
        combined = torch.cat([acc_tensor, acc_mirrored], dim=1)
        
        # Hidden + output
        h = F.relu(self.fc_hidden(combined))
        out = self.fc_output(h).squeeze(-1)
        
        return out
    
    def forward_dense(self, feature_vectors):
        """
        Forward pass with dense feature vectors (for inference).
        feature_vectors: (batch, INPUT_FEATURES) float tensor
        """
        acc = F.relu(self.fc_features(feature_vectors))
        acc_mirrored = torch.flip(acc, dims=[1])
        combined = torch.cat([acc, acc_mirrored], dim=1)
        h = F.relu(self.fc_hidden(combined))
        return self.fc_output(h).squeeze(-1)
    
    def export_weights_int16(self):
        """Export weights in int16 format for Rust inference."""
        weights = {}
        
        # Feature weights: (accumulator_size, INPUT_FEATURES)
        fc_features_w = self.fc_features.weight.data
        fc_features_q = torch.clamp(fc_features_w * 127, -128, 127).to(torch.int8)
        weights['fc_features_weight'] = fc_features_q
        weights['fc_features_scale'] = 127.0
        
        # Hidden weights
        fc_hidden_w = self.fc_hidden.weight.data
        fc_hidden_b = self.fc_hidden.bias.data
        hidden_scale = 64.0
        weights['fc_hidden_weight'] = torch.clamp(fc_hidden_w * hidden_scale, -32768, 32767).to(torch.int16)
        weights['fc_hidden_bias'] = torch.clamp(fc_hidden_b * hidden_scale, -32768, 32767).to(torch.int16)
        weights['fc_hidden_scale'] = hidden_scale
        
        # Output weights
        fc_output_w = self.fc_output.weight.data
        fc_output_b = self.fc_output.bias.data
        output_scale = 128.0
        weights['fc_output_weight'] = torch.clamp(fc_output_w * output_scale, -32768, 32767).to(torch.int16)
        weights['fc_output_bias'] = torch.clamp(fc_output_b * output_scale, -32768, 32767).to(torch.int16)
        weights['fc_output_scale'] = output_scale
        
        return weights
    
    def save_quantized(self, path):
        """Save quantized weights for Rust loading in custom binary format."""
        weights = self.export_weights_int16()
        
        import struct
        
        with open(path.replace('.pt', '.bin'), 'wb') as f:
            # Header: "ONUE" magic + version
            f.write(b'ONUE')
            f.write(struct.pack('I', 1))  # version
            
            # Dimensions
            f.write(struct.pack('I', INPUT_FEATURES))      # input_features
            f.write(struct.pack('I', self.accumulator_size))  # accumulator_size
            f.write(struct.pack('I', self.hidden_size))    # hidden_size
            f.write(struct.pack('I', 1))                 # output_size
            
            # Scales as f32
            f.write(struct.pack('f', float(weights['fc_features_scale'])))
            f.write(struct.pack('f', float(weights['fc_hidden_scale'])))
            f.write(struct.pack('f', float(weights['fc_output_scale'])))
            f.write(struct.pack('f', 600.0))  # cp_scale
            
            # Feature weights (int8): (INPUT_FEATURES, accumulator_size)
            feature_w = weights['fc_features_weight']  # (accumulator_size, INPUT_FEATURES)
            # Transpose to (INPUT_FEATURES, accumulator_size) for Rust
            feature_w_t = feature_w.t().contiguous()
            f.write(feature_w_t.numpy().tobytes())
            
            # Hidden weights (int16): (hidden_size, accumulator_size * 2)
            hidden_w = weights['fc_hidden_weight']
            f.write(hidden_w.numpy().tobytes())
            
            # Hidden bias (int16)
            hidden_b = weights['fc_hidden_bias']
            f.write(hidden_b.numpy().tobytes())
            
            # Output weights (int16): (hidden_size, 1)
            output_w = weights['fc_output_weight']
            f.write(output_w.numpy().tobytes())
            
            # Output bias (int16)
            output_b = weights['fc_output_bias']
            f.write(output_b.numpy().tobytes())
