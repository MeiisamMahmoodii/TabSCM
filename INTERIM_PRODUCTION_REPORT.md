# TabSCM Production Training - Interim Report

**Report Generated**: 2025-11-24 12:20 (Training still in progress)

---

## Executive Summary

This report documents the complete development journey of the TabSCM (Tabular Structural Causal Model) project, from initial refactoring through multiple optimization attempts, culminating in the current production-scale training run.

**Current Production Training Status**:
- **Training Time**: 19+ hours elapsed
- **Progress**: ~80% complete (estimated 40,000+/50,000 steps)
- **Configuration**: 100 SCM runs, graphs up to 100 nodes, TabPFN-style embeddings
- **Status**: Running smoothly with stable loss convergence

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Development Timeline](#2-development-timeline)
3. [Architecture Evolution](#3-architecture-evolution)
4. [Training Experiments](#4-training-experiments)
5. [Current Production Run](#5-current-production-run)
6. [Technical Details](#6-technical-details)
7. [Challenges and Solutions](#7-challenges-and-solutions)

---

## 1. Project Overview

### 1.1 Objective
Develop a Transformer-based model for **zero-shot causal discovery** on tabular data that can:
- Learn from synthetic Structural Causal Models (SCMs)
- Infer causal graphs from observational and interventional data
- Generalize to unseen graph structures and mechanisms

### 1.2 Core Approach
- **Data Generation**: Random DAGs with diverse causal mechanisms (Linear, MLP, Sine, Tanh, Quadratic, Threshold)
- **Model Architecture**: Transformer-based encoder with specialized embeddings
- **Training**: Online learning from infinite stream of synthetic SCMs
- **Evaluation**: Zero-shot performance on held-out test graphs

---

## 2. Development Timeline

### Phase 1: Project Setup (Steps 1-50)
**Goal**: Transform Jupyter notebook into production-ready codebase

**Achievements**:
- ✅ Created modular project structure (`src/` directory)
- ✅ Implemented dependency management with `uv`
- ✅ Separated concerns: data generation, model, training, testing
- ✅ Added configuration system (`config.yaml`)

### Phase 2: Testing Infrastructure (Steps 51-100)
**Goal**: Add zero-shot evaluation capabilities

**Achievements**:
- ✅ Implemented comprehensive metrics (SHD, AUROC, F1, Precision, Recall)
- ✅ Created `src/test.py` for model evaluation
- ✅ Added model checkpointing

### Phase 3: Visualization & Reporting (Steps 101-150)
**Goal**: Create comprehensive reporting system

**Achievements**:
- ✅ Implemented loss tracking and visualization
- ✅ Created adjacency matrix heatmaps
- ✅ Added graph structure visualizations
- ✅ Automated report generation (`src/report.py`)

### Phase 4: GitHub Publication (Steps 151-200)
**Goal**: Publish project to GitHub

**Achievements**:
- ✅ Repository: `MeiisamMahmoodii/TabSCM`
- ✅ Comprehensive documentation
- ✅ Professional README and research report

### Phase 5: Debugging & Optimization (Steps 201-400)
**Goal**: Diagnose and fix model learning failure

**Key Issues Identified**:
1. **Mode Collapse**: Model predicting no edges (AUROC ≈ 0.5, F1 = 0.0)
2. **Sparsity Trap**: Causal graphs are sparse, model finds local minimum
3. **Memory Constraints**: OOM errors with larger batch sizes

**Solutions Implemented**:
- ✅ Gradient flow analysis
- ✅ Weighted BCE loss (pos_weight ≈ 2.33)
- ✅ Curriculum learning (linear mechanisms only)
- ✅ Gradient accumulation (effective batch size = 32)
- ✅ Mixed precision (AMP) - later disabled due to NaN loss

**Results**: Minimal improvement (AUROC 0.50 → 0.54, F1 still 0.0)

### Phase 6: Advanced Improvements (Steps 401-600)
**Goal**: Implement aggressive fixes

**Implementations**:
- ✅ Focal Loss (alpha=0.25, gamma=3.0)
- ✅ Simplified architecture (SimpleEmbedding)
- ✅ Auxiliary task (intervention prediction)
- ✅ Higher learning rate (0.0001 → 0.001)
- ✅ Larger model (embed_dim=192, n_layers=6, n_heads=8)

**Results**: Better loss convergence (0.007), but AUROC still ≈ 0.51, F1 = 0.0

### Phase 7: TabPFN-Style Embeddings (Steps 601-750)
**Goal**: Replace simple embeddings with rich, context-aware representations

**Implementation**:
- ✅ Row-level Transformer encoder (2 layers, 4 heads)
- ✅ Positional encoding for row positions
- ✅ Cross-attention summarization with learnable queries
- ✅ Captures full distribution, not just mean

**Debug Results** (5000 steps):
- Loss: 0.0034 (excellent convergence)
- AUROC: 0.4954 (no improvement)
- F1: 0.0 (still no edges predicted)

### Phase 8: Production-Scale Training (Current)
**Goal**: Full-scale training with maximum data and training steps

**Configuration**:
- 100 SCM runs (20 small, 80 large)
- Graphs: 10-100 nodes
- 50,000 training steps
- TabPFN-style embeddings
- Mixed mechanisms (50% linear, 50% non-linear)

**Status**: In progress (~80% complete)

---

## 3. Architecture Evolution

### 3.1 Initial Architecture (SetEmbedding)
```python
class SetEmbedding(nn.Module):
    # Element-wise MLP → Mean pooling → Post-processing
    # Problem: Information loss through mean pooling
```

### 3.2 Simplified Architecture (SimpleEmbedding)
```python
class SimpleEmbedding(nn.Module):
    # Direct mean pooling of (value, mask)
    # Problem: Still loses distributional information
```

### 3.3 Current Architecture (TabPFN-Style)
```python
class TabPFNStyleEmbedding(nn.Module):
    # 1. Input projection: (value, mask) → embed_dim
    # 2. Positional encoding for rows
    # 3. Transformer encoder over rows (2 layers, 4 heads)
    # 4. Cross-attention with learnable query
    # 5. Output normalization
    
    # Benefits:
    # - Captures full distribution
    # - Attention-based row importance
    # - Context-aware embeddings
```

### 3.4 Full Model Pipeline
```
Input: (batch, rows, cols) data + intervention masks
  ↓
TabPFNStyleEmbedding (per column)
  ↓
Column embeddings: (batch, cols, embed_dim=192)
  ↓
Transformer Encoder (6 layers, 8 heads)
  ↓
Bilinear Head: edge prediction
  ↓
Auxiliary Head: intervention prediction
  ↓
Output: (batch, cols, cols) adjacency logits
```

---

## 4. Training Experiments

### Experiment 1: Baseline
- **Config**: Default, no optimizations
- **Results**: AUROC 0.50, F1 0.0, Loss 0.30
- **Conclusion**: Mode collapse

### Experiment 2: Weighted Loss + Curriculum
- **Config**: pos_weight=2.33, p_linear=1.0
- **Results**: AUROC 0.54, F1 0.0, Loss 0.54
- **Conclusion**: Slight improvement, still stuck

### Experiment 3: Focal Loss + Larger Model
- **Config**: gamma=3.0, embed_dim=192, n_layers=6
- **Results**: AUROC 0.51, F1 0.0, Loss 0.007
- **Conclusion**: Better convergence, no edge prediction

### Experiment 4: TabPFN-Style (Debug)
- **Config**: Row-level Transformer, 5000 steps
- **Results**: AUROC 0.50, F1 0.0, Loss 0.0034
- **Conclusion**: Excellent loss, no improvement in metrics

### Experiment 5: Production-Scale (Current)
- **Config**: 100 SCMs, 50K steps, graphs up to 100 nodes
- **Status**: In progress (~80% complete)
- **Expected**: Final results pending completion

---

## 5. Current Production Run

### 5.1 Configuration Details

```yaml
# Data Generation
output_dir: "production_data"
n_runs: 100
  - Small: 20 runs (10-30 nodes)
  - Large: 80 runs (50-100 nodes)
p_linear: 0.5  # 50% linear, 50% non-linear
max_rows: 2000

# Model Architecture
max_cols: 128
embed_dim: 192
n_heads: 8
n_layers: 6
use_tabpfn_style: true

# Training
batch_size: 2
accumulation_steps: 16  # Effective batch = 32
lr: 0.001
total_steps: 50000
```

### 5.2 Training Progress
- **Started**: ~19 hours ago
- **Current Step**: ~40,000/50,000 (estimated)
- **Current Loss**: ~0.0037 (stable)
- **ETA**: ~2-3 hours remaining
- **Speed**: ~1.6s/iteration

### 5.3 Memory Optimization
- **VRAM Usage**: ~21-22 GB (fits in 24GB)
- **Batch Size**: 2 (reduced from 4 for TabPFN)
- **Gradient Accumulation**: 16 steps
- **Max Graph Size**: 100 nodes (reduced from 150)

---

## 6. Technical Details

### 6.1 Loss Function Evolution

**Initial**: Binary Cross-Entropy
```python
loss = BCEWithLogitsLoss()(logits, targets)
```

**Weighted BCE**:
```python
pos_weight = (1 - p_edge) / p_edge  # ≈ 2.33
loss = BCEWithLogitsLoss(pos_weight=pos_weight)(logits, targets)
```

**Focal Loss** (Current):
```python
class FocalLoss:
    def forward(self, logits, targets):
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        pt = torch.exp(-bce)
        focal = alpha * (1 - pt) ** gamma * bce
        return focal.mean()
```

### 6.2 Multi-Task Learning

**Primary Task**: Edge prediction
```python
edge_loss = compute_masked_loss(logits, y, pad_mask, focal_loss)
```

**Auxiliary Task**: Intervention prediction
```python
intervention_target = (m.sum(dim=1) > 0).float()
intervention_loss = F.binary_cross_entropy_with_logits(
    intervention_logits, intervention_target
)
```

**Combined Loss**:
```python
total_loss = edge_loss + 0.1 * intervention_loss
```

### 6.3 Data Generation

**Mechanisms**:
1. **Linear**: `Y = w₁X₁ + w₂X₂ + ... + ε`
2. **MLP**: `Y = MLP(X₁, X₂, ...) + ε`
3. **Sine**: `Y = sin(w₁X₁ + w₂X₂ + ...) + ε`
4. **Tanh**: `Y = tanh(w₁X₁ + w₂X₂ + ...) + ε`
5. **Quadratic**: `Y = (a₁X₁ + a₂X₂ + ...)² + ε`
6. **Threshold**: `Y = (w₁X₁ + w₂X₂ + ... > θ)`

**Interventions**:
- Do-calculus: `do(Xᵢ = value)`
- Random intervention targets
- Multiple interventions per graph

---

## 7. Challenges and Solutions

### Challenge 1: Mode Collapse
**Problem**: Model predicts no edges (all probabilities near 0)

**Root Cause**: Sparsity trap - graphs are 30% dense, model minimizes loss by predicting zeros

**Solutions Attempted**:
- ✅ Weighted loss (pos_weight=2.33)
- ✅ Focal loss (gamma=3.0)
- ✅ Lower prediction threshold (0.5 → 0.3)
- ❌ **Result**: No significant improvement

### Challenge 2: Information Loss
**Problem**: Simple mean pooling loses distributional information

**Root Cause**: SetEmbedding and SimpleEmbedding only capture first moment

**Solution**:
- ✅ TabPFN-style embeddings with row-level Transformer
- ✅ Attention mechanism captures full distribution
- ❌ **Result**: Better loss, but still no edge prediction

### Challenge 3: Memory Constraints
**Problem**: OOM errors with larger models and graphs

**Solutions**:
- ✅ Gradient accumulation (batch_size=2, accumulation=16)
- ✅ Reduced graph size (150 → 100 nodes)
- ✅ Reduced max_cols (256 → 128)
- ✅ Disabled AMP (caused NaN loss)
- ✅ **Result**: Fits in 24GB VRAM

### Challenge 4: Slow Convergence
**Problem**: Training takes many steps to converge

**Solutions**:
- ✅ Higher learning rate (1e-4 → 1e-3)
- ✅ More training steps (2K → 50K)
- ✅ Larger model (embed_dim=192, n_layers=6)
- ✅ **Result**: Faster, better convergence

---

## 8. Key Findings

### 8.1 What Works
1. **Loss Convergence**: Model can minimize loss effectively (0.003-0.007)
2. **Gradient Flow**: No vanishing/exploding gradients
3. **Memory Optimization**: Gradient accumulation enables large effective batch sizes
4. **TabPFN Architecture**: Richer embeddings, better loss convergence

### 8.2 What Doesn't Work
1. **Edge Prediction**: Model refuses to predict edges (F1 = 0.0 across all experiments)
2. **AUROC**: Stuck at ~0.50 (random guessing)
3. **Threshold Sensitivity**: Lowering threshold doesn't help
4. **Focal Loss**: More aggressive focusing doesn't break sparsity trap

### 8.3 Hypothesis
The model has learned to minimize loss by predicting very low probabilities for all edges, rather than learning to distinguish true edges from non-edges. This suggests:

1. **Fundamental Architecture Issue**: Current approach may not be suitable for causal discovery
2. **Insufficient Supervision**: Binary edge labels may not provide enough signal
3. **Optimization Landscape**: Local minimum is too attractive to escape

---

## 9. Production Run Expectations

### 9.1 Best Case Scenario
- **AUROC**: 0.55-0.60 (modest improvement)
- **F1**: 0.05-0.10 (some edges predicted)
- **Loss**: 0.002-0.004 (excellent convergence)

### 9.2 Likely Scenario
- **AUROC**: 0.50-0.52 (minimal improvement)
- **F1**: 0.0-0.02 (few/no edges predicted)
- **Loss**: 0.003-0.005 (good convergence)

### 9.3 What We'll Learn
- Whether scale (100 SCMs, 50K steps) helps
- Performance on larger graphs (up to 100 nodes)
- Generalization across mechanism types
- Whether TabPFN-style embeddings provide any benefit

---

## 10. Next Steps (Post-Production)

### 10.1 If Production Run Succeeds (AUROC > 0.6)
1. Scale up further (200+ SCMs, 100K steps)
2. Increase graph sizes (up to 200 nodes)
3. Add more mechanism types
4. Benchmark against classical methods (PC, GES, NOTEARS)

### 10.2 If Production Run Fails (AUROC ≈ 0.5)
1. **Try Alternative Architectures**:
   - Graph Neural Networks (GNN)
   - Variational Autoencoders (VAE)
   - NOTEARS-style differentiable DAG constraints

2. **Change Supervision Signal**:
   - Use edge scores instead of binary labels
   - Add graph-level objectives
   - Incorporate causal ordering information

3. **Benchmark Against Baselines**:
   - Compare with PC algorithm
   - Compare with GES
   - Validate data generation pipeline

---

## 11. Technical Specifications

### 11.1 Hardware
- **GPU**: 24GB VRAM (CUDA-enabled)
- **Training Time**: ~21 hours for 50K steps
- **Memory Usage**: ~21-22 GB VRAM

### 11.2 Software Stack
- **Python**: 3.12
- **PyTorch**: Latest (with CUDA support)
- **Dependencies**: `numpy`, `pandas`, `networkx`, `matplotlib`, `scikit-learn`, `tqdm`, `pyyaml`
- **Package Manager**: `uv`

### 11.3 Code Structure
```
TabSCM/
├── src/
│   ├── mechanism.py       # Causal mechanisms
│   ├── model.py           # TabPFN-style architecture
│   ├── fast_generator.py  # Online SCM generation
│   ├── dataset.py         # Data loading
│   ├── train.py           # Training loop (Focal Loss, multi-task)
│   ├── test.py            # Evaluation
│   ├── visualize.py       # Plotting
│   └── report.py          # Report generation
├── config.yaml            # Default config
├── debug_config.yaml      # Quick testing
├── production_config.yaml # Current production run
└── main.py                # Entry point
```

---

## 12. Conclusion

### 12.1 Progress Summary
Over the course of this project, we have:
- ✅ Built a complete, production-ready codebase
- ✅ Implemented comprehensive testing and reporting
- ✅ Tried 5+ different optimization strategies
- ✅ Developed TabPFN-style embeddings
- ✅ Launched production-scale training (100 SCMs, 50K steps)

### 12.2 Current Status
The production training is **80% complete** with:
- Excellent loss convergence (0.0037)
- Stable training dynamics
- No memory issues
- ETA: 2-3 hours to completion

### 12.3 Open Questions
1. Will scale help? (100 SCMs vs. 20 SCMs)
2. Will longer training help? (50K vs. 5K steps)
3. Can TabPFN-style embeddings learn causal structure?
4. Is the Transformer architecture fundamentally limited for this task?

### 12.4 Final Thoughts
Despite extensive optimization efforts, the model has consistently failed to predict edges. The production run represents our best attempt with:
- Maximum data diversity (100 SCMs, mixed mechanisms)
- Maximum training (50K steps)
- Best architecture (TabPFN-style)
- Optimal hyperparameters (Focal Loss, high LR, large model)

**The final results will determine whether this approach is viable or if we need to explore fundamentally different architectures.**

---

## Appendix A: All Experiment Results

| Experiment | Config | Steps | AUROC | F1 | Loss | Notes |
|------------|--------|-------|-------|----|----|-------|
| Baseline | Default | 2000 | 0.50 | 0.0 | 0.30 | Mode collapse |
| Weighted Loss | pos_weight=2.33 | 2000 | 0.54 | 0.0 | 0.54 | Slight improvement |
| Curriculum | p_linear=1.0 | 2000 | 0.54 | 0.0 | 0.54 | Linear only |
| Grad Accum | batch=8, accum=4 | 2000 | 0.50 | 0.0 | 0.15 | Better loss |
| Focal Loss | gamma=3.0 | 5000 | 0.51 | 0.0 | 0.007 | Excellent loss |
| TabPFN Debug | Row Transformer | 5000 | 0.50 | 0.0 | 0.0034 | Best loss yet |
| **Production** | **100 SCMs, 50K** | **~40K** | **TBD** | **TBD** | **0.0037** | **In progress** |

---

## Appendix B: Configuration Files

### Debug Config
```yaml
embed_dim: 192
n_heads: 8
n_layers: 6
batch_size: 2
accumulation_steps: 16
lr: 0.001
total_steps: 5000
```

### Production Config
```yaml
n_runs: 100
max_nodes_large: 100
embed_dim: 192
n_heads: 8
n_layers: 6
batch_size: 2
accumulation_steps: 16
lr: 0.001
total_steps: 50000
```

---

**Report Status**: Interim (Training in progress)
**Next Update**: Upon training completion with full test results
**Estimated Completion**: 2-3 hours from now
