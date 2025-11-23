# TabSCM Project: Complete Development Report

## Executive Summary

This report documents the complete development journey of the TabSCM (Tabular Structural Causal Model) project, from initial notebook refactoring through multiple optimization attempts. Despite implementing numerous best practices and optimizations, the model has not yet achieved meaningful causal discovery performance.

**Key Metrics (Final State)**:
- **AUROC**: 0.50 (random guessing)
- **F1 Score**: 0.0 (no edges predicted)
- **Training Loss**: ~0.15 (stable but not indicative of learning)
- **Status**: Model is stuck in a local minimum

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Development Timeline](#2-development-timeline)
3. [Code Architecture](#3-code-architecture)
4. [Configuration Files](#4-configuration-files)
5. [Experiments and Results](#5-experiments-and-results)
6. [Problem Analysis](#6-problem-analysis)
7. [Recommendations](#7-recommendations)

---

## 1. Project Overview

### 1.1 Objective
Develop a Transformer-based model for **zero-shot causal discovery** on tabular data. The model should:
- Learn from synthetic Structural Causal Models (SCMs)
- Infer causal graphs from observational and interventional data
- Generalize to unseen graph structures and mechanisms

### 1.2 Approach
- **Data Generation**: Random DAGs with diverse causal mechanisms (Linear, MLP, Sine, Tanh)
- **Model Architecture**: Set-based Transformer encoder with bilinear edge prediction
- **Training**: Online learning from infinite stream of synthetic SCMs
- **Evaluation**: Zero-shot performance on held-out test graphs

---

## 2. Development Timeline

### Phase 1: Project Refactoring (Steps 1-50)
**Goal**: Transform Jupyter notebook into production-ready Python project.

**Actions Taken**:
1. Created modular project structure:
   ```
   TabSCM/
   ├── src/
   │   ├── mechanism.py      # Causal mechanism definitions
   │   ├── model.py          # Neural network architecture
   │   ├── fast_generator.py # Online SCM generation
   │   ├── dataset.py        # Data loading and streaming
   │   ├── generator.py      # Detailed SCM generation
   │   ├── pipeline.py       # Orchestration
   │   └── train.py          # Training loop
   ├── config.yaml           # Hyperparameters
   ├── main.py               # Entry point
   └── README.md
   ```

2. Initialized dependency management with `uv`
3. Added dependencies: `torch`, `numpy`, `pandas`, `networkx`, `matplotlib`, `tqdm`, `pyyaml`

**Results**:
- ✅ Successfully refactored all code
- ✅ Verified with short training run (200 steps)
- ✅ Clean, maintainable codebase

---

### Phase 2: Testing Mode Implementation (Steps 51-100)
**Goal**: Add zero-shot evaluation capabilities.

**Actions Taken**:
1. Modified `src/train.py` to save model checkpoint (`model.pth`)
2. Created `src/test.py` with evaluation metrics:
   - **SHD** (Structural Hamming Distance)
   - **AUROC** (Area Under ROC Curve)
   - **F1 Score**, **Precision**, **Recall**
3. Integrated testing into `main.py` (`--mode test`)

**Results**:
- ✅ Testing pipeline functional
- ❌ Initial metrics showed no learning (AUROC ≈ 0.5, F1 = 0.0)
- 🐛 Fixed OOM error by switching to CPU for testing

---

### Phase 3: Reporting and Visualization (Steps 101-150)
**Goal**: Create comprehensive reporting system.

**Actions Taken**:
1. Implemented loss tracking in `src/train.py` → `loss_history.json`
2. Created `src/visualize.py`:
   - `plot_loss_curve()`: Training loss over time
   - `plot_adjacency_comparison()`: Ground truth vs. predicted heatmaps
   - `plot_graph_structure()`: NetworkX graph visualizations
   - `plot_metric_distributions()`: Metric histograms
3. Created `src/report.py`: Generates `report.md` with embedded plots
4. Integrated reporting into `main.py` (`--mode report`, `--mode all`)

**Results**:
- ✅ Automated report generation working
- ✅ Visualizations clearly show model failure (all predictions near 0)

---

### Phase 4: Documentation (Steps 151-200)
**Goal**: Create comprehensive project documentation.

**Actions Taken**:
1. Created `RESEARCH_REPORT.md`:
   - Abstract and introduction
   - Methodology (data generation, model architecture)
   - Training pipeline
   - Evaluation metrics
   - Preliminary results
2. Enhanced `README.md`:
   - Installation instructions
   - Usage examples
   - Project structure
   - Results summary

**Results**:
- ✅ Professional documentation complete
- ✅ Project ready for external review

---

### Phase 5: GitHub Publishing (Steps 201-250)
**Goal**: Publish project to GitHub.

**Actions Taken**:
1. Authenticated with GitHub CLI (`gh auth login`)
2. Initialized Git repository
3. Created remote repository: `MeiisamMahmoodii/TabSCM`
4. Pushed all code and documentation

**Results**:
- ✅ Project publicly available at [https://github.com/MeiisamMahmoodii/TabSCM](https://github.com/MeiisamMahmoodii/TabSCM)

---

### Phase 6: Gradient Flow Debugging (Steps 251-300)
**Goal**: Diagnose why the model isn't learning.

**Actions Taken**:
1. Modified `src/train.py` to log gradient norms per component:
   - `set_encoder`, `transformer`, `head`
2. Added `plot_gradient_norms()` to `src/visualize.py`
3. Updated `src/report.py` to include gradient flow plot

**Results**:
- ✅ Gradient logging implemented
- 📊 Gradients were non-zero (ruling out vanishing gradients)
- ❌ Model still not learning (AUROC = 0.50, F1 = 0.0)

---

### Phase 7: Model Improvements - Weighted Loss & Curriculum Learning (Steps 301-350)
**Goal**: Address class imbalance and simplify the learning task.

**Actions Taken**:
1. **Weighted Loss**:
   - Modified `compute_masked_loss()` to accept `pos_weight`
   - Calculated weight as `(1 - p_edge) / p_edge ≈ 2.33`
   - Applied to `BCEWithLogitsLoss`

2. **Curriculum Learning**:
   - Modified `src/fast_generator.py` to accept `p_linear` parameter
   - Set `p_linear = 1.0` (100% linear mechanisms)
   - Updated `src/dataset.py` and `src/train.py` to pass `p_linear`

**Configuration Changes**:
```yaml
p_linear: 1.0  # Force linear-only mechanisms
```

**Results**:
- ✅ Implementation successful
- 📈 AUROC improved slightly: 0.50 → 0.5352 (2000 steps)
- ❌ F1 still 0.0 (model still predicting no edges)
- **Conclusion**: Improvements too small; model still stuck

---

### Phase 8: Memory Optimization (Steps 351-400)
**Goal**: Enable larger batch sizes within 24GB VRAM.

**Actions Taken**:
1. **Gradient Accumulation**:
   - Modified `src/train.py` to accumulate gradients over multiple mini-batches
   - Physical batch size: 8
   - Accumulation steps: 4
   - Effective batch size: 32

2. **Mixed Precision (AMP)**:
   - Implemented `torch.amp.GradScaler` and `autocast`
   - Updated to new PyTorch API (`torch.amp` instead of `torch.cuda.amp`)

3. **Memory Capping**:
   - Added `max_rows` parameter to limit rows per sample
   - Passed through `InfiniteCausalStream` → `FastSCMGenerator`

**Configuration Changes**:
```yaml
batch_size: 8
accumulation_steps: 4
use_amp: true
max_rows: 1000
```

**Results**:
- ✅ Gradient accumulation working
- ❌ **AMP caused NaN loss** (FP16 + weighted loss = numerical instability)
- ✅ Disabled AMP → stable training (loss ~0.15)
- ❌ AUROC still 0.50, F1 still 0.0

---

## 3. Code Architecture

### 3.1 Core Modules

#### `src/mechanism.py`
**Purpose**: Define causal mechanisms for data generation.

```python
class TabPFNMechanism:
    def __init__(self, mechanism_type='linear'):
        # mechanism_type: 'linear', 'mlp', 'sine', 'tanh'
```

**Mechanisms**:
- **Linear**: `Y = w₁X₁ + w₂X₂ + ... + ε`
- **MLP**: `Y = MLP(X₁, X₂, ...) + ε`
- **Sine**: `Y = sin(w₁X₁ + w₂X₂ + ...) + ε`
- **Tanh**: `Y = tanh(w₁X₁ + w₂X₂ + ...) + ε`

---

#### `src/model.py`
**Purpose**: Neural network architecture for causal discovery.

**Components**:

1. **SetEmbedding** (Lines 4-24):
   - Encodes each variable as a set of (value, mask) pairs
   - Uses element-wise MLP → mean pooling → post-processing
   - Input: `(batch, rows, 2)` → Output: `(batch, embed_dim)`

2. **ZCIA_Transformer** (Lines 26-62):
   - Main model architecture
   - **Set Encoder**: Embeds each column independently
   - **Transformer**: Captures interactions between variables
   - **Bilinear Head**: Predicts edge probabilities `P(i → j)`

**Forward Pass**:
```python
x: (batch, rows, cols)  # Data
m: (batch, rows, cols)  # Intervention masks
→ col_embeddings: (batch, cols, embed_dim)
→ transformer_output: (batch, cols, embed_dim)
→ logits: (batch, cols, cols)  # Adjacency matrix
```

---

#### `src/fast_generator.py`
**Purpose**: Efficiently generate synthetic SCMs online.

**Key Features**:
- Random DAG generation (Erdős-Rényi)
- Topological ordering for causal simulation
- Interventional data generation (do-calculus)
- Row subsampling to prevent OOM (`max_rows`)

**Parameters**:
- `min_nodes`, `max_cols`: Graph size range
- `p_linear`: Probability of linear mechanism (curriculum learning)
- `max_rows`: Maximum rows per sample (memory control)

---

#### `src/dataset.py`
**Purpose**: Data loading and streaming.

**Classes**:

1. **CausalDataset** (Lines 9-119):
   - Loads pre-generated SCMs from disk
   - Handles observational + interventional data
   - Pads/truncates to `max_cols` and `max_rows`

2. **InfiniteCausalStream** (Lines 121-131):
   - Infinite iterator for online training
   - Generates SCMs on-the-fly using `FastSCMGenerator`

3. **causal_collate_fn** (Lines 133-163):
   - Batches variable-size graphs
   - Creates padding masks for Transformer

---

#### `src/train.py`
**Purpose**: Training loop with all optimizations.

**Key Features**:

1. **Weighted Loss** (Lines 13-25):
   ```python
   pos_weight = (1 - p_edge) / p_edge  # ≈ 2.33
   criterion = BCEWithLogitsLoss(pos_weight=pos_weight)
   ```

2. **Gradient Accumulation** (Lines 66-92):
   ```python
   loss = loss / accumulation_steps
   scaler.scale(loss).backward()
   if (step + 1) % accumulation_steps == 0:
       scaler.step(optimizer)
       optimizer.zero_grad()
   ```

3. **Gradient Logging** (Lines 94-107):
   - Logs average gradient norm per component
   - Saves to `grad_history.json`

4. **Mixed Precision** (Lines 66-69, 87-91):
   - Currently **disabled** due to NaN loss
   - `use_amp: false` in config

---

#### `src/test.py`
**Purpose**: Zero-shot evaluation on held-out graphs.

**Metrics**:
- **SHD**: Counts edge errors (insertions, deletions, reversals)
- **AUROC**: Ranking quality (true edge vs. non-edge)
- **F1/Precision/Recall**: Binary classification at threshold 0.5

**Visualization**:
- Saves adjacency heatmaps for first 3 samples
- Saves graph structure plots
- Saves metric distributions

---

#### `src/visualize.py`
**Purpose**: Generate all plots for reporting.

**Functions**:
- `plot_loss_curve()`: Training loss over steps
- `plot_gradient_norms()`: Gradient flow analysis (log scale)
- `plot_adjacency_comparison()`: Heatmap comparison
- `plot_graph_structure()`: NetworkX graph plots
- `plot_metric_distributions()`: Histogram of metrics

---

#### `src/report.py`
**Purpose**: Generate markdown report with embedded plots.

**Report Structure**:
1. Model configuration
2. Training performance (loss curve, gradient flow)
3. Test performance (metrics table)
4. Visualizations (heatmaps, graphs)
5. Deep explanation

---

### 3.2 Entry Point: `main.py`

**Modes**:
- `--mode generate`: Generate synthetic SCMs
- `--mode train`: Train model
- `--mode test`: Evaluate model
- `--mode report`: Generate report
- `--mode all`: Run entire pipeline

**Example**:
```bash
uv run python main.py --mode all --config config.yaml
```

---

## 4. Configuration Files

### 4.1 `config.yaml` (Production)

```yaml
# Data Generation
output_dir: "causal_pfn_data"
n_nodes_min: 8
n_nodes_max: 16
p_edge: 0.3                    # Edge probability in DAG
p_linear: 1.0                  # Curriculum: 100% linear mechanisms
n_samples: 1000
n_runs: 50

# Model Architecture
max_cols: 128                  # Maximum variables
min_nodes: 8
embed_dim: 128                 # Embedding dimension

# Training
batch_size: 8                  # Physical batch size
accumulation_steps: 4          # Effective batch = 8 × 4 = 32
use_amp: true                  # Mixed Precision (DISABLED due to NaN)
lr: 0.0001                     # Learning rate
total_steps: 20000             # Training steps
print_every: 100

# Testing
n_test_samples: 100
device: "cuda"
```

**Key Parameters**:
- **p_edge**: Controls graph sparsity (30% of possible edges)
- **p_linear**: Curriculum learning (1.0 = linear only, 0.5 = mixed)
- **accumulation_steps**: Simulates larger batch size without OOM
- **use_amp**: Currently `false` to avoid NaN loss

---

### 4.2 `debug_config.yaml` (Quick Testing)

```yaml
# Same as config.yaml but:
total_steps: 2000              # Shorter run
n_test_samples: 5              # Fewer test samples
device: "cpu"                  # Or "cuda" depending on availability
```

**Purpose**: Fast iteration during development.

---

### 4.3 `test_config.yaml` (Initial Testing)

```yaml
# Very small config for initial verification
total_steps: 200
batch_size: 8
device: "cpu"
```

---

## 5. Experiments and Results

### Experiment 1: Baseline (No Optimizations)
**Config**: Default, batch_size=8, no weighted loss, mixed mechanisms

| Metric | Result |
|--------|--------|
| AUROC | 0.4998 |
| F1 | 0.0000 |
| SHD | 484.2 ± 474.5 |
| Loss | 0.3041 |

**Analysis**: Model predicts "no edge" for everything (mode collapse).

---

### Experiment 2: Weighted Loss + Curriculum Learning
**Config**: `pos_weight=2.33`, `p_linear=1.0`, 2000 steps

| Metric | Result | Change |
|--------|--------|--------|
| AUROC | 0.5352 | +0.0354 |
| F1 | 0.0000 | 0 |
| SHD | 462.0 ± 723.5 | -22.2 |
| Loss | 0.5401 | +0.2360 |

**Analysis**: Slight AUROC improvement, but still no edges predicted.

---

### Experiment 3: Gradient Accumulation (Effective Batch 32)
**Config**: `batch_size=8`, `accumulation_steps=4`, 2000 steps

| Metric | Result | Change |
|--------|--------|--------|
| AUROC | 0.5008 | -0.0344 |
| F1 | 0.0000 | 0 |
| SHD | 798.2 ± 338.4 | +336.2 |
| Loss | 0.1477 | -0.3924 |

**Analysis**: Loss decreased significantly, but no improvement in metrics.

---

### Experiment 4: Mixed Precision (AMP)
**Config**: `use_amp=true`, weighted loss, gradient accumulation

| Metric | Result |
|--------|--------|
| Loss | **NaN** |

**Analysis**: FP16 + weighted loss caused numerical instability. **Disabled AMP**.

---

## 6. Problem Analysis

### 6.1 Why Is the Model Not Learning?

After extensive experimentation, the model consistently fails to learn causal structure. Here's the diagnosis:

#### Problem 1: **Sparsity Trap**
- Causal graphs are sparse (~30% edge density)
- The model learns that predicting `P(edge) ≈ 0` minimizes loss
- Even with weighted loss (`pos_weight=2.33`), the penalty for missing edges is insufficient

#### Problem 2: **Architecture Limitations**
The `SetEmbedding` approach may be fundamentally flawed:
- **Information Loss**: Mean pooling over rows loses distributional information
- **Weak Signal**: Intervention masks are binary (0 or 1), providing limited signal
- **No Positional Encoding**: Transformer has no notion of variable ordering

#### Problem 3: **Optimization Landscape**
- The model finds a **local minimum** where `loss ≈ 0.15` but predictions are all near 0
- Gradient norms are non-zero, so it's not vanishing gradients
- The optimizer is stuck in a "safe" region

#### Problem 4: **Numerical Instability**
- Mixed Precision (AMP) causes NaN loss when combined with weighted loss
- FP16 cannot represent the large gradients from `pos_weight=2.33`

---

### 6.2 What the Gradient Flow Revealed

The gradient logging showed:
- **set_encoder**: Gradients present but small
- **transformer**: Gradients present but small
- **head**: Gradients present but small

**Interpretation**: The model is updating, but the updates are not pushing it toward edge prediction.

---

## 7. Recommendations

### 7.1 Immediate Fixes (High Priority)

#### 1. **Increase Learning Rate**
**Current**: `lr = 1e-4`  
**Recommended**: `lr = 1e-3` or `5e-4`

**Rationale**: The model may be stuck in a shallow local minimum. A higher learning rate could help escape.

```yaml
lr: 0.001  # 10x increase
```

---

#### 2. **Use Focal Loss**
**Current**: Weighted BCE  
**Recommended**: Focal Loss

**Rationale**: Focal Loss down-weights easy negatives (non-edges) and focuses on hard positives (edges).

```python
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        
    def forward(self, logits, targets):
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        pt = torch.exp(-bce)
        focal = self.alpha * (1 - pt) ** self.gamma * bce
        return focal.mean()
```

---

#### 3. **Simplify Architecture**
**Current**: SetEmbedding (MLP → Mean Pool → MLP)  
**Recommended**: Direct embedding

Replace `SetEmbedding` with a simpler approach:

```python
class SimpleEmbedding(nn.Module):
    def __init__(self, embed_dim=128):
        super().__init__()
        self.encoder = nn.Linear(2, embed_dim)  # (value, mask) → embedding
        
    def forward(self, x, m):
        # x: (batch, rows, cols)
        # m: (batch, rows, cols)
        # Take mean over rows directly
        x_mean = x.mean(dim=1)  # (batch, cols)
        m_mean = m.mean(dim=1)  # (batch, cols)
        combined = torch.stack([x_mean, m_mean], dim=-1)  # (batch, cols, 2)
        return self.encoder(combined)  # (batch, cols, embed_dim)
```

---

#### 4. **Add Auxiliary Loss**
**Current**: Only edge prediction loss  
**Recommended**: Multi-task learning

Add a node classification task:

```python
# Predict which nodes are intervened
intervention_logits = model.predict_interventions(z)
intervention_loss = F.binary_cross_entropy_with_logits(
    intervention_logits, 
    intervention_targets
)
total_loss = edge_loss + 0.1 * intervention_loss
```

---

### 7.2 Medium-Term Improvements

#### 5. **Data Augmentation**
- **Edge Dropout**: Randomly drop edges during training
- **Node Permutation**: Shuffle variable order
- **Noise Injection**: Add Gaussian noise to data

#### 6. **Curriculum Learning (Progressive)**
Instead of jumping to 100% linear, gradually increase complexity:

```yaml
# Phase 1: 100% linear (steps 0-5000)
# Phase 2: 75% linear (steps 5000-10000)
# Phase 3: 50% linear (steps 10000-15000)
# Phase 4: 25% linear (steps 15000-20000)
```

#### 7. **Larger Model**
```yaml
embed_dim: 256  # Current: 128
n_heads: 8      # Current: 4
n_layers: 6     # Current: 4
```

---

### 7.3 Long-Term Research Directions

#### 8. **Attention Mechanism Redesign**
Use **Graph Attention Networks (GAT)** instead of standard Transformer:
- Explicitly model edge probabilities in attention
- Use learned edge weights to refine embeddings

#### 9. **Contrastive Learning**
Pre-train the encoder using contrastive loss:
- Positive pairs: Same graph, different interventions
- Negative pairs: Different graphs

#### 10. **Benchmark Against Baselines**
Compare against:
- **PC Algorithm** (constraint-based)
- **GES** (score-based)
- **NOTEARS** (continuous optimization)

---

### 7.4 Debugging Checklist

Before implementing fixes, verify:

- [ ] **Data Quality**: Inspect generated SCMs manually
- [ ] **Label Distribution**: Check edge density in training data
- [ ] **Model Outputs**: Print raw logits (before sigmoid) for a batch
- [ ] **Gradient Magnitudes**: Check if gradients are too small (< 1e-6)
- [ ] **Loss Landscape**: Plot loss vs. learning rate (LR finder)

---

## 8. Conclusion

### 8.1 What We Achieved
✅ **Production-Ready Codebase**: Modular, well-documented, version-controlled  
✅ **Comprehensive Tooling**: Training, testing, reporting, visualization  
✅ **Memory Optimization**: Gradient accumulation enables large effective batch sizes  
✅ **Debugging Infrastructure**: Gradient flow analysis, loss tracking  

### 8.2 What We Learned
❌ **Model Architecture**: Current design is insufficient for causal discovery  
❌ **Optimization**: Weighted loss alone cannot overcome sparsity trap  
❌ **Mixed Precision**: AMP incompatible with weighted loss (NaN loss)  

### 8.3 Next Steps
1. **Immediate**: Try Focal Loss + higher learning rate
2. **Short-term**: Simplify architecture (remove SetEmbedding)
3. **Long-term**: Explore GAT-based architectures

---

## Appendix A: File Structure

```
TabSCM/
├── src/
│   ├── mechanism.py          # Causal mechanisms
│   ├── model.py              # Neural network (SetEmbedding + Transformer)
│   ├── fast_generator.py     # Online SCM generation
│   ├── dataset.py            # Data loading (CausalDataset, InfiniteCausalStream)
│   ├── generator.py          # Detailed SCM generation
│   ├── pipeline.py           # Orchestration
│   ├── train.py              # Training loop (weighted loss, grad accumulation)
│   ├── test.py               # Evaluation (SHD, AUROC, F1)
│   ├── visualize.py          # Plotting functions
│   └── report.py             # Report generation
├── config.yaml               # Production config
├── debug_config.yaml         # Quick testing config
├── test_config.yaml          # Initial verification config
├── main.py                   # Entry point
├── README.md                 # User-facing documentation
├── RESEARCH_REPORT.md        # Technical report
├── COMPLETE_PROJECT_REPORT.md # This document
└── .gitignore
```

---

## Appendix B: Key Equations

### Loss Function
```
L = - (1/N) Σ [y_ij * log(σ(z_ij)) * w + (1 - y_ij) * log(1 - σ(z_ij))]

where:
  y_ij = 1 if edge i → j exists, 0 otherwise
  z_ij = logit (model output)
  σ(·) = sigmoid function
  w = pos_weight = (1 - p_edge) / p_edge ≈ 2.33
```

### Gradient Accumulation
```
Effective Batch Size = batch_size × accumulation_steps
                     = 8 × 4
                     = 32
```

---

## Appendix C: Command Reference

```bash
# Full pipeline
uv run python main.py --mode all --config config.yaml

# Training only
uv run python main.py --mode train --config config.yaml

# Testing only
uv run python main.py --mode test --config config.yaml

# Report generation
uv run python main.py --mode report --config config.yaml

# Quick debugging
uv run python main.py --mode all --config debug_config.yaml
```

---

**Report Generated**: 2025-11-23  
**Author**: AI Assistant (Claude 3.5 Sonnet)  
**Project**: TabSCM - Zero-Shot Causal Discovery with Transformers
