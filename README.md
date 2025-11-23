# TabSCM: Zero-Shot Causal Inference with Transformers

**TabSCM** is a research project designed to perform **zero-shot causal discovery** on tabular data. By training a Transformer-based model on an infinite stream of synthetic Structural Causal Models (SCMs), this project aims to learn a generalized inference engine capable of predicting causal graphs for unseen datasets without the need for retraining or optimization per dataset.

For a deep dive into the methodology, architecture, and research results, please refer to the [**Research Report**](RESEARCH_REPORT.md).

---

## 🚀 Key Features

- **Zero-Shot Inference**: Predicts causal graphs for new datasets in a single forward pass.
- **Infinite Synthetic Data**: Generates unique random SCMs on-the-fly during training to prevent overfitting.
- **Diverse Mechanisms**: Supports Linear, MLP, Sine, Quadratic, Tanh, Threshold, and Gaussian causal mechanisms.
- **Intervention Simulation**: Simulates Do-calculus interventions to help the model distinguish correlation from causation.
- **Transformer Architecture**: Uses a Set-Transformer design to handle variable numbers of features and samples permutation-invariantly.
- **Comprehensive Reporting**: Built-in tools for generating loss curves, adjacency heatmaps, and performance metrics.

## 🛠️ Installation

1.  **Prerequisites**: Ensure you have Python 3.10+ and `uv` installed.
    ```bash
    curl -LsSf https://astral.sh/uv/install.sh | sh
    ```

2.  **Clone the Repository**:
    ```bash
    git clone https://github.com/MeiisamMahmoodii/TabSCM.git
    cd TabSCM
    ```

3.  **Install Dependencies**:
    ```bash
    uv sync
    ```

## 💻 Usage

The project is controlled via `main.py` and supports several modes of operation.

### Configuration
All hyperparameters (model size, training steps, data generation settings) are defined in `config.yaml`. You can create multiple config files (e.g., `test_config.yaml`) and pass them via the `--config` argument.

### Modes

1.  **Generate Data Only**:
    Generates synthetic SCMs (observational and interventional data) and saves them to the output directory.
    ```bash
    uv run python main.py --mode generate
    ```

2.  **Train Model**:
    Trains the ZCIA Transformer model using the online generator.
    ```bash
    uv run python main.py --mode train
    ```

3.  **Test Model (Zero-Shot)**:
    Evaluates the trained model on a batch of unseen synthetic SCMs and computes metrics (SHD, AUROC, F1).
    ```bash
    uv run python main.py --mode test
    ```

4.  **Generate Report**:
    Creates a markdown report with visualizations of loss, metrics, and predicted graphs.
    ```bash
    uv run python main.py --mode report
    ```

5.  **Run All**:
    Executes the full pipeline: Generate -> Train -> Test -> Report.
    ```bash
    uv run python main.py --mode all
    ```

## 📂 Project Structure

```
TabSCM/
├── src/
│   ├── generator.py       # Causal SCM Generator (Graph & Data)
│   ├── fast_generator.py  # Optimized Generator for online training
│   ├── mechanism.py       # Definitions of causal mechanisms (Linear, MLP, etc.)
│   ├── model.py           # ZCIA Transformer architecture
│   ├── dataset.py         # InfiniteCausalStream and DataLoaders
│   ├── pipeline.py        # Orchestration of data generation
│   ├── train.py           # Training loop with masked loss
│   ├── test.py            # Evaluation logic and metric computation
│   ├── visualize.py       # Plotting functions (Heatmaps, Graphs, Curves)
│   └── report.py          # Markdown report generation
├── config.yaml            # Main configuration file
├── main.py                # Entry point CLI
├── RESEARCH_REPORT.md     # Detailed project documentation
└── README.md              # This file
```

## 🧠 How It Works

1.  **Data Generation**: The `FastSCMGenerator` creates a random DAG and samples observational data. It then performs interventions on random nodes and samples interventional data.
2.  **Input Processing**: The model takes the stacked observational and interventional data (along with intervention masks) as input.
3.  **Set Encoding**: Each variable's samples are encoded into a single embedding vector, ensuring the model handles sets of samples correctly.
4.  **Transformer Processing**: A Transformer Encoder processes the variable embeddings to learn relationships (dependencies) between them.
5.  **Edge Prediction**: A Bilinear head predicts the probability of a directed edge between every pair of variables.

## 📊 Results Summary

Preliminary testing on small-scale synthetic data shows the model is functional but requires extensive training to achieve high accuracy.

| Metric | Description | Goal |
| :--- | :--- | :--- |
| **SHD** | Structural Hamming Distance | Minimize |
| **AUROC** | Area Under ROC Curve | Maximize (target > 0.9) |
| **F1** | Harmonic mean of Precision/Recall | Maximize |

*For detailed results and visualizations, run the reporting mode.*

## 📄 License

This project is open-source.
