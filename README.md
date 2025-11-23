# TabSCM: Causal SCM Generation and Training

This project generates synthetic data based on Structural Causal Models (SCM) and trains a Transformer model to infer causal structures.

For a comprehensive overview of the project, methodology, and results, please see [RESEARCH_REPORT.md](RESEARCH_REPORT.md).

## Setup

1. Install `uv` if not already installed.
2. Install dependencies:
   ```bash
   uv sync
   ```

## Usage

### Configuration
Edit `config.yaml` to adjust hyperparameters.

### Run Everything
To generate data and train the model:
```bash
uv run python main.py --mode all
```

### Generate Data Only
```bash
uv run python main.py --mode generate
```

### Train Model Only
```bash
uv run python main.py --mode train
```

## Project Structure
- `src/`: Source code modules.
  - `generator.py`: Causal SCM Generator.
  - `fast_generator.py`: Optimized Generator for training.
  - `mechanism.py`: Mechanism definitions.
  - `model.py`: ZCIA Transformer model.
  - `dataset.py`: Dataset and DataLoader.
  - `pipeline.py`: Data generation pipeline.
  - `train.py`: Training loop.
- `config.yaml`: Configuration file.
- `main.py`: Entry point.
