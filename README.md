# ADMET Prediction Pipeline for OpenADMET-ExpansionRx Challenge

A comprehensive machine learning pipeline for predicting ADMET (Absorption, Distribution, Metabolism, Excretion, and Toxicity) properties of drug molecules. This pipeline is designed for the **OpenADMET-ExpansionRx Blind Challenge** hosted on HuggingFace.

## Overview

This pipeline predicts 9 crucial ADMET endpoints using ensemble machine learning models trained on molecular fingerprints and descriptors. The implementation focuses on robustness, reproducibility, and ease of use.

### Predicted Endpoints

1. **LogD** - Lipophilicity at specific pH
2. **KSol** - Kinetic Solubility (µM)
3. **HLM CLint** - Human Liver Microsomal stability (mL/min/kg)
4. **MLM CLint** - Mouse Liver Microsomal stability (mL/min/kg)
5. **Caco-2 Papp A>B** - Intestinal permeability (10⁻⁶ cm/s)
6. **Caco-2 Efflux Ratio** - Active vs passive transport
7. **MPPB** - Mouse Plasma Protein Binding (% Unbound)
8. **MBPB** - Mouse Brain Protein Binding (% Unbound)
9. **MGMB** - Mouse Gastrocnemius Muscle Binding (% Unbound)

## Features

### Molecular Featurization
- **Morgan Fingerprints** (ECFP4, 2048 bits) - Circular fingerprints capturing local molecular structure
- **RDKit Descriptors** (40+ features) - Physicochemical properties including:
  - Molecular weight, LogP, TPSA
  - H-bond donors/acceptors
  - Rotatable bonds, aromatic rings
  - Topological descriptors (Chi, Kappa)
  - VSA descriptors (PEOE, SMR, SlogP, EState)

### Machine Learning Models
- **XGBoost** - Gradient boosting with tree-based learners (primary model)
- **LightGBM** - Fast gradient boosting framework (secondary model)
- **Random Forest** - Ensemble of decision trees (tertiary model)
- **Ensemble Strategy** - Weighted averaging based on cross-validation performance

### Key Capabilities
- ✅ Handles missing values per endpoint
- ✅ 5-fold cross-validation for robust evaluation
- ✅ Automatic feature scaling and variance filtering
- ✅ Per-endpoint model training
- ✅ Ensemble predictions for improved accuracy
- ✅ Easy-to-use command-line interface
- ✅ Comprehensive logging and monitoring

## Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager
- 8GB+ RAM recommended
- (Optional) GPU for faster training

### Setup

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/admet-prediction-pipeline.git
cd admet-prediction-pipeline
```

2. **Create virtual environment** (recommended)
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

### Installing RDKit

RDKit is required for molecular featurization. Install via conda (recommended):

```bash
conda install -c conda-forge rdkit
```

Or via pip:
```bash
pip install rdkit
```

## Quick Start

### 1. Download Data

Download training and test datasets from HuggingFace:

```bash
python scripts/download_data.py
```

This will:
- Download datasets from HuggingFace
- Validate data integrity
- Save locally to `data/raw/`
- Display data summary

### 2. Train Models

Train models for all 9 ADMET endpoints:

```bash
python scripts/train_all.py
```

This will:
- Generate molecular features from SMILES
- Train XGBoost, LightGBM, and Random Forest models
- Perform 5-fold cross-validation
- Save trained models to `models/saved_models/`
- Display CV scores for each endpoint

**Expected training time**: 30-60 minutes (depending on hardware)

### 3. Generate Predictions

Create submission file for the challenge:

```bash
python scripts/generate_submission.py
```

This will:
- Load trained models
- Generate predictions for test set
- Create `submissions/submission.csv`
- Display sample predictions

### 4. Submit to Challenge

Upload the generated `submissions/submission.csv` file to the [HuggingFace challenge space](https://huggingface.co/spaces/openadmet/OpenADMET-ExpansionRx-Challenge).

## Project Structure

```
admet-prediction-pipeline/
├── README.md                  # This file
├── requirements.txt           # Python dependencies
├── config/
│   └── config.yaml           # Configuration parameters
├── data/
│   ├── raw/                  # Downloaded datasets
│   └── processed/            # Processed features (optional)
├── src/
│   ├── __init__.py
│   ├── utils.py              # Helper functions
│   ├── data_loader.py        # Data loading utilities
│   ├── feature_engineering.py # Molecular featurization
│   └── models.py             # Model training and prediction
├── scripts/
│   ├── download_data.py      # Download datasets
│   ├── train_all.py          # Train all models
│   └── generate_submission.py # Generate submission file
├── models/
│   └── saved_models/         # Trained models
├── submissions/              # Generated submission files
└── notebooks/                # Jupyter notebooks (optional)
```

## Configuration

Edit `config/config.yaml` to customize:

### Feature Engineering
```yaml
features:
  morgan:
    radius: 2          # ECFP4 = radius 2
    nBits: 2048        # Fingerprint size
  use_rdkit_descriptors: true
  remove_low_variance: true
  variance_threshold: 0.01
```

### Model Hyperparameters
```yaml
models:
  xgboost:
    n_estimators: 1000
    learning_rate: 0.05
    max_depth: 8
    subsample: 0.8
    colsample_bytree: 0.8
```

### Training Settings
```yaml
training:
  cv_folds: 5
  random_seed: 42
  ensemble_weights:
    xgboost: 0.5
    lightgbm: 0.3
    random_forest: 0.2
```

## Advanced Usage

### Custom Feature Engineering

Modify `src/feature_engineering.py` to add custom features:

```python
def calculate_custom_descriptors(self, mol):
    """Add your custom molecular descriptors"""
    descriptors = {}
    # Add custom calculations
    return descriptors
```

### Hyperparameter Optimization

Enable Optuna-based hyperparameter tuning in `config.yaml`:

```yaml
training:
  optimize_hyperparameters: true
  n_trials: 50
```

### Using Individual Models

To use only a specific model (e.g., XGBoost):

```python
predictions = predictor.predict(X_test, use_ensemble=False)
```

### Training Specific Endpoints

To train models for specific endpoints only:

```python
from src.models import ADMETModel

model = ADMETModel('LogD', config)
model.train(X_train, y_train)
predictions = model.predict(X_test)
```

## Performance Expectations

Based on cross-validation results and baseline comparisons:

- **XGBoost**: Typically achieves the best single-model performance
- **Ensemble**: 5-10% improvement over single models
- **Expected MAE**: Varies by endpoint (see training logs)
- **Target**: Top 10-20% on challenge leaderboard

### Tips for Better Performance

1. **Feature Engineering**: Experiment with additional molecular descriptors
2. **Hyperparameter Tuning**: Use Optuna for systematic optimization
3. **External Data**: Incorporate public ADMET datasets (allowed by rules)
4. **Deep Learning**: Consider adding graph neural networks (GNNs)
5. **Ensemble Diversity**: Add more diverse model types

## Troubleshooting

### Common Issues

**Issue**: RDKit import error
```
Solution: Install RDKit via conda: conda install -c conda-forge rdkit
```

**Issue**: Out of memory during training
```
Solution: Reduce n_estimators or use smaller feature set
```

**Issue**: Missing data files
```
Solution: Run python scripts/download_data.py first
```

**Issue**: Slow training
```
Solution: Reduce cv_folds or n_estimators in config.yaml
```

### Getting Help

- Check the [challenge Discord](https://discord.gg/openadmet) for community support
- Review [challenge documentation](https://openadmet.ghost.io/openadmet-expansionrx-blind-challenge/)
- Open an issue on GitHub

## Challenge Information

- **Challenge Name**: OpenADMET-ExpansionRx Blind Challenge
- **Host**: OpenADMET & ExpansionRx
- **Platform**: HuggingFace Spaces
- **Submission Deadline**: January 19, 2026
- **Intermediate Leaderboard**: December 1, 2025
- **Winners Announced**: January 26, 2026

### Evaluation Metrics

- **Per-endpoint**: Mean Absolute Error (MAE)
- **Overall**: Macro-Averaged Relative Absolute Error (MA-RAE)
- **Statistical Testing**: Bootstrapping with significance testing

### Submission Rules

- Maximum 1 submission per day
- Only latest submission counts
- Code/methodology report required
- External data allowed
- Anonymous participation allowed

## References

### Challenge Resources
- [Challenge Announcement](https://openadmet.ghost.io/openadmet-expansionrx-blind-challenge/)
- [HuggingFace Space](https://huggingface.co/spaces/openadmet/OpenADMET-ExpansionRx-Challenge)
- [Training Dataset](https://huggingface.co/datasets/openadmet/openadmet-expansionrx-challenge-train-data)
- [Test Dataset](https://huggingface.co/datasets/openadmet/openadmet-expansionrx-challenge-test-data-blinded)

### Scientific Literature
- RDKit: Open-source cheminformatics toolkit
- XGBoost: Scalable tree boosting system
- Morgan Fingerprints: Circular fingerprints for molecular similarity

## License

This project is released under the MIT License. See LICENSE file for details.

## Citation

If you use this pipeline in your research, please cite:

```bibtex
@software{admet_prediction_pipeline,
  title={ADMET Prediction Pipeline for OpenADMET-ExpansionRx Challenge},
  author={Your Name},
  year={2025},
  url={https://github.com/yourusername/admet-prediction-pipeline}
}
```

## Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## Acknowledgments

- **OpenADMET** for organizing the challenge
- **ExpansionRx** for providing the dataset
- **HuggingFace** for hosting the challenge
- **RDKit** community for molecular featurization tools

## Contact

For questions or issues:
- Open a GitHub issue
- Join the [OpenADMET Discord](https://discord.gg/openadmet)
- Email: your.email@example.com

---

**Good luck with the challenge! 🚀**
