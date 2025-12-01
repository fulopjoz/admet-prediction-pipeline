# Quick Start Guide

Get up and running with the ADMET prediction pipeline in 5 minutes!

## Prerequisites

- Python 3.8+ installed
- pip package manager
- 8GB+ RAM
- Internet connection

## Installation (3 steps)

### 1. Clone the repository

```bash
git clone https://github.com/fulopjoz/admet-prediction-pipeline.git
cd admet-prediction-pipeline
```

### 2. Install dependencies

**Option A: Using pip (recommended)**

```bash
pip install -r requirements.txt
```

**Option B: Using conda (for RDKit)**

```bash
conda install -c conda-forge rdkit
pip install -r requirements.txt
```

**Note**: If you encounter numpy compatibility issues with RDKit, run:
```bash
pip install "numpy<2.0"
```

### 3. Verify installation

```bash
python test_pipeline.py
```

You should see:
```
✓ All tests passed successfully!
```

## Usage (3 commands)

### Step 1: Download Data

```bash
python scripts/download_data.py
```

**Expected output**: Training and test datasets downloaded to `data/raw/`

**Time**: ~1-2 minutes

### Step 2: Train Models

```bash
python scripts/train_all.py
```

**Expected output**: 
- Trained models saved to `models/saved_models/`
- Cross-validation scores displayed
- Training log saved to `training.log`

**Time**: ~30-60 minutes (depending on hardware)

### Step 3: Generate Submission

```bash
python scripts/generate_submission.py
```

**Expected output**: 
- Submission file created at `submissions/submission.csv`
- Sample predictions displayed

**Time**: ~1-2 minutes

## Submit to Challenge

1. Go to [HuggingFace Challenge Space](https://huggingface.co/spaces/openadmet/OpenADMET-ExpansionRx-Challenge)
2. Click on the **Submit** tab
3. Upload `submissions/submission.csv`
4. Wait for results on the leaderboard!

## Troubleshooting

### Issue: RDKit import error

**Solution**: Install RDKit via conda
```bash
conda install -c conda-forge rdkit
```

### Issue: Out of memory

**Solution**: Edit `config/config.yaml` and reduce:
```yaml
models:
  xgboost:
    n_estimators: 500  # Reduce from 1000
```

### Issue: Slow training

**Solution**: Reduce cross-validation folds in `config/config.yaml`:
```yaml
training:
  cv_folds: 3  # Reduce from 5
```

### Issue: Missing data files

**Solution**: Run download script first
```bash
python scripts/download_data.py
```

## What's Next?

- Review training logs to understand model performance
- Experiment with hyperparameters in `config/config.yaml`
- Add custom features in `src/feature_engineering.py`
- Try different ensemble weights
- Submit multiple times to improve scores!

## Need Help?

- Check the [full README](README.md)
- Join the [OpenADMET Discord](https://discord.gg/openadmet)
- Open an issue on GitHub

---

**Good luck! 🚀**
