"""
Utility functions for ADMET prediction pipeline
"""
import os
import yaml
import joblib
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, List, Tuple


def setup_logging(log_file: str = None):
    """Setup logging configuration"""
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    if log_file:
        logging.basicConfig(
            level=logging.INFO,
            format=log_format,
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
    else:
        logging.basicConfig(level=logging.INFO, format=log_format)
    
    return logging.getLogger(__name__)


def load_config(config_path: str = "config/config.yaml") -> Dict[str, Any]:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def save_model(model: Any, filepath: str):
    """Save model to disk"""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    joblib.dump(model, filepath)
    logging.info(f"Model saved to {filepath}")


def load_model(filepath: str) -> Any:
    """Load model from disk"""
    model = joblib.load(filepath)
    logging.info(f"Model loaded from {filepath}")
    return model


def calculate_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate Mean Absolute Error"""
    mask = ~np.isnan(y_true)
    if mask.sum() == 0:
        return np.nan
    return np.mean(np.abs(y_true[mask] - y_pred[mask]))


def calculate_rae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate Relative Absolute Error
    RAE = MAE / range(y_true)
    """
    mask = ~np.isnan(y_true)
    if mask.sum() == 0:
        return np.nan
    
    y_true_valid = y_true[mask]
    y_pred_valid = y_pred[mask]
    
    mae = np.mean(np.abs(y_true_valid - y_pred_valid))
    data_range = np.max(y_true_valid) - np.min(y_true_valid)
    
    if data_range == 0:
        return np.nan
    
    return mae / data_range


def log_transform_if_needed(values: np.ndarray, endpoint: str) -> np.ndarray:
    """
    Apply log transformation to endpoints that are not already on log scale
    LogD is already on log scale, so we don't transform it
    """
    if endpoint == "LogD":
        return values
    
    # For other endpoints, apply log10 transformation
    # Add small epsilon to avoid log(0)
    epsilon = 1e-10
    return np.log10(values + epsilon)


def inverse_log_transform_if_needed(values: np.ndarray, endpoint: str) -> np.ndarray:
    """Inverse of log_transform_if_needed"""
    if endpoint == "LogD":
        return values
    
    return 10 ** values


def create_submission_file(predictions: Dict[str, np.ndarray], 
                          molecule_names: List[str],
                          output_path: str):
    """
    Create submission CSV file
    
    Args:
        predictions: Dictionary mapping endpoint names to prediction arrays
        molecule_names: List of molecule identifiers
        output_path: Path to save submission file
    """
    submission_df = pd.DataFrame({
        'Molecule Name': molecule_names
    })
    
    # Add predictions for each endpoint
    for endpoint, preds in predictions.items():
        submission_df[endpoint] = preds
    
    # Save to CSV
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    submission_df.to_csv(output_path, index=False)
    logging.info(f"Submission file saved to {output_path}")
    
    return submission_df


def get_available_data_mask(df: pd.DataFrame, endpoint: str) -> np.ndarray:
    """Get boolean mask for samples with non-missing target values"""
    return ~df[endpoint].isna()


def print_data_summary(df: pd.DataFrame, endpoints: List[str]):
    """Print summary statistics for the dataset"""
    print("\n" + "="*80)
    print("DATASET SUMMARY")
    print("="*80)
    print(f"Total samples: {len(df)}")
    print(f"\nEndpoint availability:")
    print("-"*80)
    
    for endpoint in endpoints:
        if endpoint in df.columns:
            available = (~df[endpoint].isna()).sum()
            missing = df[endpoint].isna().sum()
            pct_available = 100 * available / len(df)
            print(f"{endpoint:40s}: {available:5d} / {len(df):5d} ({pct_available:5.1f}%)")
    
    print("="*80 + "\n")


def ensure_dir(directory: str):
    """Create directory if it doesn't exist"""
    Path(directory).mkdir(parents=True, exist_ok=True)
