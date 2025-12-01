"""
Data loading utilities for ADMET prediction pipeline
"""
import os
import pandas as pd
import logging
from datasets import load_dataset
from typing import Tuple, Dict, Any


logger = logging.getLogger(__name__)


def download_and_load_data(config: Dict[str, Any]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Download and load training and test datasets from HuggingFace
    
    Args:
        config: Configuration dictionary
        
    Returns:
        train_df: Training dataframe
        test_df: Test dataframe
    """
    logger.info("Loading datasets from HuggingFace...")
    
    # Load training data
    train_dataset_name = config['data']['train_dataset']
    logger.info(f"Loading training data: {train_dataset_name}")
    train_dataset = load_dataset(train_dataset_name, split='train')
    train_df = train_dataset.to_pandas()
    
    # Load test data (blinded)
    test_dataset_name = config['data']['test_dataset']
    logger.info(f"Loading test data: {test_dataset_name}")
    test_dataset = load_dataset(test_dataset_name, split='test')
    test_df = test_dataset.to_pandas()
    
    logger.info(f"Training data shape: {train_df.shape}")
    logger.info(f"Test data shape: {test_df.shape}")
    
    return train_df, test_df


def save_data_locally(train_df: pd.DataFrame, 
                     test_df: pd.DataFrame,
                     raw_data_path: str):
    """
    Save datasets locally for faster access
    
    Args:
        train_df: Training dataframe
        test_df: Test dataframe
        raw_data_path: Path to save raw data
    """
    os.makedirs(raw_data_path, exist_ok=True)
    
    train_path = os.path.join(raw_data_path, 'train.csv')
    test_path = os.path.join(raw_data_path, 'test.csv')
    
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    
    logger.info(f"Training data saved to {train_path}")
    logger.info(f"Test data saved to {test_path}")


def load_local_data(raw_data_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load datasets from local files
    
    Args:
        raw_data_path: Path to raw data directory
        
    Returns:
        train_df: Training dataframe
        test_df: Test dataframe
    """
    train_path = os.path.join(raw_data_path, 'train.csv')
    test_path = os.path.join(raw_data_path, 'test.csv')
    
    if not os.path.exists(train_path) or not os.path.exists(test_path):
        raise FileNotFoundError(
            f"Data files not found in {raw_data_path}. "
            "Please run download_data.py first."
        )
    
    logger.info(f"Loading training data from {train_path}")
    train_df = pd.read_csv(train_path)
    
    logger.info(f"Loading test data from {test_path}")
    test_df = pd.read_csv(test_path)
    
    logger.info(f"Training data shape: {train_df.shape}")
    logger.info(f"Test data shape: {test_df.shape}")
    
    return train_df, test_df


def validate_data(df: pd.DataFrame, endpoints: list, is_test: bool = False):
    """
    Validate dataset structure
    
    Args:
        df: Dataframe to validate
        endpoints: List of endpoint names
        is_test: Whether this is test data (targets may be missing)
    """
    # Check required columns
    required_cols = ['Molecule Name', 'SMILES']
    missing_cols = [col for col in required_cols if col not in df.columns]
    
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # Check for duplicate molecule names
    if df['Molecule Name'].duplicated().any():
        logger.warning("Duplicate molecule names found in dataset")
    
    # Check SMILES validity (basic check)
    if df['SMILES'].isna().any():
        n_missing = df['SMILES'].isna().sum()
        raise ValueError(f"Found {n_missing} missing SMILES strings")
    
    # Check endpoints (only for training data)
    if not is_test:
        for endpoint in endpoints:
            if endpoint not in df.columns:
                logger.warning(f"Endpoint '{endpoint}' not found in dataset")
            else:
                n_available = (~df[endpoint].isna()).sum()
                pct = 100 * n_available / len(df)
                logger.info(f"Endpoint '{endpoint}': {n_available}/{len(df)} ({pct:.1f}%) available")
    
    logger.info("Data validation passed")
