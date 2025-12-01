#!/usr/bin/env python3
"""
Download training and test datasets from HuggingFace
"""
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.utils import load_config, setup_logging, print_data_summary
from src.data_loader import download_and_load_data, save_data_locally, validate_data


def main():
    """Main function to download data"""
    # Setup logging
    logger = setup_logging()
    logger.info("Starting data download...")
    
    # Load configuration
    config = load_config()
    
    # Download data from HuggingFace
    train_df, test_df = download_and_load_data(config)
    
    # Validate data
    endpoints = config['endpoints']
    validate_data(train_df, endpoints, is_test=False)
    validate_data(test_df, endpoints, is_test=True)
    
    # Print summary
    print_data_summary(train_df, endpoints)
    
    # Save locally
    raw_data_path = config['paths']['raw_data']
    save_data_locally(train_df, test_df, raw_data_path)
    
    logger.info("Data download completed successfully!")


if __name__ == "__main__":
    main()
