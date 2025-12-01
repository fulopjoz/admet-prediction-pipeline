#!/usr/bin/env python3
"""
Generate submission file for ADMET challenge
"""
import sys
import os
import time

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.utils import load_config, setup_logging, create_submission_file
from src.data_loader import load_local_data
from src.feature_engineering import MolecularFeaturizer
from src.models import ADMETPredictor


def main():
    """Main prediction function"""
    start_time = time.time()
    
    # Setup logging
    logger = setup_logging("prediction.log")
    logger.info("="*80)
    logger.info("GENERATING SUBMISSION FILE")
    logger.info("="*80)
    
    # Load configuration
    config = load_config()
    logger.info("Configuration loaded")
    
    # Load data
    logger.info("\nLoading test data...")
    raw_data_path = config['paths']['raw_data']
    _, test_df = load_local_data(raw_data_path)
    logger.info(f"Test data shape: {test_df.shape}")
    
    # Load featurizer
    logger.info("\nLoading featurizer...")
    featurizer_path = os.path.join(config['paths']['models'], 'featurizer.pkl')
    featurizer = MolecularFeaturizer.load(featurizer_path)
    
    # Generate test features
    logger.info("\nGenerating test features...")
    X_test = featurizer.transform(test_df['SMILES'].tolist())
    logger.info(f"Test features shape: {X_test.shape}")
    
    # Load models
    logger.info("\nLoading trained models...")
    predictor = ADMETPredictor(config)
    models_path = config['paths']['models']
    predictor.load(models_path)
    
    # Generate predictions
    logger.info("\nGenerating predictions...")
    predictions = predictor.predict(X_test, use_ensemble=True)
    
    # Create submission file
    logger.info("\nCreating submission file...")
    submission_path = os.path.join(config['paths']['submissions'], 'submission.csv')
    submission_df = create_submission_file(
        predictions,
        test_df['Molecule Name'].tolist(),
        submission_path
    )
    
    # Display sample predictions
    logger.info("\nSample predictions (first 5 rows):")
    logger.info("\n" + submission_df.head().to_string())
    
    # Summary
    elapsed_time = time.time() - start_time
    logger.info("\n" + "="*80)
    logger.info("SUBMISSION GENERATION COMPLETED")
    logger.info("="*80)
    logger.info(f"Total time: {elapsed_time:.2f} seconds")
    logger.info(f"Submission file: {submission_path}")
    logger.info(f"Number of predictions: {len(submission_df)}")
    logger.info("\nNext steps:")
    logger.info("1. Review the submission file")
    logger.info("2. Upload to HuggingFace challenge space")
    logger.info("3. Check leaderboard for results")


if __name__ == "__main__":
    main()
