#!/usr/bin/env python3
"""
Train ADMET prediction models for all endpoints
"""
import sys
import os
import time

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.utils import load_config, setup_logging, print_data_summary
from src.data_loader import load_local_data
from src.feature_engineering import MolecularFeaturizer
from src.models import ADMETPredictor


def main():
    """Main training function"""
    start_time = time.time()
    
    # Setup logging
    logger = setup_logging("training.log")
    logger.info("="*80)
    logger.info("ADMET PREDICTION MODEL TRAINING")
    logger.info("="*80)
    
    # Load configuration
    config = load_config()
    logger.info("Configuration loaded")
    
    # Load data
    logger.info("\nLoading data...")
    raw_data_path = config['paths']['raw_data']
    train_df, test_df = load_local_data(raw_data_path)
    
    endpoints = config['endpoints']
    print_data_summary(train_df, endpoints)
    
    # Feature engineering
    logger.info("\n" + "="*80)
    logger.info("FEATURE ENGINEERING")
    logger.info("="*80)
    
    featurizer = MolecularFeaturizer(config)
    
    logger.info("\nGenerating training features...")
    X_train = featurizer.fit_transform(train_df['SMILES'].tolist())
    logger.info(f"Training features shape: {X_train.shape}")
    
    # Save featurizer
    featurizer_path = os.path.join(config['paths']['models'], 'featurizer.pkl')
    featurizer.save(featurizer_path)
    
    # Prepare target dataframe
    y_train_df = train_df[endpoints].copy()
    
    # Train models
    logger.info("\n" + "="*80)
    logger.info("MODEL TRAINING")
    logger.info("="*80)
    
    predictor = ADMETPredictor(config)
    predictor.train(X_train, y_train_df)
    
    # Save models
    models_path = config['paths']['models']
    predictor.save(models_path)
    
    # Training summary
    elapsed_time = time.time() - start_time
    logger.info("\n" + "="*80)
    logger.info("TRAINING COMPLETED")
    logger.info("="*80)
    logger.info(f"Total training time: {elapsed_time/60:.2f} minutes")
    logger.info(f"Models saved to: {models_path}")
    
    # Print CV scores summary
    logger.info("\nCross-Validation Scores Summary:")
    logger.info("-"*80)
    for endpoint in endpoints:
        if endpoint in predictor.models and predictor.models[endpoint].cv_scores:
            scores = predictor.models[endpoint].cv_scores
            logger.info(f"\n{endpoint}:")
            for model_type, score in scores.items():
                logger.info(f"  {model_type:20s}: {score:.4f}")


if __name__ == "__main__":
    main()
