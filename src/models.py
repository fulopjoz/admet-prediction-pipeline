"""
Model training and prediction for ADMET endpoints
"""
import numpy as np
import pandas as pd
import logging
from typing import Dict, Any, List, Tuple
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import xgboost as xgb
import lightgbm as lgb
import joblib
import os


logger = logging.getLogger(__name__)


class ADMETModel:
    """
    Wrapper class for ADMET endpoint prediction models
    """
    
    def __init__(self, endpoint: str, config: Dict[str, Any]):
        """
        Initialize model for specific endpoint
        
        Args:
            endpoint: Name of ADMET endpoint
            config: Configuration dictionary
        """
        self.endpoint = endpoint
        self.config = config
        self.models = {}
        self.cv_scores = {}
        
    def train_xgboost(self, X_train: np.ndarray, y_train: np.ndarray) -> xgb.XGBRegressor:
        """Train XGBoost model"""
        logger.info(f"Training XGBoost for {self.endpoint}...")
        
        params = self.config['models']['xgboost'].copy()
        
        model = xgb.XGBRegressor(**params)
        model.fit(
            X_train, y_train,
            eval_set=[(X_train, y_train)],
            verbose=False
        )
        
        return model
    
    def train_lightgbm(self, X_train: np.ndarray, y_train: np.ndarray) -> lgb.LGBMRegressor:
        """Train LightGBM model"""
        logger.info(f"Training LightGBM for {self.endpoint}...")
        
        params = self.config['models']['lightgbm'].copy()
        
        model = lgb.LGBMRegressor(**params)
        model.fit(X_train, y_train)
        
        return model
    
    def train_random_forest(self, X_train: np.ndarray, y_train: np.ndarray) -> RandomForestRegressor:
        """Train Random Forest model"""
        logger.info(f"Training Random Forest for {self.endpoint}...")
        
        params = self.config['models']['random_forest'].copy()
        
        model = RandomForestRegressor(**params)
        model.fit(X_train, y_train)
        
        return model
    
    def cross_validate(self, X: np.ndarray, y: np.ndarray, model_type: str) -> Tuple[Any, float]:
        """
        Perform cross-validation for a model type
        
        Args:
            X: Feature matrix
            y: Target values
            model_type: Type of model ('xgboost', 'lightgbm', 'random_forest')
            
        Returns:
            trained_model: Model trained on full data
            cv_score: Mean CV MAE score
        """
        n_folds = self.config['training']['cv_folds']
        random_seed = self.config['training']['random_seed']
        
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=random_seed)
        
        cv_scores = []
        
        logger.info(f"Performing {n_folds}-fold CV for {model_type} on {self.endpoint}...")
        
        for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
            X_train_fold, X_val_fold = X[train_idx], X[val_idx]
            y_train_fold, y_val_fold = y[train_idx], y[val_idx]
            
            # Train model
            if model_type == 'xgboost':
                model = self.train_xgboost(X_train_fold, y_train_fold)
            elif model_type == 'lightgbm':
                model = self.train_lightgbm(X_train_fold, y_train_fold)
            elif model_type == 'random_forest':
                model = self.train_random_forest(X_train_fold, y_train_fold)
            else:
                raise ValueError(f"Unknown model type: {model_type}")
            
            # Predict on validation set
            y_pred = model.predict(X_val_fold)
            
            # Calculate MAE
            mae = mean_absolute_error(y_val_fold, y_pred)
            cv_scores.append(mae)
            
            logger.info(f"  Fold {fold + 1}/{n_folds}: MAE = {mae:.4f}")
        
        mean_cv_score = np.mean(cv_scores)
        std_cv_score = np.std(cv_scores)
        
        logger.info(f"CV Score for {model_type}: {mean_cv_score:.4f} ± {std_cv_score:.4f}")
        
        # Train final model on full data
        if model_type == 'xgboost':
            final_model = self.train_xgboost(X, y)
        elif model_type == 'lightgbm':
            final_model = self.train_lightgbm(X, y)
        elif model_type == 'random_forest':
            final_model = self.train_random_forest(X, y)
        
        return final_model, mean_cv_score
    
    def train(self, X: np.ndarray, y: np.ndarray):
        """
        Train all model types for this endpoint
        
        Args:
            X: Feature matrix
            y: Target values
        """
        logger.info(f"\n{'='*80}")
        logger.info(f"Training models for endpoint: {self.endpoint}")
        logger.info(f"Training samples: {len(y)}")
        logger.info(f"{'='*80}\n")
        
        # Train each model type
        model_types = ['xgboost', 'lightgbm', 'random_forest']
        
        for model_type in model_types:
            model, cv_score = self.cross_validate(X, y, model_type)
            self.models[model_type] = model
            self.cv_scores[model_type] = cv_score
        
        logger.info(f"\nTraining completed for {self.endpoint}")
        logger.info(f"CV Scores:")
        for model_type, score in self.cv_scores.items():
            logger.info(f"  {model_type}: {score:.4f}")
    
    def predict(self, X: np.ndarray, use_ensemble: bool = True) -> np.ndarray:
        """
        Make predictions using trained models
        
        Args:
            X: Feature matrix
            use_ensemble: Whether to use ensemble of models
            
        Returns:
            Predictions
        """
        if not self.models:
            raise ValueError(f"No models trained for endpoint {self.endpoint}")
        
        if use_ensemble:
            # Ensemble prediction using weighted average
            weights = self.config['training']['ensemble_weights']
            predictions = np.zeros(len(X))
            total_weight = 0
            
            for model_type, model in self.models.items():
                weight = weights.get(model_type, 0)
                if weight > 0:
                    pred = model.predict(X)
                    predictions += weight * pred
                    total_weight += weight
            
            predictions /= total_weight
            
        else:
            # Use only XGBoost (typically best single model)
            predictions = self.models['xgboost'].predict(X)
        
        return predictions
    
    def save(self, directory: str):
        """Save all models to directory"""
        os.makedirs(directory, exist_ok=True)
        
        for model_type, model in self.models.items():
            filepath = os.path.join(directory, f"{self.endpoint}_{model_type}.pkl")
            joblib.dump(model, filepath)
        
        # Save CV scores
        scores_path = os.path.join(directory, f"{self.endpoint}_cv_scores.pkl")
        joblib.dump(self.cv_scores, scores_path)
        
        logger.info(f"Models for {self.endpoint} saved to {directory}")
    
    def load(self, directory: str):
        """Load all models from directory"""
        model_types = ['xgboost', 'lightgbm', 'random_forest']
        
        for model_type in model_types:
            filepath = os.path.join(directory, f"{self.endpoint}_{model_type}.pkl")
            if os.path.exists(filepath):
                self.models[model_type] = joblib.load(filepath)
        
        # Load CV scores
        scores_path = os.path.join(directory, f"{self.endpoint}_cv_scores.pkl")
        if os.path.exists(scores_path):
            self.cv_scores = joblib.load(scores_path)
        
        logger.info(f"Models for {self.endpoint} loaded from {directory}")


class ADMETPredictor:
    """
    Multi-endpoint ADMET predictor
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize predictor
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.endpoints = config['endpoints']
        self.models = {}
        
        # Initialize model for each endpoint
        for endpoint in self.endpoints:
            self.models[endpoint] = ADMETModel(endpoint, config)
    
    def train(self, X: np.ndarray, y_df: pd.DataFrame):
        """
        Train models for all endpoints
        
        Args:
            X: Feature matrix
            y_df: Dataframe with target values for all endpoints
        """
        for endpoint in self.endpoints:
            if endpoint not in y_df.columns:
                logger.warning(f"Endpoint {endpoint} not found in target dataframe, skipping")
                continue
            
            # Get samples with non-missing target values
            mask = ~y_df[endpoint].isna()
            n_samples = mask.sum()
            
            if n_samples == 0:
                logger.warning(f"No training samples for endpoint {endpoint}, skipping")
                continue
            
            X_train = X[mask]
            y_train = y_df[endpoint][mask].values
            
            # Train model
            self.models[endpoint].train(X_train, y_train)
    
    def predict(self, X: np.ndarray, use_ensemble: bool = True) -> Dict[str, np.ndarray]:
        """
        Predict all endpoints
        
        Args:
            X: Feature matrix
            use_ensemble: Whether to use ensemble predictions
            
        Returns:
            Dictionary mapping endpoint names to predictions
        """
        predictions = {}
        
        for endpoint in self.endpoints:
            if endpoint in self.models and self.models[endpoint].models:
                predictions[endpoint] = self.models[endpoint].predict(X, use_ensemble)
            else:
                logger.warning(f"No model available for endpoint {endpoint}, using zeros")
                predictions[endpoint] = np.zeros(len(X))
        
        return predictions
    
    def save(self, directory: str):
        """Save all models"""
        os.makedirs(directory, exist_ok=True)
        
        for endpoint, model in self.models.items():
            model.save(directory)
        
        logger.info(f"All models saved to {directory}")
    
    def load(self, directory: str):
        """Load all models"""
        for endpoint, model in self.models.items():
            model.load(directory)
        
        logger.info(f"All models loaded from {directory}")
