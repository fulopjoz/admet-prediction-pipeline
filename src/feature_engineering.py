"""
Feature engineering for molecular data
"""
import numpy as np
import pandas as pd
import logging
from typing import Tuple, Dict, Any, List
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
import joblib
import os


logger = logging.getLogger(__name__)


class MolecularFeaturizer:
    """
    Class for generating molecular features from SMILES strings
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize featurizer with configuration
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.scaler = StandardScaler()
        self.variance_selector = None
        self.feature_names = []
        
    def smiles_to_mol(self, smiles: str):
        """Convert SMILES string to RDKit molecule object"""
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                logger.warning(f"Failed to parse SMILES: {smiles}")
            return mol
        except Exception as e:
            logger.error(f"Error parsing SMILES {smiles}: {e}")
            return None
    
    def generate_morgan_fingerprint(self, mol, radius: int = 2, nBits: int = 2048) -> np.ndarray:
        """
        Generate Morgan (circular) fingerprint
        
        Args:
            mol: RDKit molecule object
            radius: Fingerprint radius (ECFP4 = radius 2)
            nBits: Number of bits in fingerprint
            
        Returns:
            Fingerprint as numpy array
        """
        if mol is None:
            return np.zeros(nBits)
        
        try:
            fp = AllChem.GetMorganFingerprintAsBitVect(
                mol, 
                radius=radius, 
                nBits=nBits,
                useFeatures=self.config['features']['morgan'].get('useFeatures', False)
            )
            return np.array(fp)
        except Exception as e:
            logger.error(f"Error generating Morgan fingerprint: {e}")
            return np.zeros(nBits)
    
    def calculate_rdkit_descriptors(self, mol) -> Dict[str, float]:
        """
        Calculate RDKit molecular descriptors
        
        Args:
            mol: RDKit molecule object
            
        Returns:
            Dictionary of descriptor names and values
        """
        if mol is None:
            return {}
        
        descriptors = {}
        
        # Get all available descriptors
        descriptor_list = [
            ('MolWt', Descriptors.MolWt),
            ('LogP', Descriptors.MolLogP),
            ('NumHDonors', Descriptors.NumHDonors),
            ('NumHAcceptors', Descriptors.NumHAcceptors),
            ('NumRotatableBonds', Descriptors.NumRotatableBonds),
            ('NumAromaticRings', Descriptors.NumAromaticRings),
            ('TPSA', Descriptors.TPSA),
            ('NumAliphaticRings', Descriptors.NumAliphaticRings),
            ('NumSaturatedRings', Descriptors.NumSaturatedRings),
            ('NumHeteroatoms', Descriptors.NumHeteroatoms),
            ('NumAmideBonds', rdMolDescriptors.CalcNumAmideBonds),
            ('FractionCsp3', Descriptors.FractionCSP3),
            ('NumSpiroAtoms', rdMolDescriptors.CalcNumSpiroAtoms),
            ('NumBridgeheadAtoms', rdMolDescriptors.CalcNumBridgeheadAtoms),
            ('Chi0v', Descriptors.Chi0v),
            ('Chi1v', Descriptors.Chi1v),
            ('Chi2v', Descriptors.Chi2v),
            ('Chi3v', Descriptors.Chi3v),
            ('Chi4v', Descriptors.Chi4v),
            ('Kappa1', Descriptors.Kappa1),
            ('Kappa2', Descriptors.Kappa2),
            ('Kappa3', Descriptors.Kappa3),
            ('LabuteASA', Descriptors.LabuteASA),
            ('PEOE_VSA1', Descriptors.PEOE_VSA1),
            ('PEOE_VSA2', Descriptors.PEOE_VSA2),
            ('PEOE_VSA3', Descriptors.PEOE_VSA3),
            ('SMR_VSA1', Descriptors.SMR_VSA1),
            ('SMR_VSA2', Descriptors.SMR_VSA2),
            ('SMR_VSA3', Descriptors.SMR_VSA3),
            ('SlogP_VSA1', Descriptors.SlogP_VSA1),
            ('SlogP_VSA2', Descriptors.SlogP_VSA2),
            ('SlogP_VSA3', Descriptors.SlogP_VSA3),
            ('EState_VSA1', Descriptors.EState_VSA1),
            ('EState_VSA2', Descriptors.EState_VSA2),
            ('VSA_EState1', Descriptors.VSA_EState1),
            ('VSA_EState2', Descriptors.VSA_EState2),
            ('BalabanJ', Descriptors.BalabanJ),
            ('BertzCT', Descriptors.BertzCT),
            ('HallKierAlpha', Descriptors.HallKierAlpha),
        ]
        
        for name, func in descriptor_list:
            try:
                descriptors[name] = func(mol)
            except Exception as e:
                logger.debug(f"Error calculating descriptor {name}: {e}")
                descriptors[name] = np.nan
        
        return descriptors
    
    def featurize_molecules(self, smiles_list: List[str]) -> Tuple[np.ndarray, List[str]]:
        """
        Generate features for a list of SMILES strings
        
        Args:
            smiles_list: List of SMILES strings
            
        Returns:
            features: Feature matrix (n_samples, n_features)
            feature_names: List of feature names
        """
        logger.info(f"Featurizing {len(smiles_list)} molecules...")
        
        all_features = []
        feature_names = []
        
        for i, smiles in enumerate(smiles_list):
            if (i + 1) % 1000 == 0:
                logger.info(f"Processed {i + 1}/{len(smiles_list)} molecules")
            
            mol = self.smiles_to_mol(smiles)
            
            # Morgan fingerprint
            morgan_config = self.config['features']['morgan']
            morgan_fp = self.generate_morgan_fingerprint(
                mol, 
                radius=morgan_config['radius'],
                nBits=morgan_config['nBits']
            )
            
            features = list(morgan_fp)
            
            if i == 0:
                feature_names.extend([f'Morgan_{j}' for j in range(len(morgan_fp))])
            
            # RDKit descriptors
            if self.config['features']['use_rdkit_descriptors']:
                descriptors = self.calculate_rdkit_descriptors(mol)
                
                if i == 0:
                    feature_names.extend(list(descriptors.keys()))
                
                features.extend(list(descriptors.values()))
            
            all_features.append(features)
        
        features_array = np.array(all_features, dtype=np.float32)
        
        # Handle NaN values in descriptors
        nan_mask = np.isnan(features_array)
        if nan_mask.any():
            logger.warning(f"Found {nan_mask.sum()} NaN values in features, filling with 0")
            features_array = np.nan_to_num(features_array, nan=0.0)
        
        logger.info(f"Generated features with shape: {features_array.shape}")
        
        self.feature_names = feature_names
        
        return features_array, feature_names
    
    def fit_transform(self, smiles_list: List[str]) -> np.ndarray:
        """
        Fit featurizer and transform SMILES to features
        
        Args:
            smiles_list: List of SMILES strings
            
        Returns:
            Transformed feature matrix
        """
        # Generate features
        features, feature_names = self.featurize_molecules(smiles_list)
        
        # Fit and apply variance threshold
        if self.config['features'].get('remove_low_variance', False):
            threshold = self.config['features'].get('variance_threshold', 0.01)
            logger.info(f"Applying variance threshold: {threshold}")
            
            self.variance_selector = VarianceThreshold(threshold=threshold)
            features = self.variance_selector.fit_transform(features)
            
            n_removed = len(feature_names) - features.shape[1]
            logger.info(f"Removed {n_removed} low-variance features")
        
        # Fit and apply scaling
        logger.info("Fitting scaler on features...")
        features = self.scaler.fit_transform(features)
        
        return features
    
    def transform(self, smiles_list: List[str]) -> np.ndarray:
        """
        Transform SMILES to features using fitted featurizer
        
        Args:
            smiles_list: List of SMILES strings
            
        Returns:
            Transformed feature matrix
        """
        # Generate features
        features, _ = self.featurize_molecules(smiles_list)
        
        # Apply variance threshold if fitted
        if self.variance_selector is not None:
            features = self.variance_selector.transform(features)
        
        # Apply scaling
        features = self.scaler.transform(features)
        
        return features
    
    def save(self, filepath: str):
        """Save featurizer to disk"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump(self, filepath)
        logger.info(f"Featurizer saved to {filepath}")
    
    @staticmethod
    def load(filepath: str):
        """Load featurizer from disk"""
        featurizer = joblib.load(filepath)
        logger.info(f"Featurizer loaded from {filepath}")
        return featurizer
