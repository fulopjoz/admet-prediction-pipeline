#!/usr/bin/env python3
"""
Quick test to verify pipeline components
"""
import sys
import numpy as np
import pandas as pd

print("Testing ADMET Prediction Pipeline...")
print("="*80)

# Test imports
print("\n1. Testing imports...")
try:
    from src.utils import load_config, setup_logging
    from src.feature_engineering import MolecularFeaturizer
    from src.models import ADMETPredictor
    print("   ✓ All imports successful")
except Exception as e:
    print(f"   ✗ Import failed: {e}")
    sys.exit(1)

# Test configuration
print("\n2. Testing configuration...")
try:
    config = load_config()
    print(f"   ✓ Configuration loaded")
    print(f"   - Endpoints: {len(config['endpoints'])}")
    print(f"   - Morgan radius: {config['features']['morgan']['radius']}")
except Exception as e:
    print(f"   ✗ Configuration failed: {e}")
    sys.exit(1)

# Test molecular featurization
print("\n3. Testing molecular featurization...")
try:
    test_smiles = [
        "CCO",  # Ethanol
        "CC(=O)O",  # Acetic acid
        "c1ccccc1",  # Benzene
    ]
    
    featurizer = MolecularFeaturizer(config)
    features, feature_names = featurizer.featurize_molecules(test_smiles)
    
    print(f"   ✓ Featurization successful")
    print(f"   - Input: {len(test_smiles)} molecules")
    print(f"   - Output shape: {features.shape}")
    print(f"   - Features: {len(feature_names)}")
except Exception as e:
    print(f"   ✗ Featurization failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test model initialization
print("\n4. Testing model initialization...")
try:
    predictor = ADMETPredictor(config)
    print(f"   ✓ Model initialization successful")
    print(f"   - Endpoints: {len(predictor.endpoints)}")
except Exception as e:
    print(f"   ✗ Model initialization failed: {e}")
    sys.exit(1)

# Test data structures
print("\n5. Testing data structures...")
try:
    # Create dummy training data
    X_dummy = np.random.randn(10, features.shape[1])
    y_dummy = pd.DataFrame({
        'LogD': np.random.randn(10),
        'KSol': np.random.randn(10) * 100,
    })
    
    print(f"   ✓ Data structures created")
    print(f"   - X shape: {X_dummy.shape}")
    print(f"   - y shape: {y_dummy.shape}")
except Exception as e:
    print(f"   ✗ Data structure test failed: {e}")
    sys.exit(1)

print("\n" + "="*80)
print("✓ All tests passed successfully!")
print("="*80)
print("\nThe pipeline is ready to use. Next steps:")
print("1. Run: python scripts/download_data.py")
print("2. Run: python scripts/train_all.py")
print("3. Run: python scripts/generate_submission.py")
