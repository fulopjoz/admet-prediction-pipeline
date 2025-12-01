# Changelog

All notable changes to this project will be documented in this file.

## [1.0.1] - 2025-12-01

### Fixed
- **Critical Bug**: Fixed test dataset split specification
  - Changed from `split='train'` to `split='test'` in `src/data_loader.py`
  - The test dataset only has a 'test' split, not a 'train' split
  - This was causing a `ValueError: Unknown split "train". Should be one of ['test']`
  - Identified by reviewing the official OpenADMET tutorial

- **Column Name Fix**: Corrected KSOL endpoint name
  - Changed from `KSol` to `KSOL` in `config/config.yaml`
  - The actual dataset uses uppercase `KSOL`, not `KSol`
  - This was preventing the KSOL endpoint from being recognized

### Changed
- Updated requirements.txt to specify `numpy>=1.24.0,<2.0.0` for RDKit compatibility

### Added
- Test script (`test_pipeline.py`) to verify installation
- QUICKSTART.md for 5-minute setup guide
- CHANGELOG.md to track changes

## [1.0.0] - 2025-12-01

### Added
- Initial release of ADMET prediction pipeline
- Support for 9 ADMET endpoints
- XGBoost, LightGBM, and Random Forest ensemble models
- Morgan fingerprints and RDKit descriptors for molecular featurization
- 5-fold cross-validation
- Automated data download, training, and submission generation
- Comprehensive documentation (README.md)
- MIT License
