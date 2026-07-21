# NS2B-NS3-dl-benchmark

Benchmarking a PyTorch MLP against classical ML (Random Forest, SVM) for classifying bioactive compounds against the dengue NS2B-NS3 protease.

## Background

This project extends the ML classification pipeline from my MS thesis (ChEMBL-derived NS2B-NS3 bioactivity data, Morgan fingerprints, RF/SVM achieving ROC-AUC 0.995) by adding a deep learning baseline for direct comparison.

## Pipeline

1. `src/featurize.py` Converts SMILES to Morgan fingerprints (radius=2, 2048 bits)
2. `src/model.py` PyTorch MLP architecture (2048 → 256 → 64 → 1)
3. `src/train.py' Trains the MLP and classical RF/SVM baselines, reports ROC-AUC for all three

## Setup

```bash
pip install -r requirements.txt
```

## Usage

```bash
python src/featurize.py   # generate fingerprints from data/NS2B_NS3_compounds.csv
python src/train.py       # train MLP + baselines, print ROC-AUC comparison
```

## Results

| Model | ROC-AUC |
|---|---|
| Random Forest (thesis baseline) | 0.995 |
| SVM | *TBD* |
| PyTorch MLP | *TBD* |

## Discussion

*To be added after running experiments comparing why classical ML vs. deep learning performs differently on this dataset size and feature representation.*

## Author

Samra khaliq

