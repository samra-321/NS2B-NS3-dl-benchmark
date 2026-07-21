"""Generate Morgan fingerprints from raw ChEMBL export for NS2B-NS3 bioactivity data.
Applies IC50 <= 1000 nM threshold to derive binary activity labels, matching thesis methodology."""
from rdkit import Chem
from rdkit.Chem import AllChem
import numpy as np
import pandas as pd


def get_morgan_fp(smiles, radius=2, n_bits=2048):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, n_bits)
    return np.array(fp)


def featurize_dataset(csv_path, smiles_col="Smiles", value_col="Standard Value",
                       ic50_cutoff_nm=1000):
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=[smiles_col, value_col])

    fps, labels = [], []
    skipped = 0
    for _, row in df.iterrows():
        fp = get_morgan_fp(row[smiles_col])
        if fp is None:
            skipped += 1
            continue
        label = 1 if row[value_col] <= ic50_cutoff_nm else 0
        fps.append(fp)
        labels.append(label)

    print(f"Processed {len(fps)} compounds, skipped {skipped} invalid SMILES")
    print(f"Active: {sum(labels)}, Inactive: {len(labels) - sum(labels)}")

    return np.array(fps), np.array(labels)


if __name__ == "__main__":
    X, y = featurize_dataset("data/chEMBL Data.csv")
    np.save("data/X_fingerprints.npy", X)
    np.save("data/y_labels.npy", y)
    print(f"Saved: {X.shape[0]} compounds, {X.shape[1]} bits each")
