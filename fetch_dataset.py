from pathlib import Path

import pandas as pd
from ucimlrepo import fetch_ucirepo


def fetch_heart_disease_dataset() -> pd.DataFrame:
    """
    Fetch the UCI Heart Disease dataset and save it to data/heart_disease.csv.
    Dataset ID: 45
    """
    print("Fetching Heart Disease dataset from UCI ML Repository...")

    heart_disease = fetch_ucirepo(id=45)
    X = heart_disease.data.features
    y = heart_disease.data.targets

    data = pd.concat([X, y.rename("target")], axis=1)

    print("\n=== Dataset Metadata ===")
    print(heart_disease.metadata)
    print("\n=== Variable Information ===")
    print(heart_disease.variables)

    data_dir = Path(__file__).resolve().parent / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    dataset_path = data_dir / "heart_disease.csv"
    data.to_csv(dataset_path, index=False)

    print(f"\nDataset saved to {dataset_path}")
    print(f"Dataset shape: {data.shape}")
    print(f"Missing values:\n{data.isnull().sum()}")
    print(f"\nTarget distribution:\n{data['target'].value_counts()}")

    return data


if __name__ == "__main__":
    fetch_heart_disease_dataset()
