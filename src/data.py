"""Data loading and schema validation."""
from pathlib import Path
import pandas as pd

DATASETS = {
    'human': ('Human-AI Interaction.csv', 'latent_trust_score'),
    'flood': ('train.csv', 'FloodProbability'),
}

def load_dataset(kind='human', data_dir='data', csv_path=None, target=None, nrows=None):
    if csv_path is None:
        filename, target_default = DATASETS[kind]
        csv_path = Path(data_dir) / filename
        target = target or target_default
    if target is None:
        raise ValueError('target is required for custom CSV')
    df = pd.read_csv(csv_path, nrows=nrows)
    if target not in df:
        raise ValueError(f'Target column not found: {target}')
    df[target] = pd.to_numeric(df[target], errors='coerce')
    df = df.dropna(subset=[target]).copy()
    excluded = [c for c in ['id'] if c in df.columns and c != target]
    X = df.drop(columns=[target]+excluded)
    X = X.drop(columns=[c for c in X if X[c].isna().all()])
    if len(X) < 10 or X.shape[1] < 1:
        raise ValueError('Not enough data to train')
    return X, df[target]
