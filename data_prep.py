import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from statsmodels.datasets import sunspots

def load_sunspots():
    data = sunspots.load_pandas().data
    years = data["YEAR"].values
    series = data["SUNACTIVITY"].values.astype(float)
    return years, series

def normalize_series(series, feature_range=(0,1)):
    scaler = MinMaxScaler(feature_range=feature_range)
    series_norm = scaler.fit_transform(series.reshape(-1,1)).flatten()
    return series_norm, scaler

def split_series(series_norm, train_ratio=0.8):
    split = int(len(series_norm) * train_ratio)
    train = series_norm[:split]
    test = series_norm[split:]
    return train, test

if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from pathlib import Path
    years, series = load_sunspots()
    series_norm, _ = normalize_series(series)
    out_dir = Path("reports/figures")
    out_dir.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(12,5))
    plt.plot(years, series)
    plt.title("Serie histórica de manchas solares")
    plt.xlabel("Año")
    plt.ylabel("Número de manchas solares")
    out_path = out_dir / "sunspots_full_series.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(str(out_path))
