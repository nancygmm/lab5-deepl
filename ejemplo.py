from pathlib import Path
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from train import train_rnn, make_windows
from data_prep import load_sunspots, normalize_series, split_series

years, series = load_sunspots()
series_norm, scaler = normalize_series(series)
series_norm_train, series_norm_test = split_series(series_norm, train_ratio=0.8)

seq_len = 10
model, best_loss = train_rnn(train_series=series_norm_train, seq_len=seq_len, hidden_size=32, epochs=50, lr=1e-3)

Xte, yte = make_windows(series_norm_test, seq_len)
Xte_t = torch.tensor(Xte, dtype=torch.float32)

model.eval()
with torch.no_grad():
    pred = model(Xte_t).numpy()

import numpy as np
yte_inv = scaler.inverse_transform(np.array(yte).reshape(-1,1)).flatten()
pred_inv = scaler.inverse_transform(pred).flatten()

out_dir = Path("reports/figures")
out_dir.mkdir(parents=True, exist_ok=True)
plt.figure(figsize=(12,5))
plt.plot(yte_inv, label="Real")
plt.plot(pred_inv, label="Predicción")
plt.legend()
plt.title("Predicción de manchas solares con RNN")
out_path = out_dir / "rnn_sunspots.png"
plt.savefig(out_path, dpi=150, bbox_inches="tight")
print(str(out_path))
