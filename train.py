import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from model import RNNForecast

def make_windows(series, seq_len=10):
    X, y = [], []
    for i in range(len(series) - seq_len):
        X.append(series[i:i+seq_len])
        y.append(series[i+seq_len])
    X = np.array(X).reshape(-1, seq_len, 1)
    y = np.array(y).reshape(-1, 1)
    return X, y

def train_rnn(train_series, val_series=None, seq_len=10, hidden_size=32,
              batch_size=64, epochs=100, lr=1e-3, device="cpu"):
    Xtr, ytr = make_windows(train_series, seq_len)
    Xtr = torch.tensor(Xtr, dtype=torch.float32)
    ytr = torch.tensor(ytr, dtype=torch.float32)

    dtr = DataLoader(TensorDataset(Xtr, ytr), batch_size=batch_size, shuffle=True)

    model = RNNForecast(hidden_size=hidden_size).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    lossf = nn.MSELoss()

    best_loss = float("inf")
    for ep in range(1, epochs+1):
        model.train()
        running = 0.0
        for xb, yb in dtr:
            xb, yb = xb.to(device), yb.to(device)
            optim.zero_grad()
            pred = model(xb)
            loss = lossf(pred, yb)
            loss.backward()
            optim.step()
            running += loss.item() * xb.size(0)
        epoch_loss = running / len(dtr.dataset)

        if epoch_loss < best_loss:
            best_loss = epoch_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if ep % 10 == 0:
            print(f"Epoch {ep:03d} | train MSE: {epoch_loss:.6f}")

    model.load_state_dict(best_state)
    return model, best_loss
