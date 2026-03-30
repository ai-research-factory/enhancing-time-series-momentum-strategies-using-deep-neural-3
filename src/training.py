"""
Training pipeline for Deep Momentum LSTM on multi-asset data.

Implements train_single_split for Phase 3: single 80/20 time-series split
training with the LSTM model and differentiable Sharpe ratio loss.
"""
import os
import pickle

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler

from src.model import DeepMomentumLSTM
from src.loss import sharpe_loss


def load_processed_data(path: str = None) -> dict:
    """Load processed multi-asset data from pickle."""
    if path is None:
        path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "data", "processed", "timeseries.pkl",
        )
    with open(path, "rb") as f:
        return pickle.load(f)


def _scale_features(train_X: np.ndarray, test_X: np.ndarray) -> tuple:
    """Scale 3D features (N, lookback, n_assets) using train data only.

    Fits a scaler per asset on the flattened time dimension, then reshapes back.
    """
    n_train, lookback, n_assets = train_X.shape
    n_test = test_X.shape[0]

    # Reshape to 2D: (N*lookback, n_assets) for per-asset scaling
    train_flat = train_X.reshape(-1, n_assets)
    test_flat = test_X.reshape(-1, n_assets)

    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_flat).reshape(n_train, lookback, n_assets)
    test_scaled = scaler.transform(test_flat).reshape(n_test, lookback, n_assets)

    return train_scaled.astype(np.float32), test_scaled.astype(np.float32), scaler


def train_single_split(
    features: np.ndarray = None,
    forward_returns: np.ndarray = None,
    train_ratio: float = 0.8,
    hidden_size: int = 64,
    num_layers: int = 2,
    dropout: float = 0.1,
    lr: float = 1e-3,
    epochs: int = 100,
    turnover_penalty: float = 0.01,
    batch_size: int = 256,
    device: str = None,
    data_path: str = None,
) -> dict:
    """
    Train on a single 80/20 time-series split.

    Args:
        features: (N, lookback, n_assets) array. If None, loads from pickle.
        forward_returns: (N, n_assets) array. If None, loads from pickle.
        train_ratio: fraction of data for training (default 0.8)
        hidden_size: LSTM hidden dimension
        num_layers: number of LSTM layers
        dropout: dropout rate
        lr: learning rate
        epochs: max training epochs
        turnover_penalty: lambda for turnover regularization
        batch_size: mini-batch size for training
        device: 'cpu' or 'cuda'
        data_path: path to processed pickle

    Returns:
        dict with keys: model, train_positions, test_positions,
        train_returns, test_returns, train_indices, test_indices, scaler
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load data if not provided
    if features is None or forward_returns is None:
        data = load_processed_data(data_path)
        features = data["features"]
        forward_returns = data["forward_returns"]

    n_samples, lookback, n_assets = features.shape

    # Time-series split: first train_ratio for training, rest for test
    split_idx = int(n_samples * train_ratio)
    train_X = features[:split_idx]
    train_y = forward_returns[:split_idx]
    test_X = features[split_idx:]
    test_y = forward_returns[split_idx:]

    print(f"Train: {len(train_X)} samples, Test: {len(test_X)} samples")
    print(f"Input shape: ({lookback}, {n_assets}), Device: {device}")

    # Scale features using train data only
    train_X_scaled, test_X_scaled, scaler = _scale_features(train_X, test_X)

    # Convert to tensors
    train_Xt = torch.tensor(train_X_scaled, dtype=torch.float32, device=device)
    train_yt = torch.tensor(train_y, dtype=torch.float32, device=device)
    test_Xt = torch.tensor(test_X_scaled, dtype=torch.float32, device=device)

    # Initialize LSTM model
    model = DeepMomentumLSTM(
        n_assets=n_assets,
        lookback=lookback,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)

    # Training loop with mini-batches and early stopping
    best_loss = float("inf")
    best_state = None
    patience_counter = 0
    patience = 20

    n_train = len(train_Xt)

    for epoch in range(epochs):
        model.train()
        epoch_losses = []

        # Shuffle indices for mini-batching (within training set only)
        indices = torch.randperm(n_train, device=device)

        for start in range(0, n_train, batch_size):
            end = min(start + batch_size, n_train)
            batch_idx = indices[start:end]

            batch_X = train_Xt[batch_idx]
            batch_y = train_yt[batch_idx]

            optimizer.zero_grad()
            positions = model(batch_X)
            loss = sharpe_loss(positions, batch_y, turnover_penalty=turnover_penalty)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_losses.append(loss.item())

        avg_loss = np.mean(epoch_losses)

        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")

        if avg_loss < best_loss - 1e-6:
            best_loss = avg_loss
            patience_counter = 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  Early stopping at epoch {epoch+1}")
                break

    # Restore best model
    if best_state is not None:
        model.load_state_dict(best_state)

    # Generate predictions
    model.eval()
    with torch.no_grad():
        train_positions = model(train_Xt).cpu().numpy()
        test_positions = model(test_Xt).cpu().numpy()

    return {
        "model": model,
        "train_positions": train_positions,
        "test_positions": test_positions,
        "train_returns": train_y,
        "test_returns": test_y,
        "train_indices": list(range(split_idx)),
        "test_indices": list(range(split_idx, n_samples)),
        "scaler": scaler,
    }
