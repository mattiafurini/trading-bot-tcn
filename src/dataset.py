import torch
from torch.utils.data import Dataset
import numpy as np
from typing import Tuple, Optional


class TimeSeriesDataset(Dataset):
    """
    Dataset PyTorch per serie temporali con finestra di lookback scorrevole.
    
    Genera tensori di input (C, L) dove:
    - C = numero di canali/feature
    - L = lunghezza finestra di lookback (es. 64 barre)
    
    E il target y corrispondente alla barra di fine finestra (t = i + lookback - 1).
    """
    def __init__(self, features: np.ndarray, labels: Optional[np.ndarray] = None, lookback: int = 64):
        """
        Args:
            features: Matrice 2D (num_bars, num_features) già scalata.
            labels: Vettore 1D (num_bars,) con label 0, 1, 2. (None per inferenza live).
            lookback: Numero di barre storiche di contesto (default 64).
        """
        self.features = features.astype(np.float32)
        self.labels = labels.astype(np.int64) if labels is not None else None
        self.lookback = lookback
        
        self.n_samples = len(features) - lookback + 1
        if self.n_samples <= 0:
            raise ValueError(f"Dati insufficienti ({len(features)} barre) per lookback di {lookback} barre.")

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # Finestra di lookback [idx : idx + lookback]
        # Shape risultante per Conv1d: (num_features, lookback)
        window = self.features[idx : idx + self.lookback]
        x_tensor = torch.tensor(window.T, dtype=torch.float32)  # (C, L)
        
        if self.labels is not None:
            # La label appartiene all'ultima barra della finestra temporale
            target_idx = idx + self.lookback - 1
            y_tensor = torch.tensor(self.labels[target_idx], dtype=torch.long)
            return x_tensor, y_tensor
        else:
            return x_tensor, torch.tensor(-1, dtype=torch.long)
