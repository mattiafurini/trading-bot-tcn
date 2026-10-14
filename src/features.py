import numpy as np
import pandas as pd
from typing import List, Tuple, Optional
import pickle
from sklearn.preprocessing import RobustScaler, StandardScaler


FEATURE_COLUMNS = [
    "ret_close",
    "ret_high",
    "ret_low",
    "hl_spread",
    "norm_atr",
    "vol_vs_sma",
    "ret_volume",
    "dist_sma_50",
    "dist_sma_200",
    "dist_sma_50_200",
    "funding_rate"
]


def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Calcola l'Average True Range (ATR) a N periodi."""
    high = df["high"]
    low = df["low"]
    close = df["close"]
    prev_close = close.shift(1)

    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=period, min_periods=period).mean()
    return atr


def extract_features(
    df: pd.DataFrame,
    sma_short: int = 50,
    sma_long: int = 200,
    atr_period: int = 14,
    vol_period: int = 20
) -> pd.DataFrame:
    """
    Calcola le feature rigorosamente stazionarie per il tensore di input della TCN.
    
    1. Log-returns su Close, High, Low
    2. High-Low log spread
    3. ATR normalizzato su Close
    4. Volume relativo rispetto a SMA20 e variazione log-volume
    5. Distanza percentuale da SMA50 e SMA200
    6. Funding Rate
    """
    df = df.copy()

    # 1. Log-returns stazionari
    df["ret_close"] = np.log(df["close"] / df["close"].shift(1))
    df["ret_high"] = np.log(df["high"] / df["high"].shift(1))
    df["ret_low"] = np.log(df["low"] / df["low"].shift(1))
    df["hl_spread"] = np.log(df["high"] / df["low"])

    # 2. Volatilità normalizzata (ATR / Close)
    atr = calculate_atr(df, period=atr_period)
    df["norm_atr"] = atr / df["close"]

    # 3. Volume e dinamica
    vol_sma = df["volume"].rolling(window=vol_period, min_periods=vol_period).mean()
    df["vol_vs_sma"] = (df["volume"] - vol_sma) / (vol_sma + 1e-8)
    df["ret_volume"] = np.log((df["volume"] + 1.0) / (df["volume"].shift(1) + 1.0))

    # 4. Distanza da indicatori strutturali (SMA 50 e SMA 200)
    sma_50 = df["close"].rolling(window=sma_short, min_periods=sma_short).mean()
    sma_200 = df["close"].rolling(window=sma_long, min_periods=sma_long).mean()
    
    df["dist_sma_50"] = (df["close"] - sma_50) / (sma_50 + 1e-8)
    df["dist_sma_200"] = (df["close"] - sma_200) / (sma_200 + 1e-8)
    df["dist_sma_50_200"] = (sma_50 - sma_200) / (sma_200 + 1e-8)

    # 5. Funding Rate (assicura presenza)
    if "funding_rate" not in df.columns:
        df["funding_rate"] = 0.0
    else:
        df["funding_rate"] = df["funding_rate"].fillna(0.0)

    # Rimuove NaN iniziali dovuti alle finestre di rolling (SMA 200)
    df.dropna(subset=FEATURE_COLUMNS, inplace=True)
    df.reset_index(drop=True, inplace=True)

    return df


class FeatureScaler:
    """
    Gestore dello scaling delle feature. Fitta lo scaler esclusivamente
    sul Train Set per impedire qualsiasi forma di Data Snooping / Data Leakage.
    """
    def __init__(self, method: str = "robust", clip_quantiles: Optional[Tuple[float, float]] = (0.01, 0.99)):
        self.method = method
        self.clip_quantiles = clip_quantiles
        self.scaler = RobustScaler() if method == "robust" else StandardScaler()
        self.clip_bounds = {}
        self.feature_names = FEATURE_COLUMNS
        self.is_fitted = False

    def fit(self, df: pd.DataFrame):
        X = df[self.feature_names].copy()
        
        # Calcolo opzionale dei limiti di clipping (winsorization) per evitare valori assurdi
        if self.clip_quantiles is not None:
            q_low, q_high = self.clip_quantiles
            for col in self.feature_names:
                low_val = X[col].quantile(q_low)
                high_val = X[col].quantile(q_high)
                self.clip_bounds[col] = (low_val, high_val)

        # Applica clipping prima del fit
        if self.clip_bounds:
            for col, (low, high) in self.clip_bounds.items():
                X[col] = X[col].clip(lower=low, upper=high)

        self.scaler.fit(X.values)
        self.is_fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        feature_cols = getattr(self, "feature_names", FEATURE_COLUMNS)
        X = df[feature_cols].copy()
        if hasattr(self, "clip_bounds") and self.clip_bounds:
            for col, (low, high) in self.clip_bounds.items():
                if col in X.columns:
                    X[col] = X[col].clip(lower=low, upper=high)

        return self.scaler.transform(X.values)

    def fit_transform(self, df: pd.DataFrame) -> np.ndarray:
        return self.fit(df).transform(df)

    def save(self, filepath: str):
        with open(filepath, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, filepath: str) -> "FeatureScaler":
        import sys
        # Gestisce oggetti serializzati da Jupyter Notebooks (__main__.FeatureScaler)
        current_main = sys.modules.get("__main__")
        if current_main and not hasattr(current_main, "FeatureScaler"):
            setattr(current_main, "FeatureScaler", cls)

        class NotebookSafeUnpickler(pickle.Unpickler):
            def find_class(self, module, name):
                if name == "FeatureScaler":
                    return cls
                return super().find_class(module, name)

        with open(filepath, "rb") as f:
            return NotebookSafeUnpickler(f).load()
