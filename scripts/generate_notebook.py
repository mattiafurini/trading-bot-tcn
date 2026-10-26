import os
import json

notebook = {
    "cells": [],
    "metadata": {
        "accelerator": "GPU",
        "colab": {"provenance": []},
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "codemirror_mode": {"name": "ipython", "version": 3},
            "file_extension": ".py",
            "mimetype": "text/x-python",
            "name": "python",
            "nbconvert_exporter": "python",
            "pygments_lexer": "ipython3",
            "version": "3.12.0"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 4
}

def add_md(source: str):
    notebook["cells"].append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" for line in source.strip().splitlines()]
    })

def add_code(source: str):
    notebook["cells"].append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in source.strip().splitlines()]
    })

# =====================================================================
# Cell 1: Intro Markdown
# =====================================================================
add_md("""# 🛡️ Quantitative TCN Trading Bot: Anti-Chop & False Breakout Filter Suite
### Robust Dynamic ATR Trading on BTC/USDT 4H Perpetual Futures

Questo notebook valuta la configurazione vincente **Dynamic ATR** sui 3 macro-periodi storici (2022 Bear, 2024 Bull, 2025-2026 Recente) confrontando la versione Standard con i filtri Anti-Chop per proteggere il capitale nelle fasi laterali:

---

### 🎛️ Le 4 Configurazioni a Confronto:
1. **1. Standard (Benchmark di Controllo):**
   * `P(dir) >= 0.38`, `Ratio >= 1.25`, nessun filtro volatilità.
2. **2. Anti-Chop Confidenza (0.42 / 1.30):**
   * `P(dir) >= 0.42`, `Ratio >= 1.30`.
   * Esige che il modello abbia forte convinzione direzionale ($P \ge 42\%$) e un distacco del 30% rispetto all'opposto, tagliando i trade di rumore e proteggendo il capitale.
3. **3. Anti-Chop Ibrido (Confidence + Volatility Gate):**
   * `P(dir) >= 0.42`, `Ratio >= 1.30` + `norm_atr >= 0.010` (1.0% min).
4. **4. Anti-Chop Sniper (High-Conviction Breakout):**
   * `P(dir) >= 0.45`, `Ratio >= 1.35` + `norm_atr >= 0.011` (1.1% min).

---

### 📅 Valutazione sui 3 Macro-Periodi Storici:
* **Periodo 1 (2022 Bear):** Test di resistenza sul crollo di BTC da 48k a 16k.
* **Periodo 2 (2024 Bull):** Test di cattura del rally da 40k a 95k+.
* **Periodo 3 (2025-2026 Recente):** Valutazione sulle condizioni contemporanee.
""")

# =====================================================================
# Cell 2: Dependencies
# =====================================================================
add_code("""# 1. Installazione dipendenze minime
!pip install ccxt pyyaml --quiet
print("Ambiente configurato con successo!")
""")

# =====================================================================
# Cell 3: Imports & Global Configuration
# =====================================================================
add_code("""# 2. Import delle librerie e configurazione base
import os
import glob
import time
import json
import pickle
import logging
from datetime import datetime, timedelta, timezone
from typing import List, Tuple, Dict, Any, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import classification_report, f1_score, precision_score, confusion_matrix

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)

BASE_CONFIG = {
    "symbol": "BTC/USDT",
    "timeframe": "4h",
    "lookback_window": 64,         # 64 barre 4H (~10.6 giorni)
    "prediction_horizon": 6,       # 6 barre = 24 ore di orizzonte
    "taker_fee": 0.0005,           # 0.05% taker (0.10% round-trip)
    "embargo_bars": 12,            # 12 barre = 48 ore anti-leakage ai confini
    "batch_size": 64,
    "channels": [32, 32, 48, 48, 64], # ~55k parametri
    "kernel_size": 3,
    "max_epochs": 30,
    "patience": 8
}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Dispositivo di calcolo attivo: {DEVICE}")
if torch.cuda.is_available():
    print(f"GPU rilevata: {torch.cuda.get_device_name(0)}")
""")

# =====================================================================
# Cell 4: Data Loader
# =====================================================================
add_code("""# 3. Caricamento dati storici OHLCV
def load_market_data():
    search_dirs = ["/kaggle/input", "data", "."]
    for d in search_dirs:
        if os.path.exists(d):
            for root, _, files in os.walk(d):
                for f in files:
                    if f.endswith(".parquet") and ("BTC" in f or "btc" in f):
                        file_path = os.path.join(root, f)
                        print(f"Caricamento dataset: {file_path}")
                        df = pd.read_parquet(file_path)
                        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
                        print(f"Caricate {len(df)} barre da {df['timestamp'].iloc[0]} a {df['timestamp'].iloc[-1]}")
                        return df

    for d in search_dirs:
        if os.path.exists(d):
            for root, _, files in os.walk(d):
                for f in files:
                    if f.endswith(".parquet"):
                        file_path = os.path.join(root, f)
                        print(f"Caricamento dataset: {file_path}")
                        df = pd.read_parquet(file_path)
                        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
                        return df

    try:
        import ccxt
        exchange = ccxt.binance({"enableRateLimit": True, "options": {"defaultType": "future"}})
        exchange.load_markets()
        symbol = BASE_CONFIG["symbol"]
        target = f"{symbol}:USDT" if f"{symbol}:USDT" in exchange.markets else symbol
        ohlcv = exchange.fetch_ohlcv(target, timeframe=BASE_CONFIG["timeframe"], limit=1000)
        df = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        for c in ["open", "high", "low", "close", "volume"]:
            df[c] = pd.to_numeric(df[c])
        df["funding_rate"] = 0.0
        return df
    except Exception as e:
        raise RuntimeError("Carica il file BTC_USDT_4h.parquet negli input di Kaggle!") from e

df_raw = load_market_data()
print("Distribuzione annuale barre:")
print(df_raw["timestamp"].dt.year.value_counts().sort_index())
df_raw.head(3)
""")

# =====================================================================
# Cell 5: Features & Scaling
# =====================================================================
add_code("""# 4. Feature Engineering Stazionaria & Scaler Anti-Leakage
FEATURE_COLUMNS = [
    "ret_close", "ret_high", "ret_low", "hl_spread",
    "norm_atr", "vol_vs_sma", "ret_volume",
    "dist_sma_50", "dist_sma_200", "dist_sma_50_200", "funding_rate"
]

def calculate_atr(df, period=14):
    high, low, close = df["high"], df["low"], df["close"]
    prev_close = close.shift(1)
    tr = pd.concat([high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    return tr.rolling(window=period, min_periods=period).mean()

def extract_features(df):
    df = df.copy()
    df["ret_close"] = np.log(df["close"] / df["close"].shift(1))
    df["ret_high"] = np.log(df["high"] / df["high"].shift(1))
    df["ret_low"] = np.log(df["low"] / df["low"].shift(1))
    df["hl_spread"] = np.log(df["high"] / df["low"])
    df["norm_atr"] = calculate_atr(df, 14) / df["close"]
    vol_sma = df["volume"].rolling(20, min_periods=20).mean()
    df["vol_vs_sma"] = (df["volume"] - vol_sma) / (vol_sma + 1e-8)
    df["ret_volume"] = np.log((df["volume"] + 1.0) / (df["volume"].shift(1) + 1.0))
    sma50 = df["close"].rolling(50, min_periods=50).mean()
    sma200 = df["close"].rolling(200, min_periods=200).mean()
    df["dist_sma_50"] = (df["close"] - sma50) / (sma50 + 1e-8)
    df["dist_sma_200"] = (df["close"] - sma200) / (sma200 + 1e-8)
    df["dist_sma_50_200"] = (sma50 - sma200) / (sma200 + 1e-8)
    if "funding_rate" not in df.columns:
        df["funding_rate"] = 0.0
    else:
        df["funding_rate"] = df["funding_rate"].fillna(0.0)
    df.dropna(subset=FEATURE_COLUMNS, inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df

class FeatureScaler:
    def __init__(self):
        self.scaler = RobustScaler()
        self.clip_bounds = {}
        self.feature_names = FEATURE_COLUMNS
        
    def fit(self, df):
        X = df[self.feature_names].copy()
        for col in self.feature_names:
            self.clip_bounds[col] = (X[col].quantile(0.01), X[col].quantile(0.99))
            X[col] = X[col].clip(lower=self.clip_bounds[col][0], upper=self.clip_bounds[col][1])
        self.scaler.fit(X.values)
        return self

    def transform(self, df):
        X = df[self.feature_names].copy()
        for col in self.feature_names:
            if col in self.clip_bounds:
                X[col] = X[col].clip(lower=self.clip_bounds[col][0], upper=self.clip_bounds[col][1])
        return self.scaler.transform(X.values)

df_features = extract_features(df_raw)
print(f"Feature estratte: {len(df_features)} barre stazionarie.")
print(f"Statistiche Volatilità norm_atr: Mediana={df_features['norm_atr'].median()*100:.2f}% | 25% Quantile={df_features['norm_atr'].quantile(0.25)*100:.2f}% | 10% Quantile={df_features['norm_atr'].quantile(0.10)*100:.2f}%")
""")

# =====================================================================
# Cell 6: Triple Barrier Dynamic ATR Labeling
# =====================================================================
add_code("""# 5. Triple Barrier Dynamic ATR Labeling (Architettura Vincente)
def apply_triple_barrier(
    df,
    horizon=6,
    atr_mult_tp=2.2,
    atr_mult_sl=1.4,
    taker_fee=0.0005
):
    n = len(df)
    labels = np.zeros(n, dtype=int)
    closes = df["close"].values
    highs = df["high"].values
    lows = df["low"].values
    atrs = df["norm_atr"].values
    fee = 2 * taker_fee

    for t in range(n - horizon):
        p0 = closes[t]
        target_tp = max(0.015, atrs[t] * atr_mult_tp)
        target_sl = max(0.010, atrs[t] * atr_mult_sl)

        long_tp = p0 * (1.0 + target_tp + fee)
        long_sl = p0 * (1.0 - target_sl - fee)
        short_tp = p0 * (1.0 - target_tp - fee)
        short_sl = p0 * (1.0 + target_sl + fee)

        l_hit, s_hit = None, None

        for k in range(1, horizon + 1):
            h, l = highs[t + k], lows[t + k]
            if h >= long_tp and l <= long_sl:
                break
            elif l <= long_sl:
                break
            elif h >= long_tp:
                l_hit = k
                break

        for k in range(1, horizon + 1):
            h, l = highs[t + k], lows[t + k]
            if l <= short_tp and h >= short_sl:
                break
            elif h >= short_sl:
                break
            elif l <= short_tp:
                s_hit = k
                break

        if l_hit and s_hit:
            labels[t] = 1 if l_hit < s_hit else (2 if s_hit < l_hit else 0)
        elif l_hit:
            labels[t] = 1
        elif s_hit:
            labels[t] = 2
        else:
            labels[t] = 0

    df_out = df.copy()
    df_out["label"] = labels
    return df_out.iloc[:-horizon].copy().reset_index(drop=True)

print("Generazione dataset con label dinamiche ATR (2.2 TP / 1.4 SL)...")
df_labeled_atr = apply_triple_barrier(df_features, horizon=6, atr_mult_tp=2.2, atr_mult_sl=1.4)
print(f"Distribuzione label: Neutral={np.mean(df_labeled_atr['label']==0)*100:.1f}% | Long={np.mean(df_labeled_atr['label']==1)*100:.1f}% | Short={np.mean(df_labeled_atr['label']==2)*100:.1f}%")
""")

# =====================================================================
# Cell 7: Causal TCN Architecture
# =====================================================================
add_code("""# 6. Architettura TCN Causale
class CausalConv1d(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, dilation=1):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size, stride=1, padding=0, dilation=dilation)

    def forward(self, x):
        return self.conv(F.pad(x, (self.padding, 0)))

class TemporalResidualBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, dilation, dropout=0.30):
        super().__init__()
        self.conv1 = CausalConv1d(in_ch, out_ch, kernel_size, dilation)
        self.norm1 = nn.LayerNorm(out_ch)
        self.act1 = nn.GELU()
        self.drop1 = nn.Dropout(dropout)

        self.conv2 = CausalConv1d(out_ch, out_ch, kernel_size, dilation)
        self.norm2 = nn.LayerNorm(out_ch)
        self.act2 = nn.GELU()
        self.drop2 = nn.Dropout(dropout)

        self.res = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        self.final_act = nn.GELU()

    def _apply_norm(self, x, norm):
        return norm(x.transpose(1, 2)).transpose(1, 2)

    def forward(self, x):
        res = self.res(x)
        out = self.drop1(self.act1(self._apply_norm(self.conv1(x), self.norm1)))
        out = self.drop2(self.act2(self._apply_norm(self.conv2(out), self.norm2)))
        return self.final_act(out + res)

class TemporalConvolutionalNetwork(nn.Module):
    def __init__(self, num_features, channels=[32, 32, 48, 48, 64], kernel_size=3, dropout=0.30, num_classes=3):
        super().__init__()
        layers = []
        in_ch = num_features
        for i, out_ch in enumerate(channels):
            dilation = 2 ** i
            layers.append(TemporalResidualBlock(in_ch, out_ch, kernel_size, dilation, dropout))
            in_ch = out_ch
        self.network = nn.Sequential(*layers)
        
        last_ch = channels[-1]
        self.head = nn.Sequential(
            nn.Linear(last_ch, last_ch // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(last_ch // 2, num_classes)
        )

    def forward(self, x):
        feats = self.network(x)
        return self.head(feats[:, :, -1])

    def predict_probabilities(self, x):
        self.eval()
        with torch.no_grad():
            return F.softmax(self.forward(x), dim=-1)
""")

# =====================================================================
# Cell 8: Split Engine, Trainer & Anti-Chop Backtester
# =====================================================================
add_code("""# 7. Split Engine, Training Loop & Simulatore Anti-Chop

def extract_windows_from_segment(features, labels, lookback=64):
    n = len(features)
    if n < lookback:
        return np.empty((0, features.shape[1], lookback), dtype=np.float32), np.empty((0,), dtype=np.int64)
    X_list, y_list = [], []
    for i in range(n - lookback + 1):
        X_list.append(features[i : i + lookback].T)
        if labels is not None:
            y_list.append(labels[i + lookback - 1])
    return np.array(X_list, dtype=np.float32), (np.array(y_list, dtype=np.int64) if labels is not None else None)

def prepare_period_splits(df_labeled, test_years, lookback=64, embargo=12, val_ratio=0.15, batch_size=64):
    test_idx = df_labeled.index[df_labeled["timestamp"].dt.year.isin(test_years)].tolist()
    t_start, t_end = test_idx[0], test_idx[-1]

    seg_before = (0, max(0, t_start - embargo)) if t_start > embargo else None
    seg_after = (t_end + embargo + 1, len(df_labeled)) if (t_end + embargo + 1 < len(df_labeled)) else None

    segments = []
    if seg_before and seg_before[1] > seg_before[0]:
        segments.append(df_labeled.iloc[seg_before[0]:seg_before[1]])
    if seg_after and seg_after[1] > seg_after[0]:
        segments.append(df_labeled.iloc[seg_after[0]:seg_after[1]])

    train_pool_df = pd.concat(segments, axis=0).reset_index(drop=True)
    scaler = FeatureScaler().fit(train_pool_df)

    total_pool_len = sum(len(s) for s in segments)
    n_val_target = int(total_pool_len * val_ratio)

    primary_val_seg = segments[-1]
    val_cutoff = max(lookback + embargo, len(primary_val_seg) - n_val_target)

    train_dfs = segments[:-1] + [primary_val_seg.iloc[:val_cutoff - embargo]]
    val_dfs = [primary_val_seg.iloc[val_cutoff:]]

    train_X_list, train_y_list = [], []
    val_X_list, val_y_list = [], []

    for t_df in train_dfs:
        feat = scaler.transform(t_df)
        lab = t_df["label"].values
        X, y = extract_windows_from_segment(feat, lab, lookback)
        train_X_list.append(X)
        train_y_list.append(y)

    for v_df in val_dfs:
        feat = scaler.transform(v_df)
        lab = v_df["label"].values
        X, y = extract_windows_from_segment(feat, lab, lookback)
        val_X_list.append(X)
        val_y_list.append(y)

    train_X = np.concatenate(train_X_list, axis=0)
    train_y = np.concatenate(train_y_list, axis=0)
    val_X = np.concatenate(val_X_list, axis=0)
    val_y = np.concatenate(val_y_list, axis=0)

    test_slice_start = max(0, t_start - lookback + 1)
    test_slice_df = df_labeled.iloc[test_slice_start : t_end + 1].copy().reset_index(drop=True)
    test_feat = scaler.transform(test_slice_df)
    test_X, test_y = extract_windows_from_segment(test_feat, test_slice_df["label"].values, lookback)

    test_bars = df_labeled.iloc[t_start : t_end + 1].copy().reset_index(drop=True)
    assert len(test_X) == len(test_bars), f"Test mismatch: {len(test_X)} windows vs {len(test_bars)} barre"

    train_loader = DataLoader(TensorDataset(torch.tensor(train_X), torch.tensor(train_y)), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(torch.tensor(val_X), torch.tensor(val_y)), batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(TensorDataset(torch.tensor(test_X), torch.tensor(test_y)), batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader, test_bars, scaler, (len(train_X), len(val_X), len(test_X))

def train_tcn_model(train_loader, val_loader, train_y, lr=3e-4, dropout=0.30, weight_decay=0.01, max_epochs=30, patience=8, seed=42):
    set_seed(seed)
    model = TemporalConvolutionalNetwork(
        num_features=len(FEATURE_COLUMNS),
        channels=BASE_CONFIG["channels"],
        kernel_size=BASE_CONFIG["kernel_size"],
        dropout=dropout
    ).to(DEVICE)

    class_counts = np.bincount(train_y, minlength=3)
    weights = len(train_y) / (3.0 * np.maximum(class_counts, 1.0))
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32).to(DEVICE))
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)

    best_val_loss = float("inf")
    best_weights = None
    best_val_f1 = 0.0
    patience_cnt = 0

    for ep in range(1, max_epochs + 1):
        model.train()
        tot_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            tot_loss += loss.item() * len(yb)

        model.eval()
        v_loss, preds_all, targets_all = 0.0, [], []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                out = model(xb)
                v_loss += criterion(out, yb).item() * len(yb)
                preds_all.extend(torch.argmax(out, dim=-1).cpu().numpy())
                targets_all.extend(yb.cpu().numpy())

        v_loss /= len(val_loader.dataset)
        val_f1 = f1_score(targets_all, preds_all, average="macro", zero_division=0)
        scheduler.step(v_loss)

        if v_loss < best_val_loss:
            best_val_loss = v_loss
            best_val_f1 = val_f1
            best_weights = model.state_dict().copy()
            patience_cnt = 0
        else:
            patience_cnt += 1
            if patience_cnt >= patience:
                break

    model.load_state_dict(best_weights)
    return model, best_val_loss, best_val_f1

def predict_probabilities(model, loader):
    model.eval()
    all_probs = []
    with torch.no_grad():
        for xb, _ in loader:
            xb = xb.to(DEVICE)
            all_probs.append(model.predict_probabilities(xb).cpu().numpy())
    return np.concatenate(all_probs, axis=0)

def run_simulation(
    test_bars,
    probs,
    prob_threshold=0.38,
    ratio_threshold=1.25,
    min_norm_atr=0.0,
    atr_mult_tp=2.2,
    atr_mult_sl=1.4,
    capital_allocation_pct=0.10,
    initial_capital=1000.0,
    taker_fee=0.0005,
    max_holding=6
):
    capital = initial_capital
    equity_curve = [capital]
    active_trade = None
    trades = []
    filtered_signals_count = 0

    for t in range(len(test_bars) - 1):
        curr_bar = test_bars.iloc[t]
        next_bar = test_bars.iloc[t + 1]

        # 1. Gestione trade aperto
        if active_trade is not None:
            active_trade["duration"] += 1
            h, l, c = curr_bar["high"], curr_bar["low"], curr_bar["close"]
            entry = active_trade["entry"]
            exit_p, reason = None, None

            tp_dist = entry * active_trade["tp_target_pct"]
            sl_dist = entry * active_trade["sl_target_pct"]
            curr_tp = entry + tp_dist if active_trade["dir"] == "LONG" else entry - tp_dist
            curr_sl = entry - sl_dist if active_trade["dir"] == "LONG" else entry + sl_dist

            if active_trade["dir"] == "LONG":
                if h >= curr_tp and l <= curr_sl:
                    exit_p, reason = curr_sl, "SL"
                elif l <= curr_sl:
                    exit_p, reason = curr_sl, "SL"
                elif h >= curr_tp:
                    exit_p, reason = curr_tp, "TP"
                elif active_trade["duration"] >= max_holding:
                    exit_p, reason = c, "TIME"
            else:
                if l <= curr_tp and h >= curr_sl:
                    exit_p, reason = curr_sl, "SL"
                elif h >= curr_sl:
                    exit_p, reason = curr_sl, "SL"
                elif l <= curr_tp:
                    exit_p, reason = curr_tp, "TP"
                elif active_trade["duration"] >= max_holding:
                    exit_p, reason = c, "TIME"

            if exit_p is not None:
                gross_ret = (exit_p - entry)/entry if active_trade["dir"] == "LONG" else (entry - exit_p)/entry
                net_ret = gross_ret - (2 * taker_fee)
                pnl = active_trade["size"] * net_ret
                capital += pnl
                trades.append({
                    "pnl": pnl, "net_ret": net_ret, "reason": reason, "dir": active_trade["dir"]
                })
                active_trade = None

        # 2. Apertura nuovo trade con Filtro Anti-Chop
        if active_trade is None and t < len(probs):
            p_neu, p_l, p_s = probs[t]
            atr_val = curr_bar["norm_atr"]

            volatility_ok = (atr_val >= min_norm_atr)

            signal = None
            raw_signal = None
            if p_l >= prob_threshold and p_l >= ratio_threshold * p_s:
                raw_signal = "LONG"
            elif p_s >= prob_threshold and p_s >= ratio_threshold * p_l:
                raw_signal = "SHORT"

            if raw_signal is not None:
                if volatility_ok:
                    signal = raw_signal
                else:
                    filtered_signals_count += 1

            if signal is not None:
                trade_size = max(50.0, capital * capital_allocation_pct)
                active_trade = {
                    "dir": signal,
                    "entry": next_bar["open"],
                    "duration": 0,
                    "size": trade_size,
                    "tp_target_pct": max(0.015, atr_val * atr_mult_tp),
                    "sl_target_pct": max(0.010, atr_val * atr_mult_sl)
                }

        equity_curve.append(capital)

    eq_series = pd.Series(equity_curve)
    drawdowns = (eq_series - eq_series.cummax()) / eq_series.cummax() * 100
    pnls = [t["pnl"] for t in trades]
    gains = [p for p in pnls if p > 0]
    losses = [abs(p) for p in pnls if p < 0]

    win_rate = (len(gains) / len(pnls) * 100) if pnls else 0.0
    profit_factor = (sum(gains) / sum(losses)) if sum(losses) > 0 else (99.0 if sum(gains) > 0 else 0.0)
    total_ret = ((capital - initial_capital) / initial_capital) * 100.0

    return {
        "total_return_pct": round(total_ret, 2),
        "max_drawdown_pct": round(abs(drawdowns.min()), 2),
        "win_rate_pct": round(win_rate, 1),
        "profit_factor": round(profit_factor, 2),
        "total_trades": len(trades),
        "filtered_trades": filtered_signals_count,
        "equity_curve": equity_curve,
        "trades": trades
    }

# Le 4 Configurazioni di Test
ANTI_CHOP_CONFIGS = [
    {
        "key": "v1_std",
        "name": "1. Standard (Controllo: 0.38 / 1.25)",
        "p_thresh": 0.38,
        "r_thresh": 1.25,
        "min_atr": 0.0,
        "color": "#0077b6"
    },
    {
        "key": "v2_conf",
        "name": "2. Anti-Chop Confidenza (0.42 / 1.30)",
        "p_thresh": 0.42,
        "r_thresh": 1.30,
        "min_atr": 0.0,
        "color": "#ffaa00"
    },
    {
        "key": "v3_hybrid",
        "name": "3. Anti-Chop Ibrido (0.42 / 1.30 + ATR>=1.0%)",
        "p_thresh": 0.42,
        "r_thresh": 1.30,
        "min_atr": 0.010,
        "color": "#00a86b"
    },
    {
        "key": "v4_sniper",
        "name": "4. Anti-Chop Sniper (0.45 / 1.35 + ATR>=1.1%)",
        "p_thresh": 0.45,
        "r_thresh": 1.35,
        "min_atr": 0.011,
        "color": "#d90429"
    }
]

def evaluate_period_antichop(
    period_name,
    test_years,
    df_labeled_atr,
    lr=3e-4,
    dropout=0.30,
    weight_decay=0.01,
    max_epochs=30,
    patience=8
):
    print("\\n" + "=" * 80)
    print(f"🚀 AVVIO VALUTAZIONE: {period_name}")
    print(f"   Anni di Test: {test_years} | Embargo: {BASE_CONFIG['embargo_bars']} barre | Lookback: {BASE_CONFIG['lookback_window']} barre")
    print("=" * 80)

    tr_loader, val_loader, test_loader, test_bars, scaler, counts = prepare_period_splits(
        df_labeled_atr, test_years,
        lookback=BASE_CONFIG["lookback_window"],
        embargo=BASE_CONFIG["embargo_bars"],
        batch_size=BASE_CONFIG["batch_size"]
    )
    print(f"   Dati: {counts[0]} train windows, {counts[1]} val windows, {counts[2]} test bars")

    train_y = []
    for _, yb in tr_loader:
        train_y.extend(yb.numpy())
    train_y = np.array(train_y)

    print("   Addestramento Rete TCN Dynamic ATR...")
    model, val_loss, val_f1 = train_tcn_model(
        tr_loader, val_loader, train_y,
        lr=lr, dropout=dropout, weight_decay=weight_decay,
        max_epochs=max_epochs, patience=patience, seed=42
    )
    print(f"   Modello addestrato -> Val Loss: {val_loss:.4f} | Val F1: {val_f1:.4f}")

    print("   Calcolo distribuzioni di probabilità Softmax sul test set...")
    probs = predict_probabilities(model, test_loader)

    results = {}
    summary_rows = []

    for cfg in ANTI_CHOP_CONFIGS:
        res = run_simulation(
            test_bars, probs,
            prob_threshold=cfg["p_thresh"],
            ratio_threshold=cfg["r_thresh"],
            min_norm_atr=cfg["min_atr"],
            capital_allocation_pct=0.10
        )
        results[cfg["key"]] = res
        summary_rows.append({
            "Configurazione": cfg["name"],
            "Rendimento %": f"{res['total_return_pct']:+.2f}%",
            "Max Drawdown %": f"{res['max_drawdown_pct']:.2f}%",
            "Win Rate %": f"{res['win_rate_pct']:.1f}%",
            "Profit Factor": res["profit_factor"],
            "Trade Totali": res["total_trades"],
            "Trade Filtrati": res["filtered_trades"]
        })
        print(f"   -> {cfg['name']}: Ret={res['total_return_pct']:+.2f}% | PF={res['profit_factor']:.2f} | WR={res['win_rate_pct']:.1f}% | DD={res['max_drawdown_pct']:.2f}% | Trades={res['total_trades']} (Filtrati: {res['filtered_trades']})")

    df_summary = pd.DataFrame(summary_rows)
    print(f"\\n--- TABELLA: {period_name} ---")
    print(df_summary.to_string(index=False))
    print("=" * 80 + "\\n")

    return {
        "period_name": period_name,
        "test_years": test_years,
        "results": results,
        "model": model,
        "scaler": scaler,
        "test_bars": test_bars
    }
""")

# =====================================================================
# Cell 9: Period 1: 2022 Bear Market
# =====================================================================
add_code("""# =====================================================================
# 📉 PERIODO 1: 2022 BEAR MARKET (Crollo BTC da 48k a 16k)
# =====================================================================

res_period1 = evaluate_period_antichop(
    period_name="Periodo 1: 2022 (Bear Market)",
    test_years=[2022],
    df_labeled_atr=df_labeled_atr,
    lr=3e-4,
    dropout=0.30,
    weight_decay=0.01,
    max_epochs=BASE_CONFIG["max_epochs"],
    patience=BASE_CONFIG["patience"]
)
""")

# =====================================================================
# Cell 10: Period 2: 2024 Bull Market
# =====================================================================
add_code("""# =====================================================================
# 📈 PERIODO 2: 2024 BULL MARKET (Rally da 40k a 95k+)
# =====================================================================

res_period2 = evaluate_period_antichop(
    period_name="Periodo 2: 2024 (Bull Market)",
    test_years=[2024],
    df_labeled_atr=df_labeled_atr,
    lr=3e-4,
    dropout=0.30,
    weight_decay=0.01,
    max_epochs=BASE_CONFIG["max_epochs"],
    patience=BASE_CONFIG["patience"]
)
""")

# =====================================================================
# Cell 11: Period 3: 2025-2026 Recent Market
# =====================================================================
add_code("""# =====================================================================
# ⚡ PERIODO 3: 2025-2026 RECENT MARKET (Condizioni Contemporanee)
# =====================================================================

res_period3 = evaluate_period_antichop(
    period_name="Periodo 3: 2025-2026 (Recent Market)",
    test_years=[2025, 2026],
    df_labeled_atr=df_labeled_atr,
    lr=3e-4,
    dropout=0.30,
    weight_decay=0.01,
    max_epochs=BASE_CONFIG["max_epochs"],
    patience=BASE_CONFIG["patience"]
)
""")

# =====================================================================
# Cell 12: Macro Benchmark Cross-Periodi, Grafico Multi-Pannello & Export
# =====================================================================
add_code("""# =====================================================================
# 📊 CONFRONTO BENCHMARK CROSS-PERIODI & EXPORT CHAMPION MODEL
# =====================================================================

periods_data = [
    ("2022 (Bear)", res_period1),
    ("2024 (Bull)", res_period2),
    ("2025-26 (Recent)", res_period3)
]

matrix_rows = []
for cfg in ANTI_CHOP_CONFIGS:
    k = cfg["key"]
    name = cfg["name"]

    r1 = res_period1["results"][k]
    r2 = res_period2["results"][k]
    r3 = res_period3["results"][k]

    ret1, pf1 = r1["total_return_pct"], r1["profit_factor"]
    ret2, pf2 = r2["total_return_pct"], r2["profit_factor"]
    ret3, pf3 = r3["total_return_pct"], r3["profit_factor"]

    avg_ret = np.mean([ret1, ret2, ret3])
    cum_ret = ((1 + ret1/100) * (1 + ret2/100) * (1 + ret3/100) - 1) * 100
    worst_dd = max(r1["max_drawdown_pct"], r2["max_drawdown_pct"], r3["max_drawdown_pct"])
    avg_wr = np.mean([r1["win_rate_pct"], r2["win_rate_pct"], r3["win_rate_pct"]])
    avg_pf = np.mean([pf1, pf2, pf3])
    total_trades = r1["total_trades"] + r2["total_trades"] + r3["total_trades"]

    matrix_rows.append({
        "Configurazione": name,
        "2022 Bear (Ret / PF)": f"{ret1:+.2f}% / {pf1:.2f}",
        "2024 Bull (Ret / PF)": f"{ret2:+.2f}% / {pf2:.2f}",
        "2025-26 Recente (Ret / PF)": f"{ret3:+.2f}% / {pf3:.2f}",
        "Rendimento Medio %": f"{avg_ret:+.2f}%",
        "Rendimento Cumulato %": f"{cum_ret:+.2f}%",
        "Max DD Peggiore %": f"{worst_dd:.2f}%",
        "Win Rate Medio %": f"{avg_wr:.1f}%",
        "Profit Factor Medio": f"{avg_pf:.2f}",
        "Trade Totali": total_trades
    })

matrix_df = pd.DataFrame(matrix_rows)

print("\\n" + "=" * 100)
print("🏆 SUPER-TABELLA COMPARATIVA CROSS-PERIODO")
print("=" * 100)
print(matrix_df.to_string(index=False))
print("=" * 100 + "\\n")

# Grafico Multi-Pannello a 4 riquadri (2x2)
fig, axes = plt.subplots(2, 2, figsize=(16, 10))

# Subplot 1: 2022 Bear Market
ax1 = axes[0, 0]
for cfg in ANTI_CHOP_CONFIGS:
    k = cfg["key"]
    res = res_period1["results"][k]
    ax1.plot(res["equity_curve"], label=f"{cfg['name']} ({res['total_return_pct']:+.1f}%)", color=cfg["color"], lw=1.8)
ax1.set_title("Periodo 1: 2022 (Bear Market - Crollo -65%)", fontweight="bold")
ax1.set_ylabel("Equity ($)")
ax1.set_xlabel("Barre 4H")
ax1.grid(True, alpha=0.3)
ax1.legend(loc="upper left")

# Subplot 2: 2024 Bull Market
ax2 = axes[0, 1]
for cfg in ANTI_CHOP_CONFIGS:
    k = cfg["key"]
    res = res_period2["results"][k]
    ax2.plot(res["equity_curve"], label=f"{cfg['name']} ({res['total_return_pct']:+.1f}%)", color=cfg["color"], lw=1.8)
ax2.set_title("Periodo 2: 2024 (Bull Market - Rally +140%)", fontweight="bold")
ax2.set_ylabel("Equity ($)")
ax2.set_xlabel("Barre 4H")
ax2.grid(True, alpha=0.3)
ax2.legend(loc="upper left")

# Subplot 3: 2025-2026 Recent Market
ax3 = axes[1, 0]
for cfg in ANTI_CHOP_CONFIGS:
    k = cfg["key"]
    res = res_period3["results"][k]
    ax3.plot(res["equity_curve"], label=f"{cfg['name']} ({res['total_return_pct']:+.1f}%)", color=cfg["color"], lw=1.8)
ax3.set_title("Periodo 3: 2025-2026 (Recent Market)", fontweight="bold")
ax3.set_ylabel("Equity ($)")
ax3.set_xlabel("Barre 4H")
ax3.grid(True, alpha=0.3)
ax3.legend(loc="upper left")

# Subplot 4: Bar Chart Comparativo Rendimenti
ax4 = axes[1, 1]
x_indices = np.arange(len(periods_data))
bar_width = 0.20

for i, cfg in enumerate(ANTI_CHOP_CONFIGS):
    k = cfg["key"]
    returns = [p_res["results"][k]["total_return_pct"] for _, p_res in periods_data]
    ax4.bar(x_indices + (i - 1.5) * bar_width, returns, width=bar_width, label=cfg["name"], color=cfg["color"], alpha=0.9)

ax4.axhline(0, color="black", linestyle="--", lw=1)
ax4.set_xticks(x_indices)
ax4.set_xticklabels([p_label for p_label, _ in periods_data], fontweight="bold")
ax4.set_ylabel("Rendimento Netto % (Post-Fee)")
ax4.set_title("Confronto Rendimenti nei 3 Regimi", fontweight="bold")
ax4.grid(True, alpha=0.3, axis="y")
ax4.legend(loc="best")

plt.tight_layout()
chart_path = "/kaggle/working/antichop_comparison.png" if os.path.exists("/kaggle/working") else "models/antichop_comparison.png"
os.makedirs(os.path.dirname(chart_path), exist_ok=True)
plt.savefig(chart_path, dpi=200)
plt.show()

# =====================================================================
# Salvataggio Modello Campione e Scaler per Live Agent
# =====================================================================
single_save_path = "/kaggle/working/tcn_best_model.pth" if os.path.exists("/kaggle/working") else "models/tcn_best_model.pth"
scaler_save_path = "/kaggle/working/scaler.pkl" if os.path.exists("/kaggle/working") else "models/scaler.pkl"

torch.save(res_period3["model"].state_dict(), single_save_path)
with open(scaler_save_path, "wb") as f:
    pickle.dump(res_period3["scaler"], f)

print(f"✅ Modello TCN Champion salvato in:  {single_save_path}")
print(f"✅ Scaler Campione salvato in:      {scaler_save_path}")
print(f"✅ Grafico Comparativo salvato:     {chart_path}")

best_row = max(matrix_rows, key=lambda r: float(r['Rendimento Cumulato %'].replace('%', '').replace('+', '')))
print(f\"\"\"
--------------------------------------------------------------------------------
🏆 CONFIGURAZIONE CAMPIONE: {best_row['Configurazione']}
   - Rendimento Cumulato: {best_row['Rendimento Cumulato %']}
   - Profit Factor Medio: {best_row['Profit Factor Medio']}
   - Max Drawdown:        {best_row['Max DD Peggiore %']}
--------------------------------------------------------------------------------
\"\"\")
""")

# Scrittura del notebook finale
os.makedirs("notebooks", exist_ok=True)
with open("notebooks/kaggle_train.ipynb", "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=2)

print("Notebook ripristinato con successo alla versione stabile in notebooks/kaggle_train.ipynb!")
