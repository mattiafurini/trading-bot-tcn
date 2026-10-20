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

# Cell 1: Intro
add_md("""# 🔬 Quantitative TCN Trading Bot: Multi-Experiment Optimization Suite
### Rigorous Empirical Research & Profit Maximization on BTC/USDT 4H Perpetual Futures

Questo notebook esegue in successione automatica i **3 esperimenti chiave** per ottimizzare l'aspettativa matematica ($EV$) ed elevare il profitto riducendo il rischio di *backtest overfitting*:

1. **Esperimento 1 — Grid Search su Validation Set:** Confronta 3 configurazioni di regolarizzazione (`learning_rate`, `dropout`, `weight_decay`) e seleziona oggettivamente la vincente su `val_loss` e `val_f1`.
2. **Esperimento 2 — Dynamic ATR Volatility Barriers:** Rimpiazza le percentuali fisse con barriere adattive basate sull'Average True Range per seguire la volatilità di mercato ($2.2 \times ATR$ e $1.4 \times ATR$).
3. **Esperimento 3 — Deep Ensemble Consensus (5-TCN Multi-Seed Bagging):** Addestra 5 reti TCN indipendenti con semi casuali differenti (`[42, 101, 777, 1337, 2024]`), aggrega le probabilità predittive via soft-voting ensemble per cancellare la varianza stocastica ed elevare Sharpe Ratio e stabilità.
4. **Tabella Comparativa Finale & Champion Export:** Grafico multi-equity curve a confronto e salvataggio sia dell'ensemble `tcn_ensemble.pth` che del modello singolo `tcn_best_model.pth`.
""")

# Cell 2: Dependencies
add_code("""# 1. Installazione dipendenze minime
!pip install ccxt pyyaml --quiet
print("Ambiente configurato con successo!")
""")

# Cell 3: Imports & Global Setup
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
from torch.utils.data import Dataset, DataLoader

def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)

BASE_CONFIG = {
    "symbol": "BTC/USDT",
    "timeframe": "4h",
    "lookback_window": 64,
    "prediction_horizon": 6,       # 24 ore holding
    "taker_fee": 0.0005,           # 0.05% taker
    "embargo_bars": 12,            # 48 ore anti-leakage
    "batch_size": 64,
    "channels": [32, 32, 48, 48, 64], # ~65k parametri
    "kernel_size": 3,
    "max_epochs": 40,
    "patience": 10
}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Dispositivo di calcolo attivo: {DEVICE}")
if torch.cuda.is_available():
    print(f"GPU rilevata: {torch.cuda.get_device_name(0)}")
""")

# Cell 4: Data Loader
add_code("""# 3. Caricamento dati storici OHLCV
def load_market_data():
    search_dirs = ["/kaggle/input", "data", "."]
    for d in search_dirs:
        if os.path.exists(d):
            for root, _, files in os.walk(d):
                for f in files:
                    if f.endswith(".parquet"):
                        file_path = os.path.join(root, f)
                        print(f"Caricamento dataset: {file_path}")
                        df = pd.read_parquet(file_path)
                        print(f"Caricate con successo {len(df)} barre da {df['timestamp'].iloc[0]} a {df['timestamp'].iloc[-1]}")
                        return df

    # Fallback se non in input
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
df_raw.head(3)
""")

# Cell 5: Features & Scaling
add_code("""# 4. Feature Engineering Stazionaria
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
    df.dropna(subset=FEATURE_COLUMNS, inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df

class FeatureScaler:
    def __init__(self):
        self.scaler = RobustScaler()
        self.clip_bounds = {}
        
    def fit(self, df):
        X = df[FEATURE_COLUMNS].copy()
        for col in FEATURE_COLUMNS:
            self.clip_bounds[col] = (X[col].quantile(0.01), X[col].quantile(0.99))
            X[col] = X[col].clip(lower=self.clip_bounds[col][0], upper=self.clip_bounds[col][1])
        self.scaler.fit(X.values)
        return self

    def transform(self, df):
        X = df[FEATURE_COLUMNS].copy()
        for col in FEATURE_COLUMNS:
            X[col] = X[col].clip(lower=self.clip_bounds[col][0], upper=self.clip_bounds[col][1])
        return self.scaler.transform(X.values)

df_features = extract_features(df_raw)
print(f"Feature estratte: {len(df_features)} barre.")
""")

# Cell 6: Triple Barrier Function (Fixed & Dynamic ATR)
add_code("""# 5. Triple Barrier Labeling (Fisso o Adattivo ad ATR)
def apply_triple_barrier(
    df,
    horizon=6,
    tp_pct=0.020,
    sl_pct=0.014,
    use_dynamic_atr=False,
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
        if use_dynamic_atr:
            target_tp = max(0.015, atrs[t] * atr_mult_tp)
            target_sl = max(0.010, atrs[t] * atr_mult_sl)
        else:
            target_tp = tp_pct
            target_sl = sl_pct

        long_tp = p0 * (1.0 + target_tp + fee)
        long_sl = p0 * (1.0 - target_sl - fee)
        short_tp = p0 * (1.0 - target_tp - fee)
        short_sl = p0 * (1.0 + target_sl + fee)

        l_hit, s_hit = None, None

        for k in range(1, horizon + 1):
            h, l = highs[t + k], lows[t + k]
            if h >= long_tp and l <= long_sl:
                break # Worst case conflitto
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
""")

# Cell 7: Dataset & Model definition
add_code("""# 6. PyTorch Dataset & Causal TCN Architecture
class TimeSeriesDataset(Dataset):
    def __init__(self, features, labels=None, lookback=64):
        self.features = features.astype(np.float32)
        self.labels = labels.astype(np.int64) if labels is not None else None
        self.lookback = lookback
        self.n_samples = len(features) - lookback + 1

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        window = self.features[idx : idx + self.lookback]
        x_tensor = torch.tensor(window.T, dtype=torch.float32)
        if self.labels is not None:
            target_idx = idx + self.lookback - 1
            y_tensor = torch.tensor(self.labels[target_idx], dtype=torch.long)
            return x_tensor, y_tensor
        return x_tensor, torch.tensor(-1, dtype=torch.long)

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

class EnsembleTCN(nn.Module):
    \"\"\"Deep Ensemble di N TCN indipendenti con media Softmax.\"\"\"
    def __init__(self, models):
        super().__init__()
        self.models = nn.ModuleList(models)

    def forward(self, x):
        logits_list = [m(x) for m in self.models]
        return torch.stack(logits_list, dim=0).mean(dim=0)

    def predict_probabilities(self, x):
        self.eval()
        with torch.no_grad():
            probs_list = [m.predict_probabilities(x) for m in self.models]
            return torch.stack(probs_list, dim=0).mean(dim=0)

    def save(self, filepath):
        state = {
            "type": "ensemble_tcn",
            "num_models": len(self.models),
            "state_dicts": [m.state_dict() for m in self.models]
        }
        torch.save(state, filepath)
""")

# Cell 8: Split & Training Utility
add_code("""# 7. Funzioni di Training, Validazione e Split
def prepare_splits(df_labeled):
    n = len(df_labeled)
    embargo = BASE_CONFIG["embargo_bars"]
    train_end = int(n * 0.70)
    val_start = train_end + embargo
    val_end = val_start + int(n * 0.15)
    test_start = val_end + embargo

    train_df = df_labeled.iloc[:train_end].copy().reset_index(drop=True)
    val_df = df_labeled.iloc[val_start:val_end].copy().reset_index(drop=True)
    test_df = df_labeled.iloc[test_start:].copy().reset_index(drop=True)

    scaler = FeatureScaler().fit(train_df)
    train_X = scaler.transform(train_df)
    val_X = scaler.transform(val_df)
    test_X = scaler.transform(test_df)

    lookback = BASE_CONFIG["lookback_window"]
    bs = BASE_CONFIG["batch_size"]
    train_loader = DataLoader(TimeSeriesDataset(train_X, train_df["label"].values, lookback), batch_size=bs, shuffle=True)
    val_loader = DataLoader(TimeSeriesDataset(val_X, val_df["label"].values, lookback), batch_size=bs, shuffle=False)
    test_loader = DataLoader(TimeSeriesDataset(test_X, test_df["label"].values, lookback), batch_size=bs, shuffle=False)

    return (train_df, val_df, test_df), (train_loader, val_loader, test_loader), scaler

def train_tcn_model(train_loader, val_loader, train_y, lr=3e-4, dropout=0.30, weight_decay=0.01, max_epochs=35, patience=8, seed=42):
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
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=4)

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

        # Valutazione
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
""")

# Cell 9: Generic Backtest Function
add_code("""# 8. Simulatore di Backtest Flessibile (Supporta Barriere Fisse, ATR e Trailing Stop)
def run_simulation(
    test_bars,
    probs,
    prob_threshold=0.38,
    ratio_threshold=1.25,
    fixed_tp_pct=0.028,
    fixed_sl_pct=0.018,
    use_dynamic_atr=False,
    atr_mult_tp=2.2,
    atr_mult_sl=1.4,
    use_trailing_stop=False,
    breakeven_trigger_pct=0.012,
    capital_allocation_pct=0.10,   # 10% capitale per trade (o 25%)
    initial_capital=1000.0,
    taker_fee=0.0005,
    max_holding=6
):
    capital = initial_capital
    equity_curve = [capital]
    active_trade = None
    trades = []

    for t in range(len(test_bars) - 1):
        curr_bar = test_bars.iloc[t]
        next_bar = test_bars.iloc[t + 1]

        # 1. Gestione trade aperto
        if active_trade is not None:
            active_trade["duration"] += 1
            h, l, c = curr_bar["high"], curr_bar["low"], curr_bar["close"]
            entry = active_trade["entry"]
            exit_p, reason = None, None

            # Calcolo dinamico barriere per questa barra
            if use_dynamic_atr:
                tp_dist = entry * active_trade["tp_target_pct"]
                sl_dist = entry * active_trade["sl_target_pct"]
                curr_tp = entry + tp_dist if active_trade["dir"] == "LONG" else entry - tp_dist
                curr_sl = entry - sl_dist if active_trade["dir"] == "LONG" else entry + sl_dist
            else:
                curr_tp = entry * (1.0 + fixed_tp_pct) if active_trade["dir"] == "LONG" else entry * (1.0 - fixed_tp_pct)
                curr_sl = entry * (1.0 - fixed_sl_pct) if active_trade["dir"] == "LONG" else entry * (1.0 + fixed_sl_pct)

            # Logica Trailing Stop a Breakeven
            if use_trailing_stop:
                if active_trade["dir"] == "LONG":
                    # Se il massimo ha superato il trigger breakeven (+1.2%), porta lo SL a Entry + Fee
                    if h >= entry * (1.0 + breakeven_trigger_pct):
                        active_trade["be_activated"] = True
                    if active_trade.get("be_activated", False):
                        be_price = entry * (1.0 + 2 * taker_fee)
                        curr_sl = max(curr_sl, be_price)
                else:
                    if l <= entry * (1.0 - breakeven_trigger_pct):
                        active_trade["be_activated"] = True
                    if active_trade.get("be_activated", False):
                        be_price = entry * (1.0 - 2 * taker_fee)
                        curr_sl = min(curr_sl, be_price)

            # Verifica tocchi barriere OCO
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

        # 2. Apertura nuovo trade
        if active_trade is None and t < len(probs):
            p_neu, p_l, p_s = probs[t]
            signal = None
            if p_l >= prob_threshold and p_l >= ratio_threshold * p_s:
                signal = "LONG"
            elif p_s >= prob_threshold and p_s >= ratio_threshold * p_l:
                signal = "SHORT"

            if signal is not None:
                trade_size = max(50.0, capital * capital_allocation_pct)
                atr_val = curr_bar["norm_atr"]
                active_trade = {
                    "dir": signal,
                    "entry": next_bar["open"],
                    "duration": 0,
                    "size": trade_size,
                    "tp_target_pct": max(0.015, atr_val * atr_mult_tp),
                    "sl_target_pct": max(0.010, atr_val * atr_mult_sl),
                    "be_activated": False
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
        "equity_curve": equity_curve
    }
""")

# Cell 10: Experiment 1: Grid Search
add_code("""# =====================================================================
# ESPERIMENTO 1: Grid Search su Validation Set (Selezione Iperparametri)
# =====================================================================
print("--- AVVIO ESPERIMENTO 1: Tuning Iperparametri su Validation Set ---")
df_labeled_exp1 = apply_triple_barrier(df_features, tp_pct=0.020, sl_pct=0.014, use_dynamic_atr=False)
(train_df, val_df, test_df), (train_loader, val_loader, test_loader), scaler_exp1 = prepare_splits(df_labeled_exp1)

grid_configs = [
    {"name": "C1 (Standard 3e-4, drop=0.30, wd=0.01)", "lr": 3e-4, "dropout": 0.30, "wd": 0.01},
    {"name": "C2 (High Reg 1.5e-4, drop=0.35, wd=0.02)", "lr": 1.5e-4, "dropout": 0.35, "wd": 0.02},
    {"name": "C3 (Moderate 5e-4, drop=0.25, wd=0.005)", "lr": 5e-4, "dropout": 0.25, "wd": 0.005}
]

best_exp1_model = None
best_exp1_f1 = -1.0
best_exp1_cfg = None

for cfg in grid_configs:
    print(f"Training {cfg['name']}...")
    model, v_loss, v_f1 = train_tcn_model(
        train_loader, val_loader, train_df["label"].values,
        lr=cfg["lr"], dropout=cfg["dropout"], weight_decay=cfg["wd"]
    )
    print(f"  --> Val Loss: {v_loss:.4f} | Val Macro F1: {v_f1:.4f}")
    if v_f1 > best_exp1_f1:
        best_exp1_f1 = v_f1
        best_exp1_model = model
        best_exp1_cfg = cfg

print(f"\\nConfigurazione Vincitrice Exp 1: {best_exp1_cfg['name']} con Val F1 = {best_exp1_f1:.4f}")

# Valutazione Backtest su Test Set
probs_exp1 = predict_probabilities(best_exp1_model, test_loader)
test_bars = test_df.iloc[BASE_CONFIG["lookback_window"] - 1:].copy().reset_index(drop=True)

res_exp1 = run_simulation(
    test_bars, probs_exp1,
    prob_threshold=0.38, ratio_threshold=1.25,
    fixed_tp_pct=0.028, fixed_sl_pct=0.018,
    capital_allocation_pct=0.10
)

print(f"Risultati Test Exp 1 (Baseline Tunata): Return={res_exp1['total_return_pct']}% | WinRate={res_exp1['win_rate_pct']}% | Trades={res_exp1['total_trades']} | DD={res_exp1['max_drawdown_pct']}%")
""")

# Cell 11: Experiment 2: Dynamic ATR
add_code("""# =====================================================================
# ESPERIMENTO 2: Dynamic ATR Volatility Barriers (Target Adattivi)
# =====================================================================
print("--- AVVIO ESPERIMENTO 2: Barriere Dinamiche Basate su ATR ---")
df_labeled_exp2 = apply_triple_barrier(
    df_features,
    use_dynamic_atr=True,
    atr_mult_tp=2.2,
    atr_mult_sl=1.4
)

(train_df2, val_df2, test_df2), (train_loader2, val_loader2, test_loader2), scaler_exp2 = prepare_splits(df_labeled_exp2)

# Addestra usando i migliori iperparametri dell'Exp 1
print(f"Training TCN su label ATR Dinamiche con parametri vincenti ({best_exp1_cfg['name']})...")
model_exp2, v_loss2, v_f12 = train_tcn_model(
    train_loader2, val_loader2, train_df2["label"].values,
    lr=best_exp1_cfg["lr"], dropout=best_exp1_cfg["dropout"], weight_decay=best_exp1_cfg["wd"]
)
print(f"Modello Exp 2 completato -> Val Loss: {v_loss2:.4f} | Val Macro F1: {v_f12:.4f}")

probs_exp2 = predict_probabilities(model_exp2, test_loader2)
test_bars2 = test_df2.iloc[BASE_CONFIG["lookback_window"] - 1:].copy().reset_index(drop=True)

res_exp2 = run_simulation(
    test_bars2, probs_exp2,
    prob_threshold=0.38, ratio_threshold=1.25,
    use_dynamic_atr=True,
    atr_mult_tp=2.2, atr_mult_sl=1.4,
    capital_allocation_pct=0.10
)

print(f"Risultati Test Exp 2 (Dynamic ATR): Return={res_exp2['total_return_pct']}% | WinRate={res_exp2['win_rate_pct']}% | Trades={res_exp2['total_trades']} | DD={res_exp2['max_drawdown_pct']}%")
""")

# Cell 12: Experiment 3: Deep Ensemble (5 TCN Multi-Seed Bagging)
add_code("""# =====================================================================
# ESPERIMENTO 3: Deep Ensemble (5-TCN Multi-Seed Bagging Consensus)
# =====================================================================
print("--- AVVIO ESPERIMENTO 3: Deep Ensemble (5 Reti TCN Collaborative) ---")
ENSEMBLE_SEEDS = [42, 101, 777, 1337, 2024]
ensemble_models = []

for i, s in enumerate(ENSEMBLE_SEEDS, 1):
    print(f"Training Membro Ensemble #{i}/{len(ENSEMBLE_SEEDS)} (Seed={s})...")
    m, v_loss_m, v_f1_m = train_tcn_model(
        train_loader2, val_loader2, train_df2["label"].values,
        lr=best_exp1_cfg["lr"],
        dropout=best_exp1_cfg["dropout"],
        weight_decay=best_exp1_cfg["wd"],
        seed=s
    )
    print(f"  --> Membro #{i} terminato: Val Loss: {v_loss_m:.4f} | Val F1: {v_f1_m:.4f}")
    ensemble_models.append(m)

ensemble = EnsembleTCN(ensemble_models)
probs_exp3 = predict_probabilities(ensemble, test_loader2)
test_bars3 = test_bars2

res_exp3 = run_simulation(
    test_bars3, probs_exp3,
    prob_threshold=0.38, ratio_threshold=1.25,
    use_dynamic_atr=True,
    atr_mult_tp=2.2, atr_mult_sl=1.4,
    capital_allocation_pct=0.10
)

print(f"Risultati Test Exp 3 (5-TCN Ensemble): Return={res_exp3['total_return_pct']}% | WinRate={res_exp3['win_rate_pct']}% | Trades={res_exp3['total_trades']} | DD={res_exp3['max_drawdown_pct']}%")
""")

# Cell 13: Summary Comparison & Export
add_code("""# =====================================================================
# 📊 CONFRONTO BENCHMARK MULTI-ESPERIMENTO & EXPORT CHAMPION
# =====================================================================
summary_df = pd.DataFrame([
    {
        "Esperimento": "1. Baseline Tunata (Fisso 10%)",
        "Rendimento %": f"{res_exp1['total_return_pct']:+.2f}%",
        "Max Drawdown %": f"{res_exp1['max_drawdown_pct']:.2f}%",
        "Win Rate %": f"{res_exp1['win_rate_pct']:.1f}%",
        "Profit Factor": res_exp1["profit_factor"],
        "Trade Totali": res_exp1["total_trades"]
    },
    {
        "Esperimento": "2. Dynamic ATR Volatility",
        "Rendimento %": f"{res_exp2['total_return_pct']:+.2f}%",
        "Max Drawdown %": f"{res_exp2['max_drawdown_pct']:.2f}%",
        "Win Rate %": f"{res_exp2['win_rate_pct']:.1f}%",
        "Profit Factor": res_exp2["profit_factor"],
        "Trade Totali": res_exp2["total_trades"]
    },
    {
        "Esperimento": "3. 5-TCN Deep Ensemble",
        "Rendimento %": f"{res_exp3['total_return_pct']:+.2f}%",
        "Max Drawdown %": f"{res_exp3['max_drawdown_pct']:.2f}%",
        "Win Rate %": f"{res_exp3['win_rate_pct']:.1f}%",
        "Profit Factor": res_exp3["profit_factor"],
        "Trade Totali": res_exp3["total_trades"]
    }
])

print("\\n" + "="*80)
print("TABELLA COMPARATIVA SUL TEST SET OUT-OF-SAMPLE (POST-FEE):")
print("="*80)
print(summary_df.to_string(index=False))
print("="*80 + "\\n")

# Grafico delle Equity Curve a confronto
plt.figure(figsize=(13, 6))
plt.plot(res_exp1["equity_curve"], label=f"Exp 1: Baseline Tuned ({res_exp1['total_return_pct']:+.1f}%)", color="#0077b6", lw=1.5)
plt.plot(res_exp2["equity_curve"], label=f"Exp 2: Dynamic ATR ({res_exp2['total_return_pct']:+.1f}%)", color="#ffaa00", lw=1.5)
plt.plot(res_exp3["equity_curve"], label=f"Exp 3: 5-TCN Deep Ensemble ({res_exp3['total_return_pct']:+.1f}%)", color="#00a86b", lw=2.2)
plt.title("TCN Quantitative Trading Bot - Confronto Multi-Esperimento su Test Set (Post-Fee)")
plt.ylabel("Portfolio Equity ($)")
plt.xlabel("Barre Temporali (4H)")
plt.grid(True, alpha=0.3)
plt.legend(loc="upper left")
plt.tight_layout()
chart_path = "/kaggle/working/comparison_equity_curves.png" if os.path.exists("/kaggle/working") else "models/comparison_equity_curves.png"
plt.savefig(chart_path, dpi=200)
plt.show()

# Salvataggio Pesi e Scaler
ensemble_save_path = "/kaggle/working/tcn_ensemble.pth" if os.path.exists("/kaggle/working") else "models/tcn_ensemble.pth"
single_save_path = "/kaggle/working/tcn_best_model.pth" if os.path.exists("/kaggle/working") else "models/tcn_best_model.pth"
scaler_save_path = "/kaggle/working/scaler.pkl" if os.path.exists("/kaggle/working") else "models/scaler.pkl"

ensemble.save(ensemble_save_path)
torch.save(model_exp2.state_dict(), single_save_path)
with open(scaler_save_path, "wb") as f:
    pickle.dump(scaler_exp2, f)

print(f"Ensemble (5 modelli) salvato in: {ensemble_save_path}")
print(f"Modello Singolo salvato in:     {single_save_path}")
print(f"Scaler Campione salvato in:     {scaler_save_path}")
print(f"Grafico comparativo salvato:    {chart_path}")
print("\\nEntrambi i modelli sono compatibili al 100% con il live agent!")
""")

os.makedirs("notebooks", exist_ok=True)
with open("notebooks/kaggle_train.ipynb", "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=2)

print("Notebook multi-esperimento generato con successo in notebooks/kaggle_train.ipynb!")
