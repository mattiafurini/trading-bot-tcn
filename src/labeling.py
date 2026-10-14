import numpy as np
import pandas as pd
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


def apply_triple_barrier_labeling(
    df: pd.DataFrame,
    horizon: int = 6,
    profit_target_pct: float = 0.025,
    stop_loss_pct: float = 0.015,
    taker_fee: float = 0.0005,
    use_atr_dynamic: bool = False,
    atr_multiplier_target: float = 2.0,
    atr_multiplier_sl: float = 1.5,
    worst_case_conflict: bool = True
) -> pd.DataFrame:
    """
    Applica la metodologia Triple Barrier (López de Prado) per generare label a 3 classi:
    
    0: Neutral (Movimento laterale, rumore, o Stop Loss raggiunto prima del target)
    1: Long (Take profit raggiunto prima dello stop loss entro l'orizzonte di holding)
    2: Short (Take profit al ribasso raggiunto prima dello stop loss entro l'orizzonte di holding)
    
    Gestisce rigorosamente:
    - Costo delle commissioni (0.05% taker in ingresso + 0.05% in uscita)
    - Conflitti intra-barra (se High tocca TP e Low tocca SL nella stessa barra 4H,
      assume Stop Loss colpito per primo in modalità conservativa 'worst_case_conflict').
    - Assenza di lookahead leakage: le ultime `horizon` barre vengono scartate perché
      non hanno futuro sufficiente per una valutazione oggettiva.
    """
    df = df.copy()
    n = len(df)
    labels = np.zeros(n, dtype=int)
    
    closes = df["close"].values
    highs = df["high"].values
    lows = df["low"].values
    norm_atrs = df["norm_atr"].values if "norm_atr" in df.columns else np.zeros(n)
    
    round_trip_fee = 2 * taker_fee  # 0.10%

    for t in range(n - horizon):
        entry_price = closes[t]
        
        # Calcolo barriere (fisse o dinamiche via ATR)
        if use_atr_dynamic and norm_atrs[t] > 0:
            target_pct = max(0.015, norm_atrs[t] * atr_multiplier_target)
            sl_pct = max(0.010, norm_atrs[t] * atr_multiplier_sl)
        else:
            target_pct = profit_target_pct
            sl_pct = stop_loss_pct

        # Livelli di prezzo con fee incorporate
        long_tp_price = entry_price * (1.0 + target_pct + round_trip_fee)
        long_sl_price = entry_price * (1.0 - sl_pct - round_trip_fee)
        
        short_tp_price = entry_price * (1.0 - target_pct - round_trip_fee)
        short_sl_price = entry_price * (1.0 + sl_pct + round_trip_fee)

        long_hit_step = None
        short_hit_step = None

        # Valutazione Long
        for k in range(1, horizon + 1):
            future_high = highs[t + k]
            future_low = lows[t + k]

            hit_tp = future_high >= long_tp_price
            hit_sl = future_low <= long_sl_price

            if hit_tp and hit_sl:
                # Conflitto intra-barra
                if worst_case_conflict:
                    # Pessimistic execution: assume SL hit
                    break
                else:
                    long_hit_step = k
                    break
            elif hit_sl:
                # Stop loss colpito prima del target
                break
            elif hit_tp:
                long_hit_step = k
                break

        # Valutazione Short
        for k in range(1, horizon + 1):
            future_high = highs[t + k]
            future_low = lows[t + k]

            hit_tp = future_low <= short_tp_price
            hit_sl = future_high >= short_sl_price

            if hit_tp and hit_sl:
                # Conflitto intra-barra
                if worst_case_conflict:
                    break
                else:
                    short_hit_step = k
                    break
            elif hit_sl:
                # Stop loss colpito prima del target
                break
            elif hit_tp:
                short_hit_step = k
                break

        # Risoluzione del segnale
        if long_hit_step is not None and short_hit_step is not None:
            # Entrambi hanno toccato il target in momenti diversi: vince il più veloce
            if long_hit_step < short_hit_step:
                labels[t] = 1
            elif short_hit_step < long_hit_step:
                labels[t] = 2
            else:
                labels[t] = 0  # Ambiguità nella stessa barra -> neutral
        elif long_hit_step is not None:
            labels[t] = 1
        elif short_hit_step is not None:
            labels[t] = 2
        else:
            labels[t] = 0

    df["label"] = labels
    
    # Rimuove le ultime 'horizon' barre non etichettabili
    df_valid = df.iloc[:-horizon].copy().reset_index(drop=True)
    
    counts = df_valid["label"].value_counts().to_dict()
    total = len(df_valid)
    logger.info(f"Distribuzione Label Triple Barrier ({total} barre etichettate):")
    logger.info(f"  Classe 0 (Neutral): {counts.get(0, 0)} ({counts.get(0, 0)/total*100:.1f}%)")
    logger.info(f"  Classe 1 (Long):    {counts.get(1, 0)} ({counts.get(1, 0)/total*100:.1f}%)")
    logger.info(f"  Classe 2 (Short):   {counts.get(2, 0)} ({counts.get(2, 0)/total*100:.1f}%)")

    return df_valid
