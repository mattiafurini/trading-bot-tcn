import os
import time
import yaml
import logging
from datetime import datetime, timedelta, timezone
from typing import Tuple, Dict, Any, Optional

import pandas as pd
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def load_config(config_path: str = "config/config.yaml") -> Dict[str, Any]:
    """Carica la configurazione da file YAML."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def get_exchange_instance(exchange_id: str = "binance"):
    """Inizializza l'istanza dell'exchange tramite CCXT configurata per Perpetual Futures."""
    import ccxt
    exchange_class = getattr(ccxt, exchange_id)
    exchange = exchange_class({
        "enableRateLimit": True,
        "options": {
            "defaultType": "future",  # USD(S)-M Futures su Binance
        }
    })
    return exchange


def fetch_historical_ohlcv(
    symbol: str = "BTC/USDT",
    timeframe: str = "4h",
    since_days: int = 1825,
    exchange_id: str = "binance"
) -> pd.DataFrame:
    """
    Scarica lo storico OHLCV continuo tramite CCXT con paginazione.
    
    Args:
        symbol: Coppia di trading (es. 'BTC/USDT')
        timeframe: Timeframe delle candele (es. '4h')
        since_days: Giorni di storico da scaricare (default 1825 = ~5 anni)
        exchange_id: ID exchange CCXT (default 'binance')
    
    Returns:
        pd.DataFrame con colonne [timestamp, open, high, low, close, volume]
    """
    exchange = get_exchange_instance(exchange_id)
    exchange.load_markets()
    
    # Determina il market symbol corretto per i perpetual futures
    target_symbol = symbol
    if f"{symbol}:USDT" in exchange.markets:
        target_symbol = f"{symbol}:USDT"
    elif symbol not in exchange.markets:
        # Fallback di ricerca
        for s in exchange.markets:
            if s.startswith(symbol) and "USDT" in s:
                target_symbol = s
                break

    logger.info(f"Connessione a {exchange_id} per {target_symbol} timeframe {timeframe}...")

    now = datetime.now(timezone.utc)
    start_dt = now - timedelta(days=since_days)
    since_ms = int(start_dt.timestamp() * 1000)
    
    all_ohlcv = []
    limit = 1000
    
    while True:
        try:
            ohlcv = exchange.fetch_ohlcv(target_symbol, timeframe=timeframe, since=since_ms, limit=limit)
            if not ohlcv or len(ohlcv) == 0:
                break
            
            all_ohlcv.extend(ohlcv)
            last_ts = ohlcv[-1][0]
            
            logger.info(f"Scaricati {len(all_ohlcv)} record (ultimo: {datetime.fromtimestamp(last_ts/1000, tz=timezone.utc).strftime('%Y-%m-%d %H:%M')})")
            
            # Se l'ultimo timestamp è vicino ad adesso o non avanza più
            if last_ts == since_ms or len(ohlcv) < limit:
                break
            
            since_ms = last_ts + 1  # Incrementa di 1ms per la prossima pagina
            time.sleep(exchange.rateLimit / 1000.0)
            
        except Exception as e:
            logger.warning(f"Errore durante fetch_ohlcv: {e}. Attendo 2 secondi...")
            time.sleep(2)
            break

    df = pd.DataFrame(all_ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df.drop_duplicates(subset=["timestamp"], inplace=True)
    df.sort_values(by="timestamp", inplace=True)
    df.reset_index(drop=True, inplace=True)
    
    # Casting numerico
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        
    logger.info(f"Completato download OHLCV: {len(df)} barre da {df['timestamp'].iloc[0]} a {df['timestamp'].iloc[-1]}")
    return df


def fetch_historical_funding_rates(
    symbol: str = "BTC/USDT",
    since_days: int = 1825,
    exchange_id: str = "binance"
) -> pd.DataFrame:
    """
    Scarica la cronologia dei Funding Rate storici (8-hour interval).
    Se l'API non supporta storico profondo, gestisce gracefully il fallback.
    """
    exchange = get_exchange_instance(exchange_id)
    exchange.load_markets()
    target_symbol = symbol
    if exchange.markets and f"{symbol}:USDT" in exchange.markets:
        target_symbol = f"{symbol}:USDT"

    now = datetime.now(timezone.utc)
    start_dt = now - timedelta(days=since_days)
    since_ms = int(start_dt.timestamp() * 1000)
    
    all_rates = []
    limit = 1000
    
    if not exchange.has.get("fetchFundingRateHistory"):
        logger.warning(f"{exchange_id} non supporta fetchFundingRateHistory. Uso funding rate zero di default.")
        return pd.DataFrame(columns=["timestamp", "funding_rate"])

    try:
        while True:
            rates = exchange.fetch_funding_rate_history(target_symbol, since=since_ms, limit=limit)
            if not rates or len(rates) == 0:
                break
            all_rates.extend(rates)
            last_ts = rates[-1]["timestamp"]
            if last_ts == since_ms or len(rates) < limit:
                break
            since_ms = last_ts + 1
            time.sleep(exchange.rateLimit / 1000.0)
    except Exception as e:
        logger.warning(f"Impossibile scaricare intero storico funding rate: {e}. Procedo con i dati raccolti ({len(all_rates)}).")

    if not all_rates:
        return pd.DataFrame(columns=["timestamp", "funding_rate"])

    df_rates = pd.DataFrame([{
        "timestamp": pd.to_datetime(r["timestamp"], unit="ms", utc=True),
        "funding_rate": float(r["fundingRate"]) if r.get("fundingRate") is not None else 0.0
    } for r in all_rates])
    
    df_rates.drop_duplicates(subset=["timestamp"], inplace=True)
    df_rates.sort_values(by="timestamp", inplace=True)
    df_rates.reset_index(drop=True, inplace=True)
    return df_rates


def get_market_data(config: Dict[str, Any], force_download: bool = False) -> pd.DataFrame:
    """
    Carica i dati storici dalla cache locale (parquet/csv) o li scarica via CCXT.
    Allinea le barre 4H con i funding rate storici.
    """
    data_dir = "data"
    os.makedirs(data_dir, exist_ok=True)
    
    parquet_path = os.path.join(data_dir, f"{config['market']['symbol'].replace('/', '_')}_{config['market']['timeframe']}.parquet")
    csv_path = os.path.join(data_dir, f"{config['market']['symbol'].replace('/', '_')}_{config['market']['timeframe']}.csv")

    if not force_download:
        if os.path.exists(parquet_path):
            logger.info(f"Caricamento dati dalla cache parquet: {parquet_path}")
            return pd.read_parquet(parquet_path)
        elif os.path.exists(csv_path):
            logger.info(f"Caricamento dati dalla cache csv: {csv_path}")
            df = pd.read_csv(csv_path)
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
            return df

    # Download nuovo
    df_ohlcv = fetch_historical_ohlcv(
        symbol=config["market"]["symbol"],
        timeframe=config["market"]["timeframe"],
        since_days=config["market"]["since_days"],
        exchange_id=config["market"]["exchange"]
    )
    
    # Download funding rate
    df_funding = fetch_historical_funding_rates(
        symbol=config["market"]["symbol"],
        since_days=config["market"]["since_days"],
        exchange_id=config["market"]["exchange"]
    )
    
    # Merge asof per allineare l'ultimo funding rate disponibile senza lookahead bias
    if not df_funding.empty:
        df_merged = pd.merge_asof(
            df_ohlcv,
            df_funding,
            on="timestamp",
            direction="backward"
        )
        df_merged["funding_rate"] = df_merged["funding_rate"].fillna(0.0)
    else:
        df_merged = df_ohlcv
        df_merged["funding_rate"] = 0.0

    # Salva in cache
    try:
        df_merged.to_parquet(parquet_path, index=False)
        logger.info(f"Dati salvati con successo in {parquet_path}")
    except Exception:
        df_merged.to_csv(csv_path, index=False)
        logger.info(f"Dati salvati con successo in {csv_path}")

    return df_merged


def get_chronological_splits(
    df: pd.DataFrame,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    embargo_bars: int = 12
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Suddivide il dataset cronologicamente con finestre di embargo (purging)
    tra Train e Validation, e tra Validation e Test per impedire ogni data leakage.
    
    Returns:
        Tuple[train_df, val_df, test_df]
    """
    n = len(df)
    train_end = int(n * train_ratio)
    val_start = train_end + embargo_bars
    
    val_end = val_start + int(n * val_ratio)
    test_start = val_end + embargo_bars
    
    train_df = df.iloc[:train_end].copy().reset_index(drop=True)
    val_df = df.iloc[val_start:val_end].copy().reset_index(drop=True)
    test_df = df.iloc[test_start:].copy().reset_index(drop=True)
    
    logger.info(f"Split Dati con Embargo di {embargo_bars} barre (48 ore):")
    logger.info(f"  Train: {len(train_df)} barre ({train_df['timestamp'].iloc[0]} -> {train_df['timestamp'].iloc[-1]})")
    logger.info(f"  Gap Embargo 1: {embargo_bars} barre eliminate")
    logger.info(f"  Validation: {len(val_df)} barre ({val_df['timestamp'].iloc[0]} -> {val_df['timestamp'].iloc[-1]})")
    logger.info(f"  Gap Embargo 2: {embargo_bars} barre eliminate")
    logger.info(f"  Test Out-of-Sample: {len(test_df)} barre ({test_df['timestamp'].iloc[0]} -> {test_df['timestamp'].iloc[-1]})")
    
    return train_df, val_df, test_df


if __name__ == "__main__":
    cfg = load_config()
    df_data = get_market_data(cfg)
    train, val, test = get_chronological_splits(df_data, embargo_bars=cfg["training"]["embargo_bars"])
    print(f"Dataset caricato: Totale barre = {len(df_data)}")
