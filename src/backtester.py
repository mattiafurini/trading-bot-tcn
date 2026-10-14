import os
import sys
import json
import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict

# Assicura che la root del progetto sia in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

from src.data_loader import load_config, get_market_data, get_chronological_splits
from src.features import extract_features, FeatureScaler, FEATURE_COLUMNS
from src.labeling import apply_triple_barrier_labeling
from src.dataset import TimeSeriesDataset
from src.model import TemporalConvolutionalNetwork

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


@dataclass
class Trade:
    entry_index: int
    entry_time: pd.Timestamp
    entry_price: float
    direction: str  # "LONG" o "SHORT"
    size_usdt: float
    exit_index: Optional[int] = None
    exit_time: Optional[pd.Timestamp] = None
    exit_price: Optional[float] = None
    exit_reason: Optional[str] = None  # "TAKE_PROFIT", "STOP_LOSS", "TIME_STOP"
    pnl_pct: float = 0.0
    pnl_net_usdt: float = 0.0
    fees_paid_usdt: float = 0.0
    duration_bars: int = 0
    be_activated: bool = False


class RealisticBacktester:
    """
    Simulatore Bar-by-Bar realistico per trading quantitativo con TCN.
    
    Caratteristiche:
    - Entrata sul prezzo di Open della barra immediatamente successiva al segnale (zero lookahead).
    - Gestione OCO (Take Profit vs Stop Loss) su High e Low barra per barra.
    - Supporto per Breakeven Trailing Stop (dopo +1.2%).
    - Uscita temporale forzata (Time Stop a 6 barre).
    - Detrazione accurata delle commissioni (0.05% taker in ingresso + 0.05% taker in uscita).
    - Esecuzione pessimistica conservativa sui conflitti intra-barra.
    """
    def __init__(
        self,
        df_bars: pd.DataFrame,
        probabilities: np.ndarray,
        config: Dict[str, Any],
        initial_capital: float = 1000.0,
        fixed_trade_size: float = 100.0,
        use_trailing_stop: bool = False,
        capital_allocation_pct: float = 0.10
    ):
        self.df = df_bars.copy().reset_index(drop=True)
        self.probs = probabilities  # Shape (N, 3) -> [P(Neutral), P(Long), P(Short)]
        self.config = config
        self.initial_capital = initial_capital
        self.fixed_trade_size = fixed_trade_size
        
        self.taker_fee = config["market"]["taker_fee"]
        self.tp_pct = config["decision"]["hard_take_profit_pct"]
        self.sl_pct = config["decision"]["hard_stop_loss_pct"]
        self.max_holding = config["decision"]["max_holding_bars"]
        
        self.long_thresh = config["decision"]["long_prob_threshold"]
        self.short_thresh = config["decision"]["short_prob_threshold"]
        self.ratio_thresh = config["decision"]["directional_ratio_threshold"]

        self.use_trailing_stop = use_trailing_stop or config["decision"].get("use_trailing_stop", False)
        self.breakeven_trigger_pct = config["decision"].get("breakeven_trigger_pct", 0.012)
        self.capital_allocation_pct = capital_allocation_pct or config["decision"].get("capital_allocation_pct", 0.10)

    def run(self) -> Dict[str, Any]:
        n_bars = len(self.df)
        trades: List[Trade] = []
        equity_curve = [self.initial_capital]
        timestamps = [self.df["timestamp"].iloc[0]]
        current_capital = self.initial_capital

        active_trade: Optional[Trade] = None
        
        # Iterazione barra per barra
        for t in range(n_bars - 1):
            curr_bar = self.df.iloc[t]
            next_bar = self.df.iloc[t + 1]

            # 1. Gestione del Trade Attivo
            if active_trade is not None:
                active_trade.duration_bars += 1
                exit_price = None
                exit_reason = None

                high = curr_bar["high"]
                low = curr_bar["low"]

                if active_trade.direction == "LONG":
                    tp_target = active_trade.entry_price * (1.0 + self.tp_pct)
                    sl_target = active_trade.entry_price * (1.0 - self.sl_pct)

                    # Trailing Stop a Breakeven
                    if self.use_trailing_stop:
                        if high >= active_trade.entry_price * (1.0 + self.breakeven_trigger_pct):
                            active_trade.be_activated = True
                        if active_trade.be_activated:
                            be_level = active_trade.entry_price * (1.0 + 2 * self.taker_fee)
                            sl_target = max(sl_target, be_level)

                    hit_tp = high >= tp_target
                    hit_sl = low <= sl_target

                    if hit_tp and hit_sl:
                        # Conflitto intra-barra: worst-case execution
                        exit_price = sl_target
                        exit_reason = "STOP_LOSS"
                    elif hit_sl:
                        exit_price = sl_target
                        exit_reason = "STOP_LOSS"
                    elif hit_tp:
                        exit_price = tp_target
                        exit_reason = "TAKE_PROFIT"
                    elif active_trade.duration_bars >= self.max_holding:
                        exit_price = curr_bar["close"]
                        exit_reason = "TIME_STOP"

                elif active_trade.direction == "SHORT":
                    tp_target = active_trade.entry_price * (1.0 - self.tp_pct)
                    sl_target = active_trade.entry_price * (1.0 + self.sl_pct)

                    # Trailing Stop a Breakeven
                    if self.use_trailing_stop:
                        if low <= active_trade.entry_price * (1.0 - self.breakeven_trigger_pct):
                            active_trade.be_activated = True
                        if active_trade.be_activated:
                            be_level = active_trade.entry_price * (1.0 - 2 * self.taker_fee)
                            sl_target = min(sl_target, be_level)

                    hit_tp = low <= tp_target
                    hit_sl = high >= sl_target

                    if hit_tp and hit_sl:
                        exit_price = sl_target
                        exit_reason = "STOP_LOSS"
                    elif hit_sl:
                        exit_price = sl_target
                        exit_reason = "STOP_LOSS"
                    elif hit_tp:
                        exit_price = tp_target
                        exit_reason = "TAKE_PROFIT"
                    elif active_trade.duration_bars >= self.max_holding:
                        exit_price = curr_bar["close"]
                        exit_reason = "TIME_STOP"

                # Chiusura della posizione
                if exit_price is not None:
                    active_trade.exit_index = t
                    active_trade.exit_time = curr_bar["timestamp"]
                    active_trade.exit_price = exit_price
                    active_trade.exit_reason = exit_reason

                    # Calcolo PnL
                    if active_trade.direction == "LONG":
                        gross_return = (exit_price - active_trade.entry_price) / active_trade.entry_price
                    else:
                        gross_return = (active_trade.entry_price - exit_price) / active_trade.entry_price

                    round_trip_fee = 2 * self.taker_fee
                    net_return = gross_return - round_trip_fee
                    fee_usdt = active_trade.size_usdt * round_trip_fee
                    pnl_usdt = (active_trade.size_usdt * gross_return) - fee_usdt

                    active_trade.pnl_pct = net_return
                    active_trade.pnl_net_usdt = pnl_usdt
                    active_trade.fees_paid_usdt = fee_usdt

                    current_capital += pnl_usdt
                    trades.append(active_trade)
                    active_trade = None

            # 2. Valutazione Nuovi Segnali (solo se non siamo già in posizione)
            if active_trade is None and t < len(self.probs):
                p_neutral, p_long, p_short = self.probs[t]

                signal = None
                if p_long >= self.long_thresh and p_long >= (self.ratio_thresh * p_short):
                    signal = "LONG"
                elif p_short >= self.short_thresh and p_short >= (self.ratio_thresh * p_long):
                    signal = "SHORT"

                if signal is not None:
                    # Entrata al Open della barra successiva (t + 1)
                    entry_price = next_bar["open"]
                    allocated_size = max(self.fixed_trade_size, current_capital * self.capital_allocation_pct)
                    size = min(allocated_size, current_capital * 0.95)
                    if size > 10.0:  # Minimo nozionale retail
                        active_trade = Trade(
                            entry_index=t + 1,
                            entry_time=next_bar["timestamp"],
                            entry_price=entry_price,
                            direction=signal,
                            size_usdt=size
                        )

            equity_curve.append(current_capital)
            timestamps.append(curr_bar["timestamp"])

        # Chiusura forzata trade residuo a fine test se presente
        if active_trade is not None:
            last_bar = self.df.iloc[-1]
            active_trade.exit_index = n_bars - 1
            active_trade.exit_time = last_bar["timestamp"]
            active_trade.exit_price = last_bar["close"]
            active_trade.exit_reason = "END_OF_TEST"
            
            if active_trade.direction == "LONG":
                gross_return = (last_bar["close"] - active_trade.entry_price) / active_trade.entry_price
            else:
                gross_return = (active_trade.entry_price - last_bar["close"]) / active_trade.entry_price
                
            fee_usdt = active_trade.size_usdt * 2 * self.taker_fee
            active_trade.pnl_pct = gross_return - (2 * self.taker_fee)
            active_trade.pnl_net_usdt = (active_trade.size_usdt * gross_return) - fee_usdt
            current_capital += active_trade.pnl_net_usdt
            trades.append(active_trade)

        # 3. Calcolo Metriche Quantitative
        metrics = self._calculate_performance_metrics(trades, equity_curve, timestamps)
        return {
            "metrics": metrics,
            "trades": [asdict(tr) for tr in trades],
            "equity_curve": equity_curve,
            "timestamps": [ts.isoformat() if hasattr(ts, "isoformat") else str(ts) for ts in timestamps]
        }

    def _calculate_performance_metrics(
        self,
        trades: List[Trade],
        equity: List[float],
        timestamps: List[pd.Timestamp]
    ) -> Dict[str, Any]:
        equity_series = pd.Series(equity)
        total_return_pct = ((equity[-1] - equity[0]) / equity[0]) * 100.0

        # Calcolo Drawdown
        running_max = equity_series.cummax()
        drawdown = (equity_series - running_max) / running_max
        max_drawdown_pct = abs(drawdown.min()) * 100.0

        if not trades:
            return {
                "total_trades": 0,
                "total_return_pct": 0.0,
                "max_drawdown_pct": 0.0,
                "win_rate_pct": 0.0,
                "profit_factor": 0.0,
                "sharpe_ratio": 0.0,
                "sortino_ratio": 0.0
            }

        pnls = [tr.pnl_net_usdt for tr in trades]
        pnl_pcts = [tr.pnl_pct for tr in trades]
        gains = [p for p in pnls if p > 0]
        losses = [abs(p) for p in pnls if p < 0]

        win_rate = (len(gains) / len(trades)) * 100.0
        profit_factor = (sum(gains) / sum(losses)) if sum(losses) > 0 else (999.0 if sum(gains) > 0 else 0.0)

        # Sharpe & Sortino (annualizzati su timeframe 4H: 6 barre/giorno * 365 = 2190 barre/anno)
        returns = equity_series.pct_change().dropna()
        annual_factor = np.sqrt(2190)
        
        std_dev = returns.std()
        sharpe = (returns.mean() / (std_dev + 1e-9)) * annual_factor if std_dev > 0 else 0.0
        
        downside_returns = returns[returns < 0]
        downside_std = downside_returns.std()
        sortino = (returns.mean() / (downside_std + 1e-9)) * annual_factor if (downside_std and downside_std > 0) else 0.0

        long_trades = [tr for tr in trades if tr.direction == "LONG"]
        short_trades = [tr for tr in trades if tr.direction == "SHORT"]

        return {
            "initial_capital": self.initial_capital,
            "final_equity": round(equity[-1], 2),
            "total_return_pct": round(total_return_pct, 2),
            "max_drawdown_pct": round(max_drawdown_pct, 2),
            "total_trades": len(trades),
            "long_trades": len(long_trades),
            "short_trades": len(short_trades),
            "win_rate_pct": round(win_rate, 2),
            "profit_factor": round(profit_factor, 2),
            "sharpe_ratio": round(sharpe, 2),
            "sortino_ratio": round(sortino, 2),
            "avg_trade_pnl_pct": round(float(np.mean(pnl_pcts)) * 100.0, 3),
            "avg_trade_duration_bars": round(float(np.mean([tr.duration_bars for tr in trades])), 1),
            "total_fees_paid_usdt": round(sum(tr.fees_paid_usdt for tr in trades), 2)
        }

    def plot_results(self, output_path: str = "models/equity_curve.png"):
        """Genera e salva il grafico di Equity Curve e Drawdown."""
        plt.style.use("seaborn-v0_8-darkgrid" if "seaborn-v0_8-darkgrid" in plt.style.available else "default")
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7), sharex=True, gridspec_kw={"height_ratios": [3, 1]})

        equity_s = pd.Series(self.run()["equity_curve"])
        running_max = equity_s.cummax()
        drawdown_s = (equity_s - running_max) / running_max * 100.0

        ax1.plot(equity_s, label="Portfolio Equity (USDT)", color="#00ff88" if plt.isinteractive() else "#0077b6", linewidth=1.5)
        ax1.set_title("TCN Quantitative Trading Bot - Equity Curve Post-Fee (4H BTC/USDT)")
        ax1.set_ylabel("Equity ($)")
        ax1.legend(loc="upper left")

        ax2.fill_between(range(len(drawdown_s)), drawdown_s, 0, color="#e63946", alpha=0.4, label="Drawdown %")
        ax2.set_ylabel("Drawdown %")
        ax2.set_xlabel("Barre (4H)")
        ax2.legend(loc="lower left")

        plt.tight_layout()
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=200)
        plt.close()
        logger.info(f"Grafico equity curve salvato in {output_path}")


def run_out_of_sample_backtest(
    model_path: str = "models/tcn_best_model.pth",
    scaler_path: str = "models/scaler.pkl",
    config_path: str = "config/config.yaml"
) -> Dict[str, Any]:
    """Esegue il backtest completo sui dati congelati di Test Out-of-Sample."""
    cfg = load_config(config_path)
    
    # 1. Carica dati e prepara dataset test
    df_raw = get_market_data(cfg)
    df_feat = extract_features(df_raw, sma_short=cfg["features"]["sma_short"], sma_long=cfg["features"]["sma_long"])
    df_labeled = apply_triple_barrier_labeling(df_feat, horizon=cfg["labeling"]["prediction_horizon"])
    
    _, _, test_df = get_chronological_splits(
        df_labeled,
        train_ratio=cfg["training"]["train_ratio"],
        val_ratio=cfg["training"]["val_ratio"],
        test_ratio=cfg["training"]["test_ratio"],
        embargo_bars=cfg["training"]["embargo_bars"]
    )
    
    # 2. Carica Scaler e Modello
    scaler = FeatureScaler.load(scaler_path)
    test_X = scaler.transform(test_df)
    
    lookback = cfg["features"]["lookback_window"]
    test_ds = TimeSeriesDataset(test_X, test_df["label"].values, lookback=lookback)
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=64, shuffle=False)

    device = torch.device("cpu")
    model = TemporalConvolutionalNetwork(
        num_features=len(FEATURE_COLUMNS),
        channels=cfg["model"]["channels"],
        kernel_size=cfg["model"]["kernel_size"],
        dropout=0.0,
        use_layer_norm=cfg["model"]["use_layer_norm"],
        num_classes=cfg["model"]["num_classes"]
    )
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # 3. Predizione delle probabilità
    all_probs = []
    with torch.no_grad():
        for x_batch, _ in test_loader:
            probs = model.predict_probabilities(x_batch)
            all_probs.append(probs.numpy())
    probs_matrix = np.concatenate(all_probs, axis=0)

    # Allinea le barre: il dataset inizia dopo 'lookback - 1' barre
    test_bars = test_df.iloc[lookback - 1:].copy().reset_index(drop=True)

    # 4. Esecuzione del Backtester
    backtester = RealisticBacktester(
        df_bars=test_bars,
        probabilities=probs_matrix,
        config=cfg,
        initial_capital=1000.0,
        fixed_trade_size=cfg["execution"]["trade_size_usdt"]
    )
    results = backtester.run()
    backtester.plot_results("models/equity_curve.png")

    logger.info("=" * 60)
    logger.info("RISULTATI FINANZIARI BACKTEST (OUT-OF-SAMPLE POST-FEE):")
    for k, v in results["metrics"].items():
        logger.info(f"  {k}: {v}")
    logger.info("=" * 60)

    return results


if __name__ == "__main__":
    run_out_of_sample_backtest()
