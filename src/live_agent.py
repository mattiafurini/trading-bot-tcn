import os
import sys
import time
import json
import logging
import argparse
from datetime import datetime, timezone
from typing import Dict, Any, Optional

# Assicura che la root del progetto sia in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pandas as pd
import torch
import ccxt

from src.data_loader import load_config
from src.features import extract_features, FeatureScaler, FEATURE_COLUMNS
from src.model import TemporalConvolutionalNetwork, EnsembleTCN
from src.telegram_bot import TelegramNotifier

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

LEDGER_PATH = "data/paper_trading_ledger.json"


class PaperTradingLedger:
    """Gestisce lo stato persistente delle posizioni e del bilancio simulato su file JSON."""
    def __init__(self, initial_balance: float = 100.0, circuit_breaker_min: float = 50.0):
        self.initial_balance = initial_balance
        self.circuit_breaker_min = circuit_breaker_min
        self.state = self.load()

    def load(self) -> Dict[str, Any]:
        os.makedirs(os.path.dirname(LEDGER_PATH), exist_ok=True)
        if os.path.exists(LEDGER_PATH):
            try:
                with open(LEDGER_PATH, "r") as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Errore lettura ledger ({e}). Inizializzazione nuovo ledger.")
        
        default_state = {
            "initial_balance": self.initial_balance,
            "current_balance": self.initial_balance,
            "circuit_breaker_triggered": False,
            "current_position": None,
            "closed_trades": [],
            "total_pnl_usdt": 0.0,
            "total_fees_usdt": 0.0,
            "win_rate_pct": 0.0,
            "last_update": datetime.now(timezone.utc).isoformat()
        }
        self.save(default_state)
        return default_state

    def save(self, state: Optional[Dict[str, Any]] = None):
        if state is not None:
            self.state = state
        self.state["last_update"] = datetime.now(timezone.utc).isoformat()
        with open(LEDGER_PATH, "w") as f:
            json.dump(self.state, f, indent=2)

    @property
    def balance(self) -> float:
        return self.state.get("current_balance", self.initial_balance)

    @property
    def position(self) -> Optional[Dict[str, Any]]:
        return self.state.get("current_position")

    def is_circuit_breaker_active(self) -> bool:
        if self.balance < self.circuit_breaker_min:
            self.state["circuit_breaker_triggered"] = True
            self.save()
            return True
        return False


class RealnetPaperAgent:
    """
    Agente di Paper Trading sul Mercato REALE (Realnet Binance Futures).
    - Riceve i prezzi autentici e in tempo reale dal mercato live di Binance.
    - Zero soldi veri a rischio e zero API keys richieste per il paper trading.
    - Supporta sia il Deep Ensemble (tcn_ensemble.pth) sia il modello singolo (tcn_best_model.pth).
    - Gestisce un conto virtuale da 100$ con compounding al 95% e circuit breaker a 50$.
    """
    def __init__(
        self,
        config_path: str = "config/config.yaml",
        model_path: str = "models/tcn_best_model.pth",
        scaler_path: str = "models/scaler.pkl"
    ):
        self.config = load_config(config_path)
        self.symbol = self.config["market"]["symbol"]
        self.timeframe = self.config["market"]["timeframe"]
        self.taker_fee = self.config["market"]["taker_fee"]
        self.max_holding = self.config["decision"]["max_holding_bars"]
        
        self.long_thresh = self.config["decision"]["long_prob_threshold"]
        self.short_thresh = self.config["decision"]["short_prob_threshold"]
        self.ratio_thresh = self.config["decision"]["directional_ratio_threshold"]

        # Carica il ledger di paper trading
        init_bal = self.config["execution"].get("initial_paper_balance", 100.0)
        cb_min = self.config["execution"].get("circuit_breaker_min_balance", 50.0)
        self.ledger = PaperTradingLedger(initial_balance=init_bal, circuit_breaker_min=cb_min)

        # Inizializza Scaler
        logger.info(f"Caricamento Scaler da {scaler_path}...")
        self.scaler = FeatureScaler.load(scaler_path)

        # Controlla se è disponibile un Deep Ensemble di TCN
        models_dir = os.path.dirname(model_path) if os.path.dirname(model_path) else "models"
        ensemble_candidate = os.path.join(models_dir, "tcn_ensemble.pth")
        target_path = ensemble_candidate if os.path.exists(ensemble_candidate) else model_path

        logger.info(f"Caricamento Rete Neurale da {target_path}...")
        self.device = torch.device("cpu")
        self.model = EnsembleTCN.load(
            target_path,
            num_features=len(FEATURE_COLUMNS),
            channels=self.config["model"]["channels"],
            kernel_size=self.config["model"]["kernel_size"],
            dropout=0.0,
            use_layer_norm=self.config["model"]["use_layer_norm"],
            num_classes=self.config["model"]["num_classes"],
            device=self.device
        )
        if len(self.model.models) > 1:
            logger.info(f"🧠 Deep Ensemble Attivo con {len(self.model.models)} reti TCN collaborative!")
        else:
            logger.info("Modello Singolo TCN caricato con successo.")

        # Inizializza Notificatore Telegram
        tg_cfg = self.config.get("telegram", {})
        self.telegram = TelegramNotifier(
            token=tg_cfg.get("bot_token"),
            chat_id=tg_cfg.get("chat_id"),
            enabled=tg_cfg.get("enabled", True),
            notify_on_start=tg_cfg.get("notify_on_start", True),
            notify_on_trade=tg_cfg.get("notify_on_trade", True),
            notify_on_close=tg_cfg.get("notify_on_close", True),
            notify_on_circuit_breaker=tg_cfg.get("notify_on_circuit_breaker", True)
        )

        # Inizializza CCXT su Binance Futures PRODUZIONE (Dati Reali Live)
        self._init_exchange()

    def _init_exchange(self):
        # Connessione al mercato pubblico di produzione (nessuna API key necessaria per paper trading)
        self.exchange = ccxt.binance({
            "enableRateLimit": True,
            "options": {
                "defaultType": "future",
            }
        })
        logger.info("Connesso al Mercato LIVE REALE di Binance Futures (Zero API keys richieste per Paper Trading).")

    def fetch_live_data(self, min_bars: int = 300) -> pd.DataFrame:
        """Scarica le ultime 300 barre 4H reali dal live market di Binance."""
        target_symbol = f"{self.symbol}:USDT"
        ohlcv = self.exchange.fetch_ohlcv(target_symbol, timeframe=self.timeframe, limit=min_bars)
        
        df = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        for col in ["open", "high", "low", "close", "volume"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        try:
            funding = self.exchange.fetch_funding_rate(target_symbol)
            df["funding_rate"] = float(funding.get("fundingRate", 0.0))
        except Exception:
            df["funding_rate"] = 0.0

        # Escludi l'ultima candela se ancora in formazione (aperta da meno di 4 ore)
        # per calcolare indicatori e feature solo su barre 4H storicamente consolidate
        now_ts = pd.Timestamp.now(tz=timezone.utc)
        if len(df) > 0 and (now_ts - df["timestamp"].iloc[-1]).total_seconds() < 4 * 3600:
            df = df.iloc[:-1].reset_index(drop=True)

        return df

    def get_realtime_ticker(self) -> float:
        """Recupera il prezzo esatto di ultimo scambio (tick-by-tick) da Binance."""
        ticker = self.exchange.fetch_ticker(f"{self.symbol}:USDT")
        return float(ticker["last"])

    def evaluate(self):
        """Esegue un ciclo di valutazione: gestisce trade aperti e valuta nuovi ingressi."""
        if self.ledger.is_circuit_breaker_active():
            logger.error(
                f"🚨 CIRCUIT BREAKER ATTIVO! Saldo ({self.ledger.balance:.2f} USDT) inferiore a "
                f"{self.ledger.circuit_breaker_min:.2f} USDT. Trading sospeso per tutela capitale."
            )
            self.telegram.notify_circuit_breaker(self.ledger.balance, self.ledger.circuit_breaker_min)
            return

        live_price = self.get_realtime_ticker()
        logger.info(f"=== Valutazione Mercato Realnet: {self.symbol} ===")
        logger.info(f"Prezzo BTC Live: {live_price:.2f} USDT | Saldo Paper: {self.ledger.balance:.2f} USDT")

        # 1. Controlla e gestisci l'eventuale posizione aperta
        if self.ledger.position is not None:
            self._manage_open_position(live_price)
            return

        # 2. Nessuna posizione aperta: scarica dati ed estrai feature per nuova predizione
        df_raw = self.fetch_live_data(min_bars=300)
        df_feat = extract_features(
            df_raw,
            sma_short=self.config["features"]["sma_short"],
            sma_long=self.config["features"]["sma_long"],
            atr_period=self.config["features"]["atr_period"],
            vol_period=self.config["features"]["volume_ma_period"]
        )

        lookback = self.config["features"]["lookback_window"]
        latest_window = df_feat.iloc[-lookback:].copy()
        current_atr = latest_window["norm_atr"].iloc[-1]
        last_bar_time = latest_window["timestamp"].iloc[-1]

        # Inferenza TCN
        x_scaled = self.scaler.transform(latest_window)
        x_tensor = torch.tensor(x_scaled.T, dtype=torch.float32).unsqueeze(0)

        with torch.no_grad():
            probs = self.model.predict_probabilities(x_tensor).numpy()[0]

        p_neutral, p_long, p_short = probs
        logger.info(
            f"Ultima barra chiusa: {last_bar_time} | "
            f"P(Neutral): {p_neutral*100:.1f}% | P(Long): {p_long*100:.1f}% | P(Short): {p_short*100:.1f}%"
        )

        # Decision Engine deterministico con soglie calibrate
        probs_dict = {"neutral": float(p_neutral), "long": float(p_long), "short": float(p_short)}
        if p_long >= self.long_thresh and p_long >= (self.ratio_thresh * p_short):
            logger.info(f"🟢 SEGNALE FORTE LONG RILEVATO (P={p_long:.2f})")
            self._open_trade("LONG", live_price, current_atr, probs=probs_dict)
        elif p_short >= self.short_thresh and p_short >= (self.ratio_thresh * p_long):
            logger.info(f"🔴 SEGNALE FORTE SHORT RILEVATO (P={p_short:.2f})")
            self._open_trade("SHORT", live_price, current_atr, probs=probs_dict)
        else:
            logger.info("⚪ Nessun segnale direzionale netto sopra soglia -> Mantieni stato Flat.")

    def _open_trade(self, direction: str, entry_price: float, norm_atr: float, probs: Optional[Dict[str, float]] = None):
        # Target Dinamici ATR vincenti dell'Esperimento 2
        atr_mult_tp = self.config["execution"].get("atr_multiplier_target", 2.2)
        atr_mult_sl = self.config["execution"].get("atr_multiplier_sl", 1.4)
        
        target_tp_pct = max(0.015, norm_atr * atr_mult_tp)
        target_sl_pct = max(0.010, norm_atr * atr_mult_sl)

        tp_price = entry_price * (1.0 + target_tp_pct) if direction == "LONG" else entry_price * (1.0 - target_tp_pct)
        sl_price = entry_price * (1.0 - target_sl_pct) if direction == "LONG" else entry_price * (1.0 + target_sl_pct)

        # Compounding: alloca il 95% del saldo corrente (all-in a Leva 1x con stop loss)
        alloc_pct = self.config["execution"].get("compounding_allocation_pct", 0.95)
        trade_size = max(20.0, self.ledger.balance * alloc_pct)

        pos_data = {
            "direction": direction,
            "entry_price": entry_price,
            "entry_time": datetime.now(timezone.utc).isoformat(),
            "tp_price": tp_price,
            "sl_price": sl_price,
            "tp_pct": target_tp_pct,
            "sl_pct": target_sl_pct,
            "holding_bars": 0,
            "size_usdt": trade_size
        }

        self.ledger.state["current_position"] = pos_data
        self.ledger.save()

        logger.info(
            f"✅ [PAPER TRADE APERTO] {direction} | Entry: {entry_price:.2f} USDT | "
            f"Size: {trade_size:.2f} USDT | TP: {tp_price:.2f} (+{target_tp_pct*100:.2f}%) | "
            f"SL: {sl_price:.2f} (-{target_sl_pct*100:.2f}%)"
        )

        self.telegram.notify_trade_opened(
            direction=direction,
            symbol=self.symbol,
            entry_price=entry_price,
            size_usdt=trade_size,
            tp_price=tp_price,
            sl_price=sl_price,
            tp_pct=target_tp_pct,
            sl_pct=target_sl_pct,
            probs=probs
        )

    def _manage_open_position(self, current_price: float):
        pos = self.ledger.position
        direction = pos["direction"]
        entry_price = pos["entry_price"]

        # Calcolo del tempo reale trascorso dall'apertura
        entry_time = datetime.fromisoformat(pos["entry_time"])
        now_time = datetime.now(timezone.utc)
        elapsed_hours = (now_time - entry_time).total_seconds() / 3600.0
        elapsed_bars = elapsed_hours / 4.0
        pos["holding_bars"] = round(elapsed_bars, 2)
        is_time_stop = elapsed_bars >= self.max_holding

        # Recupera candele a 1 minuto per catturare ombre/spike senza perdere opportunità
        now_ms = self.exchange.milliseconds()
        last_check_ms = pos.get("last_check_ms", now_ms - (15 * 60 * 1000))
        pos["last_check_ms"] = now_ms

        high_reach = current_price
        low_reach = current_price
        try:
            candles_1m = self.exchange.fetch_ohlcv(f"{self.symbol}:USDT", timeframe="1m", since=last_check_ms, limit=100)
            if candles_1m:
                high_reach = max(c[2] for c in candles_1m)
                low_reach = min(c[3] for c in candles_1m)
        except Exception as e:
            logger.warning(f"Impossibile scaricare candele 1m ({e}), uso prezzo istantaneo.")

        # Calcolo PnL Floating corrente
        if direction == "LONG":
            floating_pnl_pct = (current_price - entry_price) / entry_price
        else:
            floating_pnl_pct = (entry_price - current_price) / entry_price

        floating_pnl_usdt = pos["size_usdt"] * floating_pnl_pct
        logger.info(
            f"Posizione Aperta {direction}: Prezzo {current_price:.2f} USDT | "
            f"PnL Flottante: {floating_pnl_pct*100:+.2f}% ({floating_pnl_usdt:+.2f} USDT) | "
            f"Intervallo High: {high_reach:.2f} / Low: {low_reach:.2f} | "
            f"TP: {pos['tp_price']:.2f} | SL: {pos['sl_price']:.2f} | "
            f"Durata: {elapsed_hours:.1f}h ({pos['holding_bars']:.1f}/{self.max_holding} barre 4H)"
        )

        exit_reason = None
        exit_price = current_price
        if direction == "LONG":
            if high_reach >= pos["tp_price"] and low_reach <= pos["sl_price"]:
                exit_reason = "STOP_LOSS"
                exit_price = pos["sl_price"]
            elif low_reach <= pos["sl_price"]:
                exit_reason = "STOP_LOSS"
                exit_price = pos["sl_price"]
            elif high_reach >= pos["tp_price"]:
                exit_reason = "TAKE_PROFIT"
                exit_price = pos["tp_price"]
            elif is_time_stop:
                exit_reason = "TIME_STOP"
                exit_price = current_price
        else: # SHORT
            if low_reach <= pos["tp_price"] and high_reach >= pos["sl_price"]:
                exit_reason = "STOP_LOSS"
                exit_price = pos["sl_price"]
            elif high_reach >= pos["sl_price"]:
                exit_reason = "STOP_LOSS"
                exit_price = pos["sl_price"]
            elif low_reach <= pos["tp_price"]:
                exit_reason = "TAKE_PROFIT"
                exit_price = pos["tp_price"]
            elif is_time_stop:
                exit_reason = "TIME_STOP"
                exit_price = current_price

        if exit_reason:
            self._close_trade(pos, exit_price, exit_reason)
        else:
            self.ledger.save()

    def _close_trade(self, pos: Dict[str, Any], exit_price: float, reason: str):
        direction = pos["direction"]
        entry_price = pos["entry_price"]

        if direction == "LONG":
            gross_return = (exit_price - entry_price) / entry_price
        else:
            gross_return = (entry_price - exit_price) / entry_price

        round_trip_fee = 2 * self.taker_fee
        net_return = gross_return - round_trip_fee
        fee_usdt = pos["size_usdt"] * round_trip_fee
        net_pnl_usdt = pos["size_usdt"] * net_return

        # Aggiorna ledger
        new_balance = self.ledger.balance + net_pnl_usdt
        self.ledger.state["current_balance"] = round(new_balance, 2)
        self.ledger.state["total_pnl_usdt"] = round(self.ledger.state.get("total_pnl_usdt", 0.0) + net_pnl_usdt, 2)
        self.ledger.state["total_fees_usdt"] = round(self.ledger.state.get("total_fees_usdt", 0.0) + fee_usdt, 2)

        trade_record = {
            "direction": direction,
            "entry_time": pos["entry_time"],
            "exit_time": datetime.now(timezone.utc).isoformat(),
            "entry_price": entry_price,
            "exit_price": exit_price,
            "reason": reason,
            "size_usdt": pos["size_usdt"],
            "net_pnl_usdt": round(net_pnl_usdt, 2),
            "net_return_pct": round(net_return * 100, 2),
            "fee_paid_usdt": round(fee_usdt, 2),
            "duration_bars": pos["holding_bars"]
        }

        self.ledger.state["closed_trades"].append(trade_record)
        self.ledger.state["current_position"] = None

        # Ricalcola Win Rate
        closed = self.ledger.state["closed_trades"]
        wins = [t for t in closed if t["net_pnl_usdt"] > 0]
        self.ledger.state["win_rate_pct"] = round((len(wins) / len(closed)) * 100, 1) if closed else 0.0
        self.ledger.save()

        logger.info(
            f"🔔 [PAPER TRADE CHIUSO] {direction} per {reason} a {exit_price:.2f} USDT! "
            f"PnL Netto: {net_pnl_usdt:+.2f} USDT ({net_return*100:+.2f}%) | "
            f"Nuovo Saldo: {new_balance:.2f} USDT | WinRate: {self.ledger.state['win_rate_pct']}%"
        )

        self.telegram.notify_trade_closed(
            direction=direction,
            symbol=self.symbol,
            reason=reason,
            entry_price=entry_price,
            exit_price=exit_price,
            net_pnl_usdt=round(net_pnl_usdt, 2),
            net_return_pct=round(net_return * 100, 2),
            new_balance=round(new_balance, 2),
            win_rate_pct=self.ledger.state["win_rate_pct"],
            duration_bars=pos["holding_bars"]
        )

    def print_status(self):
        """Stampa un report completo dello stato corrente del Paper Trading."""
        st = self.ledger.state
        print("\n" + "=" * 65)
        print("📊 REPORT PAPER TRADING REALNET BINANCE FUTURES (LIVE)")
        print("=" * 65)
        print(f"Saldo Iniziale:     {st.get('initial_balance', 100.0):.2f} USDT")
        print(f"Saldo Corrente:     {st.get('current_balance', 100.0):.2f} USDT")
        print(f"Profitto Netto Tot: {st.get('total_pnl_usdt', 0.0):+.2f} USDT ({((st.get('current_balance',100)-st.get('initial_balance',100))/st.get('initial_balance',100)*100):+.2f}%)")
        print(f"Commissioni Pagate: {st.get('total_fees_usdt', 0.0):.2f} USDT")
        print(f"Trade Chiusi:       {len(st.get('closed_trades', []))}")
        print(f"Win Rate:           {st.get('win_rate_pct', 0.0):.1f}%")
        print(f"Circuit Breaker:    {'ATTIVO (Trading Bloccato)' if st.get('circuit_breaker_triggered') else 'OK (Soglia 50.00 USDT)'}")

        closed = st.get('closed_trades', [])
        if closed:
            print("\nSTORICO ULTIMI TRADE CHIUSI:")
            for i, t in enumerate(closed[-5:], 1):
                pnl = t.get('net_pnl_usdt', 0.0)
                ret = t.get('net_return_pct', 0.0)
                reason = t.get('reason', '-')
                dur = t.get('duration_bars', 0.0)
                print(f"  {i}. {t['direction']:<5} | In: {t['entry_price']:.1f} -> Out: {t['exit_price']:.1f} | PnL: {pnl:+.2f} USDT ({ret:+.2f}%) | {reason} ({dur:.1f} barre 4H)")

        pos = st.get("current_position")
        if pos:
            live_price = self.get_realtime_ticker()
            ret = (live_price - pos['entry_price'])/pos['entry_price'] if pos['direction'] == 'LONG' else (pos['entry_price'] - live_price)/pos['entry_price']
            pnl = pos['size_usdt'] * ret

            entry_time = datetime.fromisoformat(pos["entry_time"])
            elapsed_h = (datetime.now(timezone.utc) - entry_time).total_seconds() / 3600.0
            elapsed_b = elapsed_h / 4.0

            print("\nPOSIZIONE ATTIVA:")
            print(f"  Direzione:     {pos['direction']}")
            print(f"  Prezzo Entry:  {pos['entry_price']:.2f} USDT")
            print(f"  Prezzo Live:   {live_price:.2f} USDT")
            print(f"  PnL Flottante: {ret*100:+.2f}% ({pnl:+.2f} USDT)")
            print(f"  Take Profit:   {pos['tp_price']:.2f}")
            print(f"  Stop Loss:     {pos['sl_price']:.2f}")
            print(f"  Size Trade:    {pos['size_usdt']:.2f} USDT")
            print(f"  Durata:        {elapsed_h:.1f} ore trascorse ({elapsed_b:.1f}/{self.max_holding} barre 4H)")
        else:
            print("\nPosizione Attiva:   NESSUNA (Stato Flat)")
        print("=" * 65 + "\n")

    def get_telegram_status_report(self) -> str:
        """Restituisce un report HTML completo dello stato attuale per Telegram."""
        st = self.ledger.state
        init_bal = st.get('initial_balance', 100.0)
        curr_bal = st.get('current_balance', 100.0)
        tot_pnl = st.get('total_pnl_usdt', 0.0)
        pnl_pct = ((curr_bal - init_bal) / init_bal) * 100.0 if init_bal else 0.0
        closed_count = len(st.get('closed_trades', []))
        win_rate = st.get('win_rate_pct', 0.0)
        cb_status = "🚨 ATTIVO" if st.get('circuit_breaker_triggered') else "✅ Normale (50$ min)"

        pos = st.get("current_position")
        if pos:
            try:
                live_price = self.get_realtime_ticker()
            except Exception:
                live_price = pos["entry_price"]

            direction = pos["direction"]
            entry_p = pos["entry_price"]
            ret = (live_price - entry_p)/entry_p if direction == 'LONG' else (entry_p - live_price)/entry_p
            pnl_flot = pos["size_usdt"] * ret

            entry_time = datetime.fromisoformat(pos["entry_time"])
            elapsed_h = (datetime.now(timezone.utc) - entry_time).total_seconds() / 3600.0
            pos_icon = "🟢" if direction == "LONG" else "🔴"

            pos_text = (
                f"\n📌 <b>Posizione Aperta:</b> {pos_icon} <b>{direction}</b>\n"
                f"• Entry: <code>{entry_p:.2f} USDT</code>\n"
                f"• Prezzo Live: <code>{live_price:.2f} USDT</code>\n"
                f"• PnL Flottante: <b>{pnl_flot:+.2f} USDT ({ret*100:+.2f}%)</b>\n"
                f"• TP: <code>{pos['tp_price']:.2f}</code> | SL: <code>{pos['sl_price']:.2f}</code>\n"
                f"• Durata: <code>{elapsed_h:.1f}h / 24h</code>\n"
            )
        else:
            pos_text = "\n📌 <b>Posizione Aperta:</b> ⚪ Nessuna (Stato Flat)\n"

        report = (
            "📊 <b>REPORT TCN TRADING BOT (AZURE)</b>\n"
            "━━━━━━━━━━━━━━━━━━━━\n"
            f"💵 <b>Saldo Attuale:</b> <code>{curr_bal:.2f} USDT</code>\n"
            f"📈 <b>PnL Totale:</b> <b>{tot_pnl:+.2f} USDT ({pnl_pct:+.2f}%)</b>\n"
            f"🏆 <b>Win Rate:</b> <code>{win_rate:.1f}%</code> ({closed_count} trade chiusi)\n"
            f"🛡 <b>Circuit Breaker:</b> {cb_status}\n"
            f"{pos_text}"
            "━━━━━━━━━━━━━━━━━━━━\n"
            f"⏰ <b>Ora UTC:</b> <code>{datetime.now(timezone.utc).strftime('%H:%M:%S - %d/%m/%Y')}</code>"
        )
        return report

    def get_telegram_position_report(self) -> str:
        """Restituisce dettagli della posizione attualmente attiva per Telegram."""
        pos = self.ledger.position
        if not pos:
            return "⚪ <b>Nessuna posizione attualmente aperta.</b> Il bot è in attesa di un segnale TCN."
        try:
            live_price = self.get_realtime_ticker()
        except Exception:
            live_price = pos["entry_price"]

        direction = pos["direction"]
        entry_p = pos["entry_price"]
        ret = (live_price - entry_p)/entry_p if direction == 'LONG' else (entry_p - live_price)/entry_p
        pnl_flot = pos["size_usdt"] * ret

        entry_time = datetime.fromisoformat(pos["entry_time"])
        elapsed_h = (datetime.now(timezone.utc) - entry_time).total_seconds() / 3600.0
        elapsed_b = elapsed_h / 4.0
        dir_icon = "🟢" if direction == "LONG" else "🔴"

        return (
            f"🎯 <b>DETTAGLI POSIZIONE ATTIVA</b>\n"
            "━━━━━━━━━━━━━━━━━━━━\n"
            f"📍 <b>Direzione:</b> {dir_icon} <b>{direction}</b>\n"
            f"💵 <b>Entry Price:</b> <code>{entry_p:.2f} USDT</code>\n"
            f"⚡ <b>Prezzo Live:</b> <code>{live_price:.2f} USDT</code>\n"
            f"📈 <b>PnL Flottante:</b> <b>{pnl_flot:+.2f} USDT ({ret*100:+.2f}%)</b>\n"
            f"📦 <b>Size:</b> <code>{pos['size_usdt']:.2f} USDT</code>\n"
            f"🎯 <b>Take Profit:</b> <code>{pos['tp_price']:.2f} USDT</code>\n"
            f"🛑 <b>Stop Loss:</b> <code>{pos['sl_price']:.2f} USDT</code>\n"
            f"⏱ <b>Durata:</b> <code>{elapsed_h:.1f} ore ({elapsed_b:.1f}/{self.max_holding} barre 4H)</code>\n"
            "━━━━━━━━━━━━━━━━━━━━"
        )

    def get_telegram_balance_report(self) -> str:
        """Restituisce il bilancio attuale e performance per Telegram."""
        st = self.ledger.state
        init_bal = st.get('initial_balance', 100.0)
        curr_bal = st.get('current_balance', 100.0)
        tot_pnl = st.get('total_pnl_usdt', 0.0)
        tot_fees = st.get('total_fees_usdt', 0.0)
        pnl_pct = ((curr_bal - init_bal) / init_bal) * 100.0 if init_bal else 0.0

        return (
            "💰 <b>SALDO & PERFORMANCE</b>\n"
            "━━━━━━━━━━━━━━━━━━━━\n"
            f"🏦 <b>Saldo Attuale:</b> <code>{curr_bal:.2f} USDT</code>\n"
            f"🌱 <b>Capitale Iniziale:</b> <code>{init_bal:.2f} USDT</code>\n"
            f"📊 <b>Profitto Netto:</b> <b>{tot_pnl:+.2f} USDT ({pnl_pct:+.2f}%)</b>\n"
            f"🧾 <b>Commissioni Taker:</b> <code>{tot_fees:.2f} USDT</code>\n"
            "━━━━━━━━━━━━━━━━━━━━"
        )

    def get_telegram_last_trades_report(self) -> str:
        """Restituisce lo storico degli ultimi trade conclusi per Telegram."""
        closed = self.ledger.state.get("closed_trades", [])
        if not closed:
            return "📜 <b>Nessun trade chiuso nello storico finora.</b>"

        last_trades = closed[-5:]
        lines = []
        for i, t in enumerate(reversed(last_trades), 1):
            icon = "🟢" if t["direction"] == "LONG" else "🔴"
            res_icon = "💰" if t["net_pnl_usdt"] >= 0 else "🔻"
            pnl_str = f"{t['net_pnl_usdt']:+.2f} USDT ({t['net_return_pct']:+.2f}%)"
            lines.append(
                f"{res_icon} <b>Trade #{len(closed)-i+1}:</b> {icon} {t['direction']} ({t['reason']})\n"
                f"   • In: <code>{t['entry_price']:.1f}</code> ➔ Out: <code>{t['exit_price']:.1f}</code>\n"
                f"   • PnL: <b>{pnl_str}</b> | Durata: <code>{t.get('duration_bars', 0):.1f} barre</code>"
            )

        return (
            f"📜 <b>ULTIMI {len(last_trades)} TRADE CONCLUSI:</b>\n"
            "━━━━━━━━━━━━━━━━━━━━\n"
            + "\n\n".join(lines)
            + "\n━━━━━━━━━━━━━━━━━━━━"
        )


def main():
    parser = argparse.ArgumentParser(description="Live Realnet Paper Trading Agent con TCN")
    parser.add_argument("--status", action="store_true", help="Mostra lo stato e lo storico del Paper Trading")
    parser.add_argument("--loop", action="store_true", help="Esegui in loop continuo monitorando ogni N minuti")
    parser.add_argument("--interval", type=int, default=15, help="Intervallo di controllo in minuti per la modalità loop (default: 15)")
    args = parser.parse_args()

    agent = RealnetPaperAgent()

    if args.status:
        agent.print_status()
        return

    if args.loop:
        # Avvia listener comandi interattivi Telegram se abilitato
        if agent.config.get("telegram", {}).get("enable_commands", True):
            agent.telegram.start_command_listener(agent)

        agent.telegram.notify_startup(
            symbol=agent.symbol,
            timeframe=agent.timeframe,
            balance=agent.ledger.balance,
            interval_min=args.interval,
            mode=agent.config.get("execution", {}).get("mode", "realnet_paper")
        )

        logger.info(f"Avvio monitoraggio continuo Realnet (controllo ogni {args.interval} minuti). Premi Ctrl+C per fermare.")
        try:
            while True:
                # Auto-recovery: assicura che il listener Telegram sia sempre attivo
                if agent.config.get("telegram", {}).get("enable_commands", True):
                    agent.telegram.ensure_listener_running(agent)

                try:
                    agent.evaluate()
                except Exception as e:
                    logger.error(f"❌ Errore imprevisto durante il ciclo di valutazione: {e}", exc_info=True)
                    agent.telegram.notify_error(f"Errore ciclo di valutazione: {e}")

                time.sleep(args.interval * 60)
        except KeyboardInterrupt:
            logger.info("Monitoraggio interrotto dall'utente.")
            agent.telegram.stop_command_listener()
    else:
        # Singola valutazione istantanea
        agent.evaluate()
        agent.print_status()


if __name__ == "__main__":
    main()
