import os
import time
import logging
import threading
from datetime import datetime, timezone
from typing import Dict, Any, Optional, Callable
import requests

logger = logging.getLogger(__name__)


def load_env_file(env_path: str = ".env"):
    """Carica variabili chiave da un file .env se presente, senza richiedere dipendenze esterne."""
    if not os.path.exists(env_path):
        return
    try:
        with open(env_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                k = k.strip()
                v = v.strip().strip("'\"")
                if k not in os.environ:
                    os.environ[k] = v
    except Exception as e:
        logger.warning(f"Errore durante la lettura del file .env ({e})")


class TelegramNotifier:
    """
    Gestore di notifiche e comandi Telegram per il Trading Bot TCN.
    - Notifiche push automatiche per apertura/chiusura trade, startup e circuit breaker.
    - Ascolto comandi interattivi in background (/status, /pos, /balance, /last, /ping).
    - Sicurezza: risponde solo al chat_id autorizzato.
    """
    def __init__(
        self,
        token: Optional[str] = None,
        chat_id: Optional[str] = None,
        enabled: bool = True,
        notify_on_start: bool = True,
        notify_on_trade: bool = True,
        notify_on_close: bool = True,
        notify_on_circuit_breaker: bool = True
    ):
        load_env_file()

        self.token = token or os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
        raw_chat = chat_id or os.environ.get("TELEGRAM_CHAT_ID", "").strip()
        self.chat_id = int(raw_chat) if raw_chat else None

        self.enabled = enabled and bool(self.token) and bool(self.chat_id)
        self.notify_on_start = notify_on_start
        self.notify_on_trade = notify_on_trade
        self.notify_on_close = notify_on_close
        self.notify_on_circuit_breaker = notify_on_circuit_breaker

        self.base_url = f"https://api.telegram.org/bot{self.token}" if self.token else ""
        self._polling_thread: Optional[threading.Thread] = None
        self._stop_polling = threading.Event()

        if self.enabled:
            logger.info(f"📱 Notificatore Telegram abilitato per Chat ID: {self.chat_id}")
        else:
            logger.warning("⚠️ Notificatore Telegram DISABILITATO (Token o Chat ID non configurati).")

    def send_message(self, text: str, parse_mode: str = "HTML") -> bool:
        """Invia un messaggio Telegram formattato."""
        if not self.enabled:
            return False

        url = f"{self.base_url}/sendMessage"
        payload = {
            "chat_id": self.chat_id,
            "text": text,
            "parse_mode": parse_mode,
            "disable_web_page_preview": True
        }
        try:
            resp = requests.post(url, json=payload, timeout=10)
            if resp.status_code == 200:
                return True
            else:
                logger.error(f"Errore invio Telegram (HTTP {resp.status_code}): {resp.text}")
                return False
        except Exception as e:
            logger.error(f"Eccezione durante l'invio su Telegram: {e}")
            return False

    def notify_startup(self, symbol: str, timeframe: str, balance: float, interval_min: int, mode: str = "realnet_paper"):
        """Invia notifica di avvio del bot su Azure."""
        if not self.notify_on_start:
            return
        msg = (
            "🚀 <b>TRADING BOT TCN AVVIATO SU AZURE</b>\n"
            "━━━━━━━━━━━━━━━━━━━━\n"
            f"📊 <b>Asset:</b> <code>{symbol}</code>\n"
            f"⏱ <b>Timeframe:</b> <code>{timeframe}</code>\n"
            f"⚙️ <b>Modalità:</b> <code>{mode}</code>\n"
            f"💵 <b>Saldo Iniziale:</b> <code>{balance:.2f} USDT</code>\n"
            f"🔄 <b>Intervallo Loop:</b> <code>ogni {interval_min} minuti</code>\n"
            f"⏰ <b>Ora UTC:</b> <code>{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')}</code>\n"
            "━━━━━━━━━━━━━━━━━━━━\n"
            "💡 <i>Invia /status per richiedere un report in tempo reale.</i>"
        )
        self.send_message(msg)

    def notify_trade_opened(
        self,
        direction: str,
        symbol: str,
        entry_price: float,
        size_usdt: float,
        tp_price: float,
        sl_price: float,
        tp_pct: float,
        sl_pct: float,
        probs: Optional[Dict[str, float]] = None
    ):
        """Invia notifica di apertura posizione."""
        if not self.notify_on_trade:
            return

        icon = "🟢" if direction == "LONG" else "🔴"
        probs_text = ""
        if probs:
            probs_text = (
                "\n🧠 <b>Probabilità Rete TCN:</b>\n"
                f"   • P(Long): <code>{probs.get('long', 0.0)*100:.1f}%</code>\n"
                f"   • P(Short): <code>{probs.get('short', 0.0)*100:.1f}%</code>\n"
                f"   • P(Neutral): <code>{probs.get('neutral', 0.0)*100:.1f}%</code>\n"
            )

        msg = (
            f"{icon} <b>NUOVO TRADE APERTO [{direction}]</b>\n"
            "━━━━━━━━━━━━━━━━━━━━\n"
            f"📊 <b>Coppia:</b> <code>{symbol}</code>\n"
            f"📍 <b>Direzione:</b> <b>{direction}</b>\n"
            f"📥 <b>Entry Price:</b> <code>{entry_price:.2f} USDT</code>\n"
            f"📦 <b>Size Allocata:</b> <code>{size_usdt:.2f} USDT</code>\n"
            f"🎯 <b>Take Profit:</b> <code>{tp_price:.2f} USDT (+{tp_pct*100:.2f}%)</code>\n"
            f"🛑 <b>Stop Loss:</b> <code>{sl_price:.2f} USDT (-{sl_pct*100:.2f}%)</code>\n"
            f"{probs_text}"
            f"⏰ <b>Ora UTC:</b> <code>{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')}</code>"
        )
        self.send_message(msg)

    def notify_trade_closed(
        self,
        direction: str,
        symbol: str,
        reason: str,
        entry_price: float,
        exit_price: float,
        net_pnl_usdt: float,
        net_return_pct: float,
        new_balance: float,
        win_rate_pct: float,
        duration_bars: float
    ):
        """Invia notifica di chiusura posizione."""
        if not self.notify_on_close:
            return

        pnl_icon = "💰" if net_pnl_usdt >= 0 else "🔻"
        reason_icon = "🎯" if "PROFIT" in reason else ("🛑" if "LOSS" in reason else "⏳")
        dir_icon = "🟢" if direction == "LONG" else "🔴"

        duration_hours = duration_bars * 4.0

        msg = (
            f"{pnl_icon} <b>TRADE CHIUSO [{reason}]</b>\n"
            "━━━━━━━━━━━━━━━━━━━━\n"
            f"📊 <b>Asset:</b> <code>{symbol}</code>\n"
            f"📍 <b>Direzione:</b> {dir_icon} <b>{direction}</b>\n"
            f"🏁 <b>Motivo Uscita:</b> {reason_icon} <code>{reason}</code>\n"
            f"📥 <b>Entry:</b> <code>{entry_price:.2f} USDT</code>\n"
            f"📤 <b>Exit:</b> <code>{exit_price:.2f} USDT</code>\n"
            f"⏱ <b>Durata:</b> <code>{duration_bars:.1f} barre ({duration_hours:.1f}h)</code>\n"
            "────────────────────\n"
            f"📈 <b>PnL Netto:</b> <b>{net_pnl_usdt:+.2f} USDT ({net_return_pct:+.2f}%)</b>\n"
            f"🏦 <b>Nuovo Saldo:</b> <code>{new_balance:.2f} USDT</code>\n"
            f"🏆 <b>Win Rate Totale:</b> <code>{win_rate_pct:.1f}%</code>\n"
            f"⏰ <b>Ora UTC:</b> <code>{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')}</code>"
        )
        self.send_message(msg)

    def notify_circuit_breaker(self, current_balance: float, min_balance: float):
        """Invia notifica di scatto del circuit breaker."""
        if not self.notify_on_circuit_breaker:
            return
        msg = (
            "🚨 <b>CIRCUIT BREAKER ATTIVATO!</b> 🚨\n"
            "━━━━━━━━━━━━━━━━━━━━\n"
            f"⚠️ Saldo attuale: <b>{current_balance:.2f} USDT</b>\n"
            f"🛑 Soglia minima: <b>{min_balance:.2f} USDT</b>\n"
            "Trading automatico bloccato per tutela del capitale.\n"
            f"⏰ <b>Ora UTC:</b> <code>{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')}</code>"
        )
        self.send_message(msg)

    def notify_error(self, error_msg: str):
        """Invia avviso di errore eccezionale dal bot."""
        msg = (
            "⚠️ <b>AVVISO ERRORE BOT SU AZURE</b>\n"
            "━━━━━━━━━━━━━━━━━━━━\n"
            f"<code>{error_msg}</code>\n"
            f"⏰ <b>Ora UTC:</b> <code>{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')}</code>"
        )
        self.send_message(msg)

    def start_command_listener(self, agent: Any):
        """
        Avvia un thread in background per ascoltare i comandi inviati su Telegram
        e rispondere in tempo reale.
        """
        if not self.enabled:
            return

        def _polling_loop():
            offset = 0
            logger.info("📡 In ascolto comandi Telegram (/status, /pos, /balance, /last, /ping)...")
            while not self._stop_polling.is_set():
                try:
                    url = f"{self.base_url}/getUpdates"
                    params = {"offset": offset, "timeout": 20}
                    resp = requests.get(url, params=params, timeout=25)
                    if resp.status_code != 200:
                        time.sleep(3)
                        continue

                    data = resp.json()
                    for update in data.get("result", []):
                        offset = update["update_id"] + 1
                        message = update.get("message")
                        if not message:
                            continue

                        sender_id = message.get("chat", {}).get("id")
                        text = message.get("text", "").strip()

                        # Filtra solo comandi dal proprietario autorizzato
                        if sender_id != self.chat_id:
                            logger.warning(f"Ricevuto messaggio da utente non autorizzato: {sender_id}")
                            continue

                        self._handle_command(text, agent)
                except requests.exceptions.Timeout:
                    continue
                except Exception as e:
                    logger.debug(f"Eccezione temporanea polling Telegram: {e}")
                    time.sleep(3)

        self._stop_polling.clear()
        self._polling_thread = threading.Thread(target=_polling_loop, daemon=True, name="TelegramListener")
        self._polling_thread.start()

    def stop_command_listener(self):
        """Ferma il thread di ascolto comandi."""
        self._stop_polling.set()
        if self._polling_thread and self._polling_thread.is_alive():
            self._polling_thread.join(timeout=2)

    def _handle_command(self, cmd_text: str, agent: Any):
        """Smista ed esegue i comandi ricevuti."""
        cmd = cmd_text.split()[0].lower() if cmd_text else ""
        if "@" in cmd:
            cmd = cmd.split("@")[0]  # Rimuove eventuale tag @Trading_furo_bot

        if cmd in ["/start", "/help"]:
            reply = (
                "🤖 <b>Trading Bot TCN - Comandi Disponibili:</b>\n\n"
                "📊 /status - Report completo stato, saldo e posizioni\n"
                "🎯 /pos - Dettagli della posizione attiva (se presente)\n"
                "💵 /balance - Saldo attuale e performance %\n"
                "📜 /last - Ultimi trade chiusi\n"
                "🏓 /ping - Verifica se il bot su Azure è vivo\n"
                "ℹ️ /help - Mostra questa guida"
            )
            self.send_message(reply)

        elif cmd == "/status":
            reply = agent.get_telegram_status_report()
            self.send_message(reply)

        elif cmd in ["/pos", "/position"]:
            reply = agent.get_telegram_position_report()
            self.send_message(reply)

        elif cmd in ["/balance", "/saldo"]:
            reply = agent.get_telegram_balance_report()
            self.send_message(reply)

        elif cmd in ["/last", "/history"]:
            reply = agent.get_telegram_last_trades_report()
            self.send_message(reply)

        elif cmd == "/ping":
            now_utc = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
            reply = f"🏓 <b>Pong!</b> Il bot è attivo e operativo su Azure.\n⏰ <b>Timestamp:</b> <code>{now_utc}</code>"
            self.send_message(reply)

        else:
            self.send_message("❓ Comando non riconosciuto. Invia /help per la lista comandi.")
