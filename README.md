# 📈 Quantitative Trading Bot con Temporal Convolutional Network (TCN)
### Perpetual Futures BTC/USDT — Timeframe 4H

Questo repository implementa un'architettura completa e modulare per il trading quantitativo sistematico su **contratti Perpetual Futures BTC/USDT (Binance Futures)** con timeframe a **4 ore (4H)**.

---

## 🏛️ Architettura e Metodologia Quantitativa

1. **Timeframe e Orizzonte:**
   - **Campionamento 4H**: $\approx 11.000$ barre su 5 anni di storico. Swing trading ad alta persistenza statistica (orizzonte 24h – 48h), con incidenza delle fee di scambio trascurabile rispetto all'ampiezza dei target ($\pm 2.5\% - 5.0\%$).
   - **Lookback Window**: 64 barre ($\approx 10.6$ giorni di contesto).
   - **Commissioni reali incorporate**: 0.05% taker in ingresso + 0.05% taker in uscita (0.10% round-trip) detratte su ogni trade.

2. **Feature Engineering Stazionaria:**
   - Log-returns su Close, High e Low ($\ln(P_t / P_{t-1})$).
   - High-Low Log Spread.
   - Volatilità normalizzata ($ATR_{14} / Close_t$).
   - Volume imbalance rispetto a SMA 20 e variazione logaritmica del volume.
   - Distanze percentuali da indicatori strutturali (SMA 50 e SMA 200).
   - Funding Rate a 8 ore allineato per evitare *lookahead bias*.
   - **Robust Scaling** con clipping 1%-99% fittato **esclusivamente sul Train Set**.

3. **Target Labeling — Metodologia Tripla Barriera (López de Prado):**
   - Tre classi discrete:
     - **Classe 1 (Long)**: tocca il Take Profit prima dello Stop Loss entro 6 barre.
     - **Classe 2 (Short)**: tocca il Take Profit ribassista prima dello Stop Loss entro 6 barre.
     - **Classe 0 (Neutral)**: rumore laterale o Stop Loss colpito prima del target.
   - **Risoluzione Worst-Case**: Se nella stessa barra vengono toccati sia TP che SL, si assume l'esecuzione pessimistica dello Stop Loss.

4. **Architettura Neurale TCN Causale:**
   - Convoluzioni 1D causali con padding asimmetrico sinistro $(k-1) \cdot d$.
   - Dilatazioni esponenziali $d \in [1, 2, 4, 8, 16, 32]$ con kernel size $k=3$.
   - **Campo recettivo esteso**: $1 + 2 \times 63 = 127$ barre (copre l'intero lookback di 64 barre con ampio margine).
   - `LayerNorm` e `Spatial Dropout` per contrastare l'overfitting.
   - **Rappresentazione $h_T$ sull'ultimo timestep**: Nessuna diluizione da Global Average Pooling su vettori causali incompleti.

5. **Split Dati con Embargo Temporale:**
   - **Train (70%)** $\to$ **Embargo (48h / 12 barre)** $\to$ **Validation (15%)** $\to$ **Embargo (48h)** $\to$ **Test Out-of-Sample (15%)**.

---

## 📁 Struttura della Repository

```
trading-bot-tcn/
├── config/
│   └── config.yaml              # Configurazione centralizzata parametri
├── src/
│   ├── __init__.py
│   ├── data_loader.py           # Download storico CCXT, caching Parquet e split con embargo
│   ├── features.py              # Calcolo feature stazionarie e classe FeatureScaler
│   ├── labeling.py              # Triple Barrier Labeling con worst-case conflict resolution
│   ├── dataset.py               # PyTorch TimeSeriesDataset (finestre scorrevoli)
│   ├── model.py                 # Definizione Causal TCN con LayerNorm e $h_T$ head
│   ├── train.py                 # Loop di training locale con Early Stopping e metriche
│   ├── backtester.py            # Simulatore realistico bar-by-bar con OCO e fee post-trade
│   └── live_agent.py            # Agente disaccoppiato per Binance Futures Testnet / Paper
├── notebooks/
│   └── kaggle_train.ipynb       # NOTEBOOK AUTOCONTENUTO PRONTO PER KAGGLE GPU
├── scripts/
│   └── generate_notebook.py     # Generatore automatico del notebook Kaggle
├── models/                      # Checkpoint pesi e scaler (.pth, .pkl)
├── data/                        # Dati storici scaricati (.parquet, .csv)
├── requirements.txt
└── README.md
```

---

## ⚡ Guida Rapida: Come Usare il Bot

### 1. Installazione Locale
Crea l'ambiente virtuale e installa le dipendenze:
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Addestramento su Kaggle (GPU T4 Gratuita) 🚀
Poiché l'addestramento beneficia dell'accelerazione GPU:
1. Apri **[Kaggle](https://www.kaggle.com/)** e crea un **New Notebook**.
2. Fai clic su `File` $\to$ `Import Notebook` e carica il file:
   ```
   notebooks/kaggle_train.ipynb
   ```
3. Nel pannello di destra su Kaggle (`Settings`):
   - Imposta **Accelerator**: **GPU T4 x2** (o P100).
   - Imposta **Internet**: **On** (necessario per scaricare i dati via CCXT).
4. Clicca su **Run All**.
5. Al termine dell'esecuzione, nella cartella di output (`/kaggle/working/`), troverai:
   - `tcn_best_model.pth`
   - `scaler.pkl`
   - `equity_curve.png`
6. Scarica `tcn_best_model.pth` e `scaler.pkl` e incollali nella cartella `models/` del tuo progetto locale.

---

### 3. Backtest Realistico Locale (Out-of-Sample)
Una volta copiati i file nella cartella `models/`, esegui il backtest bar-by-bar:
```bash
python src/backtester.py
```
Il simulatore:
- Esegue gli ordini sul prezzo di Open della barra successiva al segnale (zero lookahead).
- Applica TP, SL e Time Stop barra per barra.
- Detrae 0.10% di taker fee round-trip.
- Salva l'equity curve e il grafico del drawdown in `models/equity_curve.png`.

---

### 4. Paper Trading su Mercato REALE (Realnet Binance Futures)
L'agente supporta il **Paper Trading direttamente sul Mercato Reale di Binance Futures**:
- Riceve i prezzi, le candele e i funding rate autentici di produzione (zero discrepanze da testnet).
- **Zero soldi veri a rischio e zero API keys richieste.**
- Gestisce un conto simulato da **100$** con all-in 95% compounding, barriere ATR dinamiche e **circuit breaker a 50$**.
- Memorizza lo storico e le posizioni aperte in `data/paper_trading_ledger.json`.

```bash
# 1. Valuta il mercato live ed esegui un check istantaneo:
python src/live_agent.py

# 2. Mostra lo stato del conto, il PnL flottante e lo storico trade:
python src/live_agent.py --status

# 3. Avvia il monitoraggio continuo in background (es. controllo ogni 15 minuti):
python src/live_agent.py --loop --interval 15
```

---

### 5. Deploy su Azure VM / Cloud VPS (Esecuzione Continua 24/7)
Per far girare il bot in paper trading continuo per giorni senza dover tenere il computer locale acceso:

1. **Clona la repo sulla VM Azure**:
   ```bash
   git clone <URL_DELLA_TUA_REPO_GITHUB>
   cd trading-bot-tcn
   ```

2. **Crea l'ambiente virtuale e installa le dipendenze**:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

3. **Esecuzione persistente (per non interrompere il bot alla chiusura dell'SSH)**:
   - **Opzione A — Con `tmux` (Consigliata per interagire facilmente)**:
     ```bash
     sudo apt-get install -y tmux   # se non presente
     tmux new -s trading-bot
     source .venv/bin/activate
     python src/live_agent.py --loop --interval 15
     ```
     *(Premi `Ctrl+B`, rilascia e premi `D` per disconnetterti mantenendo il bot attivo in background).*  
     *(Per ricollegarti alla console del bot:* `tmux attach -t trading-bot`*).*

   - **Opzione B — Con `nohup`**:
     ```bash
     nohup python src/live_agent.py --loop --interval 15 > bot.log 2>&1 &
     # Per consultare i log in tempo reale:
     tail -f bot.log
     ```

4. **Verifica rapida delle metriche in qualsiasi momento**:
   ```bash
   python src/live_agent.py --status
   ```

---

## ⚙️ Parametri Principali (`config/config.yaml`)

| Sezione | Parametro | Descrizione |
|---|---|---|
| `market` | `symbol` | Coppia di trading (`BTC/USDT`) |
| `market` | `timeframe` | Frequenza delle barre (`4h`) |
| `market` | `taker_fee` | Commissione di scambio (`0.0005` = 0.05%) |
| `features` | `lookback_window` | Lunghezza finestra storica (`64` barre) |
| `labeling` | `prediction_horizon` | Orizzonte di holding massimo (`6` barre = 24h) |
| `model` | `channels` | Canali TCN `[32, 32, 48, 48, 64]` (~65k param.) |
| `decision` | `long_prob_threshold` | Probabilità minima ingresso Long (`0.38`) |
| `execution` | `mode` | `realnet_paper` (mercato reale simulato), `testnet`, `live` |
| `execution` | `circuit_breaker_min_balance` | Stop automatico tutela conto (`50.0` USDT) |
| `execution` | `compounding_allocation_pct` | Percentuale allocata per trade (`0.95` = 95%) |
| `decision` | `directional_ratio_threshold` | Rapporto minimo $P(\text{Long}) / P(\text{Short})$ (`1.4`) |
