# 🔬 Dati e Feature per Migliorare il Modello Senza Overfitting
### Guida alle Fonti Dati Ortogonali per il Trading Bot TCN (BTC/USDT 4H)

Per migliorare un modello di quantitative trading **senza cadere nella trappola dell'overfitting**, c'è una regola d'oro:

> **Non aggiungere altri indicatori derivati dal prezzo (RSI, MACD, Bande di Bollinger, Stocastico).**  
> Sono solo trasformazioni matematiche ridondanti di $OHLCV$. Aggiungere 10 indicatori non porta nuova informazione: crea solo *multicollinearità* e aumenta i parametri liberi della rete, facilitando la memorizzazione del rumore.

Per estrarre vero valore (alpha) servono **dati ortogonali**, cioè sorgenti informative che non derivano dalla serie storica di prezzo grezza.

Ecco le 4 categorie di dati a più alto impatto e a basso rischio di overfitting, integrabili gratuitamente:

---

### 1. Dati di Posizionamento dei Derivati (Crypto-Native) — *Impatto Altissimo*
Questi dati sono disponibili gratis direttamente tramite le API Futures di Binance (via CCXT o endpoint dedicati) e sono tra i predittori più forti in assoluto:

* **Variazione di Open Interest ($\Delta \text{OI}$):**
  * *Cos'è:* il numero totale di contratti aperti ancora attivi.
  * *Perché evita l'overfitting:* il prezzo ti dice *dove* sta andando il mercato, l'Open Interest ti dice *se c'è vero denaro fresco o se è solo una finta*.
    * Prezzo sale + OI sale $\to$ **Breakout solido**: nuove posizioni Long aggressive stanno entrando.
    * Prezzo sale + OI scende $\to$ **Short Squeeze / Rimbalzo debole**: chiusura forzata di posizioni corte, il rialzo esaurirà presto la spinta.
* **Taker Buy/Sell Ratio (Volume Aggressivo):**
  * *Cos'è:* il rapporto tra il volume acquistato a mercato (che colpisce l'Ask) e quello venduto a mercato (che colpisce il Bid).
  * *Perché è potente:* mostra la reale aggressività istantanea degli operatori (Order Flow), anticipando le mosse a 4H rispetto al semplice volume totale passivo.
* **Basis Spread (Futures vs Spot):**
  * La differenza percentuale tra il prezzo del Future Perpetual e il prezzo Spot reale su Binance. Un premio eccessivo indica euforia (leva Long satura), uno sconto indica pessimismo estremo (leva Short satura).

---

### 2. Dati Cross-Market / Macro (Spillover di Liquidità)
Bitcoin non vive in una bolla isolata, specialmente a barre di 4 ore:

* **Rapporto ETH/BTC (o Altcoin Rotation):**
  * Quando ETH/BTC sale, il mercato è in modalità *Risk-On* (propensione al rischio alta). Quando ETH/BTC crolla bruscamente, la liquidità si rifugia difensivamente in Bitcoin o USDT. Questa informazione aiuta enormemente la rete a capire se i breakout rialzisti sono sostenuti da tutto il comparto.
* **Correlazione con l'Indice del Dollaro (DXY) o Azionario USA (S&P 500 / NASDAQ):**
  * Su barre 4H, le sessioni che coincidono con l'apertura di Wall Street (14:30–21:00 UTC) risentono fortemente della forza del dollaro e del Nasdaq. Un semplice log-return a 4H del Nasdaq o del DXY aggiunge una dimensione macro non correlata al crypto trading.

---

### 3. Feature di Regime di Mercato (Trend vs Congestione)
Uno dei motivi per cui il bot perde soldi in certe fasi è che tenta di tradare segnali di trend durante mercati laterali e rumorosi.

* **Esponente di Hurst ($H$) o Choppiness Index:**
  * Misura se la serie temporale nelle ultime 64 barre è in regime di **persistenza (trend)** ($H > 0.5$) o di **mean-reversion / rumore casuale** ($H < 0.5$).
  * Permette alla rete di far prevalere la classe *Neutral* quando il mercato è in congestione, risparmiando decine di falsi trade.
* **Volatilità di Parkinson o Garman-Klass:**
  * A differenza dell'ATR (che usa solo l'escursione High-Low), sfrutta anche Open e Close per calcolare la reale efficienza della volatilità intraday.

---

### 4. Codifica Ciclica del Tempo (Sessioni di Mercato)
I mercati crypto hanno ritmi circadiani noti:
* **Sessione Asiatica** (00:00 – 08:00 UTC) $\to$ spesso accumulazione/range ristretto.
* **Sessione Londra** (08:00 – 14:00 UTC) $\to$ inizio volatilità e prese di liquidità.
* **Sessione New York** (14:00 – 21:00 UTC) $\to$ massimi volumi direzionali.
* **Weekend** $\to$ volumi bassi e frequenti trappole di liquidità.

Basta aggiungere 4 feature cicliche (calcolate con $\sin$ e $\cos$) per l'ora del giorno e il giorno della settimana:
$$\sin\left(\frac{2\pi \cdot \text{ora}}{24}\right), \quad \cos\left(\frac{2\pi \cdot \text{ora}}{24}\right)$$
$$\sin\left(\frac{2\pi \cdot \text{giorno}}{7}\right), \quad \cos\left(\frac{2\pi \cdot \text{giorno}}{7}\right)$$
La rete imparerà a dare meno peso ai falsi breakout del sabato notte rispetto ai breakout del martedì pomeriggio.

---

### ⚠️ Cosa EVITARE tassativamente
1. **Dati di Sentiment da Twitter/Reddit:** pieni di bot, manipolazioni e problemi gravissimi di allineamento temporale (lookahead bias).
2. **Prezzi in valore assoluto:** qualsiasi dato inserito nella TCN deve essere rigorosamente stazionario (log-return, percentuale o z-score normalizzato sul passato).

---

### 🎯 Da dove partire per il massimo risultato?
La combinazione a più alto rendimento con il minimo sforzo di implementazione è:  
👉 **Open Interest Delta + Taker Buy/Sell Ratio** (entrambi nativi di Binance Futures via CCXT).
