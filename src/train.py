import os
import sys
import json
import logging
from typing import Dict, Any, Tuple

# Assicura che la root del progetto sia in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, f1_score, precision_score, confusion_matrix

from src.data_loader import load_config, get_market_data, get_chronological_splits
from src.features import extract_features, FeatureScaler, FEATURE_COLUMNS
from src.labeling import apply_triple_barrier_labeling
from src.dataset import TimeSeriesDataset
from src.model import TemporalConvolutionalNetwork, count_parameters

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def set_seed(seed: int = 42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def compute_class_weights(labels: np.ndarray, num_classes: int = 3) -> torch.Tensor:
    """Calcola i pesi inversi per CrossEntropyLoss bilanciata."""
    class_counts = np.bincount(labels, minlength=num_classes)
    total_samples = len(labels)
    # Pesi inversamente proporzionali alla frequenza
    weights = total_samples / (num_classes * np.maximum(class_counts, 1.0))
    weights_tensor = torch.tensor(weights, dtype=torch.float32)
    logger.info(f"Conteggio classi: {class_counts} -> Pesi calcolati: {weights_tensor.tolist()}")
    return weights_tensor


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device
) -> float:
    model.train()
    total_loss = 0.0
    for x_batch, y_batch in loader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)

        optimizer.zero_grad()
        logits = model(x_batch)
        loss = criterion(logits, y_batch)
        loss.backward()

        # Gradient clipping per prevenire esplosione dei gradienti
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item() * len(y_batch)

    return total_loss / len(loader.dataset)


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> Tuple[float, float, float, float, np.ndarray, np.ndarray]:
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for x_batch, y_batch in loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            logits = model(x_batch)
            loss = criterion(logits, y_batch)
            total_loss += loss.item() * len(y_batch)

            preds = torch.argmax(logits, dim=-1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(y_batch.cpu().numpy())

    avg_loss = total_loss / len(loader.dataset)
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)

    macro_f1 = f1_score(all_targets, all_preds, average="macro", zero_division=0)
    prec_long = precision_score(all_targets, all_preds, labels=[1], average="macro", zero_division=0)
    prec_short = precision_score(all_targets, all_preds, labels=[2], average="macro", zero_division=0)

    return avg_loss, macro_f1, prec_long, prec_short, all_preds, all_targets


def run_training_pipeline(config_path: str = "config/config.yaml") -> Dict[str, Any]:
    """Pipeline completa di addestramento, validazione e test out-of-sample."""
    cfg = load_config(config_path)
    set_seed(cfg["training"]["seed"])

    # 1. Caricamento dati
    df_raw = get_market_data(cfg)

    # 2. Feature Engineering stazionaria
    logger.info("Calcolo indicatori tecnici stazionari...")
    df_feat = extract_features(
        df_raw,
        sma_short=cfg["features"]["sma_short"],
        sma_long=cfg["features"]["sma_long"],
        atr_period=cfg["features"]["atr_period"],
        vol_period=cfg["features"]["volume_ma_period"]
    )

    # 3. Triple Barrier Labeling
    logger.info("Applicazione Triple Barrier Labeling...")
    df_labeled = apply_triple_barrier_labeling(
        df_feat,
        horizon=cfg["labeling"]["prediction_horizon"],
        profit_target_pct=cfg["labeling"]["profit_target_pct"],
        stop_loss_pct=cfg["labeling"]["stop_loss_pct"],
        taker_fee=cfg["market"]["taker_fee"],
        use_atr_dynamic=cfg["labeling"]["use_atr_dynamic_barrier"],
        atr_multiplier_target=cfg["labeling"]["atr_multiplier_target"],
        atr_multiplier_sl=cfg["labeling"]["atr_multiplier_sl"],
        worst_case_conflict=cfg["labeling"]["worst_case_conflict"]
    )

    # 4. Split cronologico con embargo
    train_df, val_df, test_df = get_chronological_splits(
        df_labeled,
        train_ratio=cfg["training"]["train_ratio"],
        val_ratio=cfg["training"]["val_ratio"],
        test_ratio=cfg["training"]["test_ratio"],
        embargo_bars=cfg["training"]["embargo_bars"]
    )

    # 5. Scaling (Fit rigorosamente SOLO sul Train)
    os.makedirs("models", exist_ok=True)
    scaler = FeatureScaler(method=cfg["features"].get("scaling_method", "robust"))
    scaler.fit(train_df)
    scaler.save("models/scaler.pkl")
    logger.info("Scaler fittato su Train e salvato in models/scaler.pkl")

    train_X = scaler.transform(train_df)
    val_X = scaler.transform(val_df)
    test_X = scaler.transform(test_df)

    train_y = train_df["label"].values
    val_y = val_df["label"].values
    test_y = test_df["label"].values

    lookback = cfg["features"]["lookback_window"]
    batch_size = cfg["training"]["batch_size"]

    train_ds = TimeSeriesDataset(train_X, train_y, lookback=lookback)
    val_ds = TimeSeriesDataset(val_X, val_y, lookback=lookback)
    test_ds = TimeSeriesDataset(test_X, test_y, lookback=lookback)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    # 6. Definizione Modello & Device
    device_name = cfg["training"]["device"]
    if device_name == "cuda" and not torch.cuda.is_available():
        logger.info("CUDA non disponibile. Utilizzo CPU.")
        device = torch.device("cpu")
    else:
        device = torch.device(device_name if torch.cuda.is_available() else "cpu")

    model = TemporalConvolutionalNetwork(
        num_features=len(FEATURE_COLUMNS),
        channels=cfg["model"]["channels"],
        kernel_size=cfg["model"]["kernel_size"],
        dropout=cfg["model"]["dropout"],
        use_layer_norm=cfg["model"]["use_layer_norm"],
        num_classes=cfg["model"]["num_classes"]
    ).to(device)

    logger.info(f"Modello inizializzato su {device} con {count_parameters(model):,} parametri.")

    # 7. Ottimizzazione e Loss pesata
    class_weights = compute_class_weights(train_y).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["training"]["learning_rate"],
        weight_decay=cfg["training"]["weight_decay"]
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=4
    )

    # 8. Training Loop con Early Stopping
    epochs = cfg["training"]["epochs"]
    patience = cfg["training"]["early_stopping_patience"]
    best_val_loss = float("inf")
    patience_counter = 0
    best_model_path = "models/tcn_best_model.pth"

    history = {
        "train_loss": [],
        "val_loss": [],
        "val_macro_f1": [],
        "val_prec_long": [],
        "val_prec_short": []
    }

    logger.info("Avvio del Training Loop...")
    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_f1, prec_l, prec_s, _, _ = evaluate(model, val_loader, criterion, device)
        scheduler.step(val_loss)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_macro_f1"].append(val_f1)
        history["val_prec_long"].append(prec_l)
        history["val_prec_short"].append(prec_s)

        logger.info(
            f"Epoca [{epoch:02d}/{epochs}] "
            f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
            f"Val F1: {val_f1:.4f} | Prec(Long): {prec_l:.3f} | Prec(Short): {prec_s:.3f}"
        )

        # Early Stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
            logger.info(f"  --> Miglior checkpoint salvato in {best_model_path} (Val Loss: {val_loss:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(f"Early stopping attivato all'epoca {epoch}.")
                break

    # 9. Valutazione Out-of-Sample sul Test Set
    logger.info("Caricamento del miglior modello per valutazione su Test Set Out-of-Sample...")
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    test_loss, test_f1, test_prec_l, test_prec_s, test_preds, test_targets = evaluate(
        model, test_loader, criterion, device
    )

    report_str = classification_report(
        test_targets, test_preds,
        target_names=["Neutral", "Long", "Short"],
        zero_division=0
    )
    conf_mat = confusion_matrix(test_targets, test_preds)

    logger.info("=" * 60)
    logger.info("RISULTATI OUT-OF-SAMPLE TEST SET:")
    logger.info(f"Test Loss: {test_loss:.4f} | Test Macro F1: {test_f1:.4f}")
    logger.info(f"Precision Long: {test_prec_l:.4f} | Precision Short: {test_prec_s:.4f}")
    logger.info(f"\nClassification Report:\n{report_str}")
    logger.info(f"\nMatrice di Confusione:\n{conf_mat}")
    logger.info("=" * 60)

    # Salvataggio storico
    with open("models/train_history.json", "w") as f:
        json.dump(history, f, indent=2)

    return {
        "best_val_loss": best_val_loss,
        "test_f1": test_f1,
        "test_prec_long": test_prec_l,
        "test_prec_short": test_prec_s,
        "classification_report": report_str
    }


if __name__ == "__main__":
    run_training_pipeline()
