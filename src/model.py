import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional


class CausalConv1d(nn.Module):
    """
    Convoluzione 1D rigorosamente causale con padding asimmetrico sinistro.
    Garantisce zero lookahead bias: ogni timestep t può dipendere esclusivamente
    dai timestep <= t.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dilation: int = 1):
        super().__init__()
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.padding = (kernel_size - 1) * dilation
        
        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=0,
            dilation=dilation
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Applica padding asimmetrico a sinistra: (padding_left, padding_right)
        x_padded = F.pad(x, (self.padding, 0))
        return self.conv(x_padded)


class TemporalResidualBlock(nn.Module):
    """
    Blocco residuo per TCN composto da:
    - 2 x (CausalConv1d -> Channel LayerNorm -> GELU -> Dropout)
    - Connessione residuale con proiezione 1x1 se in_channels != out_channels
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int,
        dropout: float = 0.15,
        use_layer_norm: bool = True
    ):
        super().__init__()
        self.use_layer_norm = use_layer_norm

        # Primo ramo
        self.conv1 = CausalConv1d(in_channels, out_channels, kernel_size, dilation)
        self.norm1 = nn.LayerNorm(out_channels) if use_layer_norm else nn.Identity()
        self.act1 = nn.GELU()
        self.drop1 = nn.Dropout(dropout)

        # Secondo ramo
        self.conv2 = CausalConv1d(out_channels, out_channels, kernel_size, dilation)
        self.norm2 = nn.LayerNorm(out_channels) if use_layer_norm else nn.Identity()
        self.act2 = nn.GELU()
        self.drop2 = nn.Dropout(dropout)

        # Proiezione residuale
        if in_channels != out_channels:
            self.res = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        else:
            self.res = nn.Identity()

        self.final_act = nn.GELU()

    def _apply_norm(self, x: torch.Tensor, norm_layer: nn.Module) -> torch.Tensor:
        if not self.use_layer_norm:
            return x
        # x: (B, C, L) -> permuta in (B, L, C) per LayerNorm -> ripermuta in (B, C, L)
        x = x.transpose(1, 2)
        x = norm_layer(x)
        return x.transpose(1, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = self.res(x)

        # Passaggio 1
        out = self.conv1(x)
        out = self._apply_norm(out, self.norm1)
        out = self.act1(out)
        out = self.drop1(out)

        # Passaggio 2
        out = self.conv2(out)
        out = self._apply_norm(out, self.norm2)
        out = self.act2(out)
        out = self.drop2(out)

        # Somma residuale
        return self.final_act(out + res)


class TemporalConvolutionalNetwork(nn.Module):
    """
    Rete Convoluzionale Temporale (TCN) Causale per Trading Quantitativo.
    
    Architettura:
    - Stacking di TemporalResidualBlocks con dilatazioni esponenziali (1, 2, 4, 8, 16, 32...)
    - Estrazione della rappresentazione dell'ultimo timestep h_T (nessuna diluizione da GAP)
    - Head di classificazione per predire le probabilità di [Neutral, Long, Short]
    """
    def __init__(
        self,
        num_features: int,
        channels: List[int] = [32, 32, 48, 48, 64],
        kernel_size: int = 3,
        dropout: float = 0.30,
        use_layer_norm: bool = True,
        num_classes: int = 3
    ):
        super().__init__()
        
        layers = []
        in_ch = num_features
        for i, out_ch in enumerate(channels):
            dilation = 2 ** i
            layers.append(
                TemporalResidualBlock(
                    in_channels=in_ch,
                    out_channels=out_ch,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    dropout=dropout,
                    use_layer_norm=use_layer_norm
                )
            )
            in_ch = out_ch

        self.network = nn.Sequential(*layers)
        
        # Classification Head: usa la feature map ricca all'ultimo timestep h_T
        last_ch = channels[-1]
        self.head = nn.Sequential(
            nn.Linear(last_ch, last_ch // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(last_ch // 2, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensore input di forma (Batch, Features, Lookback)
        Returns:
            Logits grezzi di forma (Batch, num_classes)
        """
        # (B, C_in, L) -> (B, C_last, L)
        features = self.network(x)
        
        # Seleziona l'ultimo step temporale t = -1 (h_T)
        h_t = features[:, :, -1]  # (B, C_last)
        
        # Output logits
        logits = self.head(h_t)   # (B, num_classes)
        return logits

    def predict_probabilities(self, x: torch.Tensor) -> torch.Tensor:
        """Restituisce le probabilità Softmax normalizzate."""
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            return F.softmax(logits, dim=-1)


class EnsembleTCN(nn.Module):
    """
    Deep Ensemble Bagging di N reti TCN causali indipendenti.
    Media le distribuzioni di probabilità Softmax per abbattere la varianza statistica,
    ridurre i falsi breakout da wick casuali ed elevare la qualità delle predizioni.
    """
    def __init__(self, models: List[TemporalConvolutionalNetwork]):
        super().__init__()
        self.models = nn.ModuleList(models)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits_list = [m(x) for m in self.models]
        return torch.stack(logits_list, dim=0).mean(dim=0)

    def predict_probabilities(self, x: torch.Tensor) -> torch.Tensor:
        self.eval()
        with torch.no_grad():
            probs_list = [m.predict_probabilities(x) for m in self.models]
            return torch.stack(probs_list, dim=0).mean(dim=0)

    def save(self, filepath: str):
        state = {
            "type": "ensemble_tcn",
            "num_models": len(self.models),
            "state_dicts": [m.state_dict() for m in self.models]
        }
        torch.save(state, filepath)

    @classmethod
    def load(
        cls,
        filepath: str,
        num_features: int,
        channels: List[int] = [32, 32, 48, 48, 64],
        kernel_size: int = 3,
        dropout: float = 0.0,
        use_layer_norm: bool = True,
        num_classes: int = 3,
        device: torch.device = torch.device("cpu")
    ):
        data = torch.load(filepath, map_location=device)
        if isinstance(data, dict) and data.get("type") == "ensemble_tcn":
            models = []
            for sd in data["state_dicts"]:
                m = TemporalConvolutionalNetwork(
                    num_features=num_features,
                    channels=channels,
                    kernel_size=kernel_size,
                    dropout=dropout,
                    use_layer_norm=use_layer_norm,
                    num_classes=num_classes
                ).to(device)
                m.load_state_dict(sd)
                m.eval()
                models.append(m)
            return cls(models)
        elif isinstance(data, dict) and "state_dicts" in data:
            models = []
            for sd in data["state_dicts"]:
                m = TemporalConvolutionalNetwork(
                    num_features=num_features,
                    channels=channels,
                    kernel_size=kernel_size,
                    dropout=dropout,
                    use_layer_norm=use_layer_norm,
                    num_classes=num_classes
                ).to(device)
                m.load_state_dict(sd)
                m.eval()
                models.append(m)
            return cls(models)
        else:
            # Fallback per modello singolo caricato come ensemble di 1
            m = TemporalConvolutionalNetwork(
                num_features=num_features,
                channels=channels,
                kernel_size=kernel_size,
                dropout=dropout,
                use_layer_norm=use_layer_norm,
                num_classes=num_classes
            ).to(device)
            m.load_state_dict(data)
            m.eval()
            return cls([m])


def count_parameters(model: nn.Module) -> int:
    """Calcola il numero totale di parametri addestrabili."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Test di consistenza tensoriale
    sample_input = torch.randn(8, 11, 64)  # Batch 8, 11 features, 64 barre lookback
    model = TemporalConvolutionalNetwork(num_features=11)
    output = model(sample_input)
    probs = model.predict_probabilities(sample_input)
    print("Modello TCN istanziato con successo:")
    print(f"  Parametri addestrabili: {count_parameters(model):,}")
    print(f"  Shape output logits: {output.shape}")
    print(f"  Shape output probs:  {probs.shape}")

    # Test EnsembleTCN
    ensemble = EnsembleTCN([model, model])
    ens_probs = ensemble.predict_probabilities(sample_input)
    print(f"Ensemble (2 modelli) output probs shape: {ens_probs.shape}")
