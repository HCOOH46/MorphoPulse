
import math

import torch

torch.autograd.set_detect_anomaly(True)


class PositionalEncoding(torch.nn.Module):
    """
    Ref: https://medium.com/@hunter-j-phillips/positional-encoding-7a93db4109e6
    """

    def __init__(self, d_model: int, dropout: float = 0.1, max_length: int = 10000):
        super().__init__()
        self.dropout = torch.nn.Dropout(p=dropout)

        pe = torch.zeros(max_length, d_model)
        k = torch.arange(0, max_length).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(k * div_term)
        pe[:, 1::2] = torch.cos(k * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:, : x.size(1)].requires_grad_(False)
        return self.dropout(x)


class TransformerSimple(torch.nn.Module):
    def __init__(self, model_config):
        super(TransformerSimple, self).__init__()

        encoder_layer = torch.nn.TransformerEncoderLayer(
            d_model=model_config['d_model'],
            nhead=model_config['nhead'],
            dim_feedforward=model_config['dim_feedforward'],
            dropout=model_config['trans_dropout'],
            batch_first=True,
        )

        self.transformer_encoder = torch.nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=model_config['num_layers'],
        )

        self.projection_head = torch.nn.Sequential(
            torch.nn.Linear(model_config['d_model'], model_config['h1']),
            torch.nn.BatchNorm1d(model_config['h1']),
            torch.nn.ReLU(),
            torch.nn.Linear(model_config['h1'], model_config['embedding_size']),
        )

    def forward(self, x):
        return self.projection_head(self.transformer_encoder(x.view(x.shape[0], -1)))


class LearnedPositionalEncoding(torch.nn.Module):
    def __init__(self, d_model: int, max_length: int = 128, dropout: float = 0.1):
        super().__init__()
        self.max_length = max_length
        self.dropout = torch.nn.Dropout(dropout)
        self.position_embedding = torch.nn.Parameter(torch.zeros(1, max_length, d_model))
        torch.nn.init.trunc_normal_(self.position_embedding, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.size(1)
        if seq_len > self.max_length:
            raise ValueError(
                f"Input sequence length {seq_len} exceeds max_length={self.max_length}."
            )
        return self.dropout(x + self.position_embedding[:, :seq_len, :])


class TransformerMoE(torch.nn.Module):
    """
    Transformer backbone aligned with the PaPaGei-S multi-task interface.
    """

    def __init__(
        self,
        in_channels: int = 1,
        d_model: int = 512,
        nhead: int = 8,
        num_layers: int = 6,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        max_length: int = 128,
        n_classes: int = 512,
        n_experts: int = 3,
        use_projection: bool = False,
        verbose: bool = False,
    ):
        super().__init__()
        self.verbose = verbose
        self.use_projection = use_projection
        self.n_experts = n_experts

        self.stem = torch.nn.Sequential(
            torch.nn.Conv1d(in_channels, d_model // 4, kernel_size=10, stride=5, padding=3),
            torch.nn.BatchNorm1d(d_model // 4),
            torch.nn.GELU(),
            torch.nn.Conv1d(d_model // 4, d_model, kernel_size=3, stride=2, padding=1),
            torch.nn.BatchNorm1d(d_model),
            torch.nn.GELU(),
        )

        self.position_encoding = LearnedPositionalEncoding(
            d_model=d_model,
            max_length=max_length,
            dropout=dropout,
        )

        encoder_layer = torch.nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.transformer_encoder = torch.nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=num_layers,
        )
        self.final_norm = torch.nn.LayerNorm(d_model)
        self.final_relu = torch.nn.ReLU(inplace=True)

        if self.use_projection:
            self.projector = torch.nn.Sequential(
                torch.nn.Linear(d_model, d_model),
                torch.nn.LayerNorm(d_model),
                torch.nn.GELU(),
                torch.nn.Linear(d_model, n_classes),
            )
        else:
            self.dense = torch.nn.Linear(d_model, n_classes)

        self.expert_layers_1 = torch.nn.ModuleList(
            [
                torch.nn.Sequential(
                    torch.nn.Linear(d_model, d_model // 2),
                    torch.nn.ReLU(),
                    torch.nn.Linear(d_model // 2, 1),
                )
                for _ in range(self.n_experts)
            ]
        )
        self.gating_network_1 = torch.nn.Sequential(
            torch.nn.Linear(d_model, self.n_experts),
            torch.nn.Softmax(dim=1),
        )

        self.expert_layers_2 = torch.nn.ModuleList(
            [
                torch.nn.Sequential(
                    torch.nn.Linear(d_model, d_model // 2),
                    torch.nn.ReLU(),
                    torch.nn.Dropout(0.3),
                    torch.nn.Linear(d_model // 2, 1),
                )
                for _ in range(self.n_experts)
            ]
        )
        self.gating_network_2 = torch.nn.Sequential(
            torch.nn.Linear(d_model, self.n_experts),
            torch.nn.Softmax(dim=1),
        )

    def forward(self, x: torch.Tensor):
        out = self.stem(x)
        if self.verbose:
            print('after stem', out.shape)

        out = out.transpose(1, 2)
        out = self.position_encoding(out)
        out = self.transformer_encoder(out)
        out = self.final_norm(out)
        out_backbone = self.final_relu(out.mean(dim=1))

        if self.verbose:
            print('backbone output', out_backbone.shape)

        out_class = self.projector(out_backbone) if self.use_projection else self.dense(out_backbone)

        expert_outputs_1 = torch.stack(
            [expert(out_backbone) for expert in self.expert_layers_1], dim=1
        )
        gate_weights_1 = self.gating_network_1(out_backbone)
        out_moe1 = torch.sum(gate_weights_1.unsqueeze(2) * expert_outputs_1, dim=1)

        expert_outputs_2 = torch.stack(
            [expert(out_backbone) for expert in self.expert_layers_2], dim=1
        )
        gate_weights_2 = self.gating_network_2(out_backbone)
        out_moe2 = torch.sum(gate_weights_2.unsqueeze(2) * expert_outputs_2, dim=1)

        return out_class, out_moe1, out_moe2, out_backbone
