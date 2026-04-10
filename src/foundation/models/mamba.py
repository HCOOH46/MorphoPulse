import torch
import torch.nn as nn
import os

os.environ['LD_LIBRARY_PATH'] = torch_lib_path + ':' + os.environ.get('LD_LIBRARY_PATH', '')
os.environ['C_INCLUDE_PATH'] = conda_prefix + '/include:' + os.environ.get('C_INCLUDE_PATH', '')


from mamba_ssm import Mamba2 as Mamba
from timm.models.layers import DropPath


from mamba_ssm import Mamba2 as Mamba

class MambaBlockDP(nn.Module):
    """ A single Mamba block with LayerNorm, Mamba, and a Feed-Forward Network (MLP). """
    def __init__(self, d_model, d_state, d_conv, expand, mlp_expand=2, dropout=0.1, droppath=0.):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.mamba = Mamba(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, mlp_expand * d_model),
            nn.GELU(),
            nn.Linear(mlp_expand * d_model, d_model),
        )
        self.dropout = nn.Dropout(dropout)
        self.drop_path = DropPath(droppath) if droppath > 0. else nn.Identity()

    def forward(self, x):
        # Pre-norm and residual connection for Mamba
        residual = x
        # x = self.norm1(x)
        # x = self.mamba(x)
        # x = residual + self.dropout(x)
        x = residual + self.drop_path(self.mamba(self.norm1(x)))

        # Pre-norm and residual connection for MLP
        residual = x
        x = self.norm2(x)
        x = self.mlp(x)
        x = residual + self.dropout(x)
        return x
    
class MambaBlock(nn.Module):
    """ A single Mamba block with LayerNorm, Mamba, and a Feed-Forward Network (MLP). """
    def __init__(self, d_model, d_state, d_conv, expand, mlp_expand=2, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.mamba = Mamba(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, mlp_expand * d_model),
            nn.GELU(),
            nn.Linear(mlp_expand * d_model, d_model),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Pre-norm and residual connection for Mamba
        residual = x
        x = self.norm1(x)
        x = self.mamba(x)
        x = residual + self.dropout(x)

        # Pre-norm and residual connection for MLP
        residual = x
        x = self.norm2(x)
        x = self.mlp(x)
        x = residual + self.dropout(x)
        return x

class MambaBackbone(nn.Module):
    """
    An enhanced Mamba-based backbone with stacked blocks, inspired by standard transformer architectures.
    """
    def __init__(self,
                 in_channels: int = 1,
                 embedding_dim: int = 512,
                 d_state: int = 16,
                 d_conv: int = 4,
                 expand: int = 2,
                 n_layers: int = 4,  # Add number of layers parameter
                 dropout: float = 0.1,
                 droppath: float = 0.): # Add dropout
        """
        Args:
            in_channels (int): Number of input channels.
            embedding_dim (int): The dimension of the model (d_model).
            d_state (int): The state space dimension (N) in Mamba.
            d_conv (int): The width of the 1D convolution in Mamba.
            expand (int): The expansion factor for the Mamba block.
            n_layers (int): The number of MambaBlocks to stack.
            dropout (float): Dropout rate.
        """
        super().__init__()
        self.embedding_dim = embedding_dim

        # Initial projection layer remains the same
        self.projection = nn.Sequential(
            nn.Conv1d(in_channels, embedding_dim // 4, kernel_size=10, stride=5, padding=3),
            nn.BatchNorm1d(embedding_dim // 4),
            nn.SiLU(),
            nn.Conv1d(embedding_dim // 4, embedding_dim, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(embedding_dim),
        )

        # Stack multiple MambaBlocks
        if droppath == 0.:
            self.layers = nn.ModuleList([
                MambaBlock(
                    d_model=embedding_dim,
                    d_state=d_state,
                    d_conv=d_conv,
                    expand=expand,
                    dropout=dropout,
                ) for _ in range(n_layers)
            ])
        else:
            self.layers = nn.ModuleList([
                MambaBlockDP(
                    d_model=embedding_dim,
                    d_state=d_state,
                    d_conv=d_conv,
                    expand=expand,
                    dropout=dropout,
                    droppath=droppath * i / (n_layers - 1)
                ) for i in range(n_layers)
            ])
        
        # Final normalization layer before pooling
        self.final_norm = nn.LayerNorm(embedding_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the enhanced Mamba backbone.
        """
        # 1. Project and downsample
        x = self.projection(x)
        x = x.transpose(1, 2)

        # 2. Pass through stacked MambaBlocks
        for layer in self.layers:
            x = layer(x)
            
        # 3. Final normalization
        x = self.final_norm(x)

        # 4. Global average pooling to get a fixed-size representation
        embedding = x.mean(dim=1)

        return embedding
    
class ManWhatCanISayMoE(nn.Module):
    """
    ResNet1D with Two Mixture of Experts (MoE) Regression Heads
    """
    def __init__(self, in_channels, base_filters, n_block, n_classes, 
                 d_state=16, d_conv=4, expand=2,
                 n_experts=2,  increasefilter_gap=4, use_bn=True, verbose=False,
                 use_projection=False, TEMP_DP=0.):
        super(ManWhatCanISayMoE, self).__init__()
        
        self.verbose = verbose 
        self.use_bn = use_bn
        self.use_projection = use_projection
        self.n_experts = n_experts

        out_channels = base_filters
        for i_block in range(1, n_block):
            if i_block % increasefilter_gap == 0:
                out_channels *= 2
        self.out_channels = out_channels

        self.encoder = MambaBackbone(
            in_channels=in_channels,
            embedding_dim=self.out_channels,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            droppath=TEMP_DP ###
        )
        
        self.final_bn = nn.BatchNorm1d(out_channels)
        self.final_relu = nn.ReLU(inplace=True)
        self.dense = nn.Linear(out_channels, n_classes)

        if self.use_projection:
            self.projector = nn.Sequential(
                nn.Linear(out_channels, 256),
                nn.LayerNorm(256),
                nn.ReLU(),
                nn.Linear(256, n_classes)
            )

        self.expert_layers_1 = nn.ModuleList([
            nn.Sequential(
                nn.Linear(out_channels, out_channels // 2),
                nn.ReLU(),
                nn.Linear(out_channels // 2, 1)
            ) for _ in range(self.n_experts)
        ])
        self.gating_network_1 = nn.Sequential(
            nn.Linear(out_channels, self.n_experts),
            nn.Softmax(dim=1)
        )

        self.expert_layers_2 = nn.ModuleList([
            nn.Sequential(
                nn.Linear(out_channels, out_channels // 2),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(out_channels // 2, 1)
            ) for _ in range(self.n_experts)
        ])
        self.gating_network_2 = nn.Sequential(
            nn.Linear(out_channels, self.n_experts),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        out = self.encoder(x)
        if self.verbose:
            print('encoder output', out.shape)

        # if self.use_bn:
        #     out = self.final_bn(out)
        out = self.final_relu(out)
        # out = out.mean(-1)
        if self.verbose:
            print('final bn relu', out.shape)

        if self.use_projection:
            out_class = self.projector(out)
        else:
            out_class = self.dense(out)

        expert_outputs_1 = torch.stack([expert(out) for expert in self.expert_layers_1], dim=1)
        gate_weights_1 = self.gating_network_1(out)
        out_moe1 = torch.sum(gate_weights_1.unsqueeze(2) * expert_outputs_1, dim=1)

        expert_outputs_2 = torch.stack([expert(out) for expert in self.expert_layers_2], dim=1)
        gate_weights_2 = self.gating_network_2(out)
        out_moe2 = torch.sum(gate_weights_2.unsqueeze(2) * expert_outputs_2, dim=1)

        return out_class, out_moe1, out_moe2, out
    
if __name__ == '__main__':
    # ================= 1. 实例化模型 (保持你的配置) =================
    from torchinfo import summary
    import torch
    
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    
    # 你的输入尺寸
    batch_size = 4
    seq_len = 1250
    input_size = (batch_size, 1, seq_len)
    _input = torch.randn(input_size).to(device)
    
    model_config = {
        'base_filters': 32,
        'kernel_size': 3,
        'stride': 2,
        'groups': 1,
        'n_block': 18,
        'n_classes': 512,
        'n_experts': 3
    }
    
    model = ManWhatCanISayMoE(in_channels=1, 
            base_filters=model_config['base_filters'], 
            n_block=model_config['n_block'],
            n_classes=model_config['n_classes'],
            n_experts=model_config['n_experts']
    )
    model.to(device)

    # ================= 2. 使用 torchview 画图 =================
    try:
        from torchview import draw_graph
        import graphviz
        
        # 设置 graph_name 会决定保存的文件名
        # expand_nested=True 会展开 MambaBlock 内部结构，设为 False 则只显示 Block 方块
        # depth=3 控制展开的深度，防止 Mamba 内部过于复杂导致图太大
        graph = draw_graph(
            model, 
            input_size=input_size, 
            device=device,
            expand_nested=True, 
            depth=3,
            graph_name='ManWhatCanISayMoE_Arch',
            save_graph=True,
            filename='model_architecture'  # 输出文件名为 model_architecture.png
        )
        
        print("模型结构图已保存为 model_architecture.png (和 .gv 文件)")
        
    except ImportError:
        print("请先安装 torchview 和 graphviz: pip install torchview graphviz")
    except Exception as e:
        print(f"绘图出错: {e}")
