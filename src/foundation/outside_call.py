# %%
import os
from pathlib import Path
from linearprobing.utils import load_model_without_module_prefix
from project_paths import FOUNDATION_CHECKPOINT_ROOT


def get_papagei(name="papagei0"): 
    ### Load Model ###
    model_config = {'base_filters': 32,
                'kernel_size': 3,
                'stride': 2,
                'groups': 1,
                'n_block': 18,
                'n_classes': 512,
                'n_experts': 3
                }
    
    if(name == "papagei1" or name == "papagei0"):
        from models.resnet import ResNet1DMoE
        model = ResNet1DMoE(in_channels=1, 
                    base_filters=model_config['base_filters'], 
                    kernel_size=model_config['kernel_size'],
                    stride=model_config['stride'],
                    groups=model_config['groups'],
                    n_block=model_config['n_block'],
                    n_classes=model_config['n_classes'],
                    n_experts=model_config['n_experts'])
    elif (name == "Mamba1" or name == "Mamba0"):
        from models.mamba import ManWhatCanISayMoE
        model = ManWhatCanISayMoE(in_channels=1, 
                base_filters=model_config['base_filters'], 
                n_block=model_config['n_block'],
                n_classes=model_config['n_classes'],
                n_experts=model_config['n_experts']
        )
    elif (name == "ppp"):
        from models.resnet import ResNet1D

        model = ResNet1D(in_channels=1, 
                    base_filters=model_config['base_filters'], 
                    kernel_size=model_config['kernel_size'],
                    stride=model_config['stride'],
                    groups=model_config['groups'],
                    n_block=model_config['n_block'],
                    n_classes=model_config['n_classes'])
    elif (name == "TFC"):
        from models.resnet import TFCResNet
        model = TFCResNet(model_config=model_config)
    else:
        raise ValueError(f"Unknown model name: {name}")

    return model

def get_param(model, name="papagei0"):
    TFC = str(FOUNDATION_CHECKPOINT_ROOT / "presets" / "tfc.pt")
    papagei0 = str(FOUNDATION_CHECKPOINT_ROOT / "presets" / "papagei_s.pt")
    ppp = str(FOUNDATION_CHECKPOINT_ROOT / "presets" / "papagei_p.pt")
    papagei1 = str(FOUNDATION_CHECKPOINT_ROOT / "presets" / "papagei1.pt")
    Mamba0 = str(FOUNDATION_CHECKPOINT_ROOT / "presets" / "mamba0.pt")
    Mamba1 = str(FOUNDATION_CHECKPOINT_ROOT / "presets" / "mamba1.pt")
    
    try:
        model_path = locals()[name]
    except KeyError:
        raise ValueError(f"Unknown model name: {name}")
    return load_model_without_module_prefix(model, model_path)

    
