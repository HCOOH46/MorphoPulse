
import pandas as pd 
import numpy as np
import os 
import torch 
import sys
import augmentations
import joblib
import torch.multiprocessing as mp
import wandb
import joblib
import torch_optimizer as toptim
from models.transformer import TransformerSimple
from models import efficientnet
from models.resnet import ResNet1DMoE
from models.mamba import ManWhatCanISayMoE
from models.transformer import TransformerMoE
from pytorch_metric_learning import losses, miners 
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
# from torch.optim import lr_scheduler
from datetime import datetime
torch.autograd.set_detect_anomaly(True)
from dataset import PPGDatasetLabelsArray, generate_dataloader
from tqdm import tqdm
from datetime import datetime
from torchvision import transforms
from training_distributed import ddp_setup, save_model
from training_pospair import harmonize_datasets
from project_paths import FOUNDATION_CHECKPOINT_ROOT, ensure_dir
import socket
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.wrap import (
    size_based_auto_wrap_policy,
    enable_wrap,
    wrap,
)

class AutomaticWeightedLoss(torch.nn.Module):
    """
    自动为多任务损失学习权重，避免手动调节 alpha。
    Reference: Multi-Task Learning Using Uncertainty to Weigh Losses for Scene Geometry and Semantics (CVPR 2018)
    """
    def __init__(self, num=2):
        super(AutomaticWeightedLoss, self).__init__()
        # 初始化 log_vars，参数是可学习的
        self.params = torch.nn.Parameter(torch.zeros(num, requires_grad=True))

    def forward(self, *losses):
        loss_sum = 0
        for i, loss in enumerate(losses):
            # 这里的 exp(-log_var) 等价于 1/sigma^2
            loss_sum += 0.5 / (self.params[i].exp()) * loss + self.params[i] * 0.5
        return loss_sum

def _find_free_port():
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port

def train_step(epoch, model, dataloader, criterion1, criterion2, optimizer, device, miner=None, use_sqi=True, use_sqi_loss=False, alpha=0.8, awl=None):
    
    """
    One training step for PaPaGei-S

    Args:
        epoch (int): Current step
        model (torch.nn.Module): Model to train
        dataloader (torch.utils.data.Dataloader): A training dataloader with signals
        criterion1 (torch.nn.<Loss>): Contrastive loss function
        criterion2 (torch.nn.<Loss>): Regression loss function
        optimizer (torch.optim): Optimizer to modify weights
        device (string): training device; use GPU
        miner (pytorch metric learning miner): Use a hard sample mining method
        use_sqi (boolean): To use signal quality index for mining
        use_sqi_loss (boolean): Multi-task loss uses SQI in addition to contrastive and ipa
        alpha (float): a value between 0 and 1 to decide the contribution of losses

    Returns:
        loss (float): The training loss for the step
    """
    
    model.to(device)
    model.train()
    dataloader.sampler.set_epoch(epoch)

    X, y = next(iter(dataloader))
    signal, svri, sqi, ipa = X.to(device), y[:, 0].to(device), y[:, 1].to(device), y[:, 2].to(device)

    embeddings, ipa_pred, sqi_pred, _ = model(signal)

    # Use a miner?
    if miner:
        # Compute hard pairs using quality or svri?
        if use_sqi:
            hard_pairs = miner(embeddings, sqi)
        else:
            hard_pairs = miner(embeddings, svri)
        contrastive_loss = criterion1(embeddings, svri, hard_pairs)
    else:
        contrastive_loss = criterion1(embeddings, svri)
    # Predict raw IPA values
    ipa_loss = criterion2(ipa_pred, ipa.unsqueeze(dim=-1))
    if awl is None:

        if use_sqi_loss:
            sqi_loss = criterion2(sqi_pred, sqi.unsqueeze(dim=-1))
            loss = alpha * contrastive_loss + (1 - alpha)/2 * ipa_loss + (1 - alpha)/2 * sqi_loss
        else:
            loss = alpha * contrastive_loss + (1 - alpha) * ipa_loss

    else:
        if use_sqi_loss:
            sqi_loss = criterion2(sqi_pred, sqi.unsqueeze(dim=-1))
            # 【修改点】：使用自动加权
            loss = awl(contrastive_loss, ipa_loss, sqi_loss)
        else:
            # 【修改点】：使用自动加权
            loss = awl(contrastive_loss, ipa_loss)


    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()

def training(model, epochs, train_dataloader, criterion1, criterion2, optimizer, device, directory, filename, scheduler=None, miner=None, wandb=None, use_sqi=True, use_sqi_loss=False, alpha=0.8, awl=None):

    """
    Training PaPaGei-S

    Args:
        model (torch.nn.Module): Model to train
        epochs (int): No. of epochs to train
        train_dataloader (torch.utils.data.Dataloader): A training dataloader with signals
        criterion1 (torch.nn.<Loss>): Contrastive loss function
        criterion2 (torch.nn.<Loss>): Regression loss function
        optimizer (torch.optim): Optimizer to modify weights
        device (string): training device; use GPU
        directory (string): directory to save model
        filename (string): model name for saving
        miner (pytorch metric learning miner): Use a hard sample mining method
        wandb (wandb): wandb object for experiment tracking
        use_sqi (boolean): To use signal quality index for mining
        use_sqi_loss (boolean): Multi-task loss uses SQI in addition to contrastive and ipa
        alpha (float): a value between 0 and 1 to decide the contribution of losses

    Returns:
        dict_log (dictionary): A dictionary log with metrics
    """

    dict_log = {'train_loss': []}
    best_loss = float('inf')
    save_check_interval = 1
    
    for step in tqdm(range(epochs), desc="Training Progress"):
        epoch_loss = train_step(epoch=step,
                                model=model,
                                dataloader=train_dataloader,
                                criterion1=criterion1,
                                criterion2=criterion2,
                                optimizer=optimizer,
                                device=device,
                                miner=miner,
                                use_sqi=use_sqi,
                                use_sqi_loss=use_sqi_loss,
                                alpha=alpha,
                                awl=awl
                                )
        if scheduler:
            scheduler.step()
        
        if wandb and device == "cuda:0":
            wandb.log({"Train Loss": epoch_loss})

        dict_log['train_loss'].append(epoch_loss)
        print(f"[{device}] Step: {step+1}/{epochs} | Train Loss: {epoch_loss:.4f}")

        if device == "cuda:0" and epoch_loss < best_loss and (step + 1) % save_check_interval == 0:
            best_loss = epoch_loss
            print(f"Saving model to: {directory}")
            content = f"step{step+1}_loss{epoch_loss:.4f}"
            save_model(model, directory, filename, content)

        if device == "cuda:0" and step == epochs - 1:
            print(f"Saving model to: {directory}")
            content = f"step{step+1}_loss{epoch_loss:.4f}"
            save_model(model, directory, filename, content)
            
        if device == "cuda:0" and step%50 == 0 and step > 0:
            print(f"Saving model to: {directory}")
            content = f"step{step+1}_loss{epoch_loss:.4f}"
            save_model(model, directory, filename, content)

    return dict_log

def main(rank, world_size, epochs, batch_size, from_ckpt=False):
    ddp_setup(rank, world_size)
    
    shuffle = True
    distributed = True
    lr = 0.0001
    prob_dictionary = {'g_p': 0.05, 'n_p': 0.05, 'w_p':0.05, 'f_p':0.05, 's_p':0.05, 'c_p':0.25}
    fs_target = 125
    bins_svri = 8
    bins_skewness = 0
    binary_ipa = False
    use_sqi = False
    use_sqi_loss = False
    alpha = 0.6
    dataset_name = "g" # "vital_mesa"
    use_awl = False
    #"mesa_mimic" #'all' #"mesa_mimic"

    simclr_transform = augmentations.get_transformations(g_p=prob_dictionary['g_p'],
                                            n_p=prob_dictionary['n_p'],
                                            w_p=prob_dictionary['w_p'],
                                            f_p=prob_dictionary['f_p'],
                                            s_p=prob_dictionary['s_p'],
                                            c_p=prob_dictionary['c_p']) 
    train_transform = transforms.Compose(simclr_transform)

    df = harmonize_datasets(dataset_name=dataset_name)

    dataset = PPGDatasetLabelsArray(df=df,
                                fs_target=fs_target, 
                                transform=train_transform,
                                bins_svri=bins_svri,
                                bins_skewness=bins_skewness,
                                binary_ipa=binary_ipa)
    # for i in range(0, len(dataset), len(dataset)//100):
    #     dataset.__getitem__(i)
    # exit()
    sampler = DistributedSampler(dataset, shuffle=shuffle)
    train_dataloader = DataLoader(dataset=dataset,
                    batch_size=batch_size,
                    num_workers=0,
                    sampler=sampler,
                    drop_last=True,
                    pin_memory=True,
)

    # # model = ResNet1D(in_channels=1, 
    # #             base_filters=model_config['base_filters'], 
    # #             kernel_size=model_config['kernel_size'],
    # #             stride=model_config['stride'],
    # #             groups=model_config['groups'],
    # #             n_block=model_config['n_block'],
    # #             n_classes=model_config['n_classes'],
    # #             use_mt_regression=True)

    # # model = ResNet1DMoE(in_channels=1, 
    # #             base_filters=model_config['base_filters'], 
    # #             kernel_size=model_config['kernel_size'],
    # #             stride=model_config['stride'],
    # #             groups=model_config['groups'],
    # #             n_block=model_config['n_block'],
    # #             n_classes=model_config['n_classes'],
    # #             n_experts=model_config['n_experts'])

    # model_config = {'base_filters': 32,
    #                 'kernel_size': 3,
    #                 'stride': 2,
    #                 'groups': 1,
    #                 'n_block': 18,
    #                 'n_classes': 512,
    #                 'n_experts': 3
    #                 }
    
    # model = ManWhatCanISayMoE(in_channels=1, 
    #             base_filters=model_config['base_filters'], 
    #             n_block=model_config['n_block'],
    #             n_classes=model_config['n_classes'],
    #             n_experts=model_config['n_experts'],
    #             # use_projection=True,
    #             # TEMP_DP=0.2,
    #             )
    model_config = {
            'd_model': 512, 'nhead': 8, 'num_layers': 6,
            'dim_feedforward': 2048, 'dropout': 0.1,
            'max_length': 256, 'n_classes': 512, 'n_experts': 3
        }
    model = TransformerMoE(
            in_channels=1,
            d_model=model_config['d_model'],
            nhead=model_config['nhead'],
            num_layers=model_config['num_layers'],
            dim_feedforward=model_config['dim_feedforward'],
            dropout=model_config['dropout'],
            max_length=model_config['max_length'],
            n_classes=model_config['n_classes'],
            n_experts=model_config['n_experts'],
        )


    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    device = "cuda:" + str(rank) 
    print(device)
    if from_ckpt:
        print("\033[32mLoading model from checkpoint...\033[0m")
        from linearprobing.utils import load_model_without_module_prefix
        model_path = os.environ.get(
            "FOUNDATION_INIT_CKPT",
            str(FOUNDATION_CHECKPOINT_ROOT / "init" / "foundation_init.pt"),
        )
        model = load_model_without_module_prefix(model, model_path)
    model.to(device)
    model = DDP(model, device_ids=[rank], output_device=rank, find_unused_parameters=True)
    criterion1 = losses.NTXentLoss()
    criterion2 = torch.nn.L1Loss()
    
    num_losses = 3 if use_sqi_loss else 2
    awl = AutomaticWeightedLoss(num=num_losses).to(device) if use_awl else None

    # 【修改点 2】: 将 awl 的参数加入 optimizer
    # 注意：这里的 params 列表要合并 model.parameters() 和 awl.parameters()
    if awl is None:
        optimizer = torch.optim.Adam(params=model.parameters(), lr=lr)
    else:
        optimizer = torch.optim.Adam(
            params=[
                {'params': model.parameters()},
                {'params': awl.parameters(), 'weight_decay': 0} # Loss权重不应该decay
            ], 
            lr=lr, 
            weight_decay=1e-5
        )

    # scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=0)
    # miner = miners.MultiSimilarityMiner()
    miner = None
    ### Experiment Tracking ###
    experiment_name = "transformer"
    try:
        name = f"mt_moe_{str(model_config['n_block'])}_{dataset_name}_"
    except:
        name = f"mt_moe_{str(model_config['dim_feedforward'])}_{dataset_name}_"
    group_name = "PPG"

    config = {"learning_rate": lr, 
         "epochs": epochs,
         "batch_size": batch_size,
         "augmentations": prob_dictionary,
         "bins_svri": bins_svri,
         "bins_skewness": bins_skewness,
         "binary_ipa": binary_ipa,
         "use_sqi":use_sqi,
         "alpha":alpha}

    # wandb.init(project=experiment_name,
    #         config=config | model_config, 
    #         name=name,
    #         group=group_name)

    # run_id = wandb.run.id
    run_id= "kwdjiu"
    time = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    model_filename = f'{experiment_name}_{name}_{run_id}_{time}'

    dict_log = training(model=model, 
                   train_dataloader=train_dataloader,
                   epochs=epochs,
                   criterion1=criterion1,
                   criterion2=criterion2,
                   optimizer=optimizer,
                   device=device,
                   directory=time,
                   filename=model_filename,
                   scheduler=None,
                   miner=miner,
                   wandb=None,
                   use_sqi=use_sqi,
                   alpha=alpha,
                   use_sqi_loss=use_sqi_loss,
                   awl=awl)
    # wandb.finish()
    joblib.dump(dict_log, ensure_dir(FOUNDATION_CHECKPOINT_ROOT / time) / f"{model_filename}_log.p")
    
    destroy_process_group()

if __name__ == "__main__":
    # torch.autograd.set_detect_anomaly(True)
    world_size = 1
    epochs =  5000#10000
    batch_size =  256 #128
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    if "MASTER_PORT" not in os.environ:
        os.environ["MASTER_PORT"] = str(_find_free_port())

    print(f"Using MASTER_ADDR={os.environ['MASTER_ADDR']} MASTER_PORT={os.environ['MASTER_PORT']}")
    mp.spawn(main, args=(world_size, epochs, batch_size, False), nprocs=world_size)
