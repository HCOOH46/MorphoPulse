
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
from models.resnet import ResNet1D
from pytorch_metric_learning import losses, miners 
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
from datetime import datetime
torch.autograd.set_detect_anomaly(True)
from dataset import PPGDatasetLabelsArray, generate_dataloader
from tqdm import tqdm
from datetime import datetime
from torchvision import transforms
from training_distributed import ddp_setup, save_model
from project_paths import DATA_ROOT, FOUNDATION_CHECKPOINT_ROOT, ensure_dir


def _pretrain_meta_csv(*parts):
    return DATA_ROOT.joinpath("pretrain", *parts)


def _pretrain_signal_dir(*parts):
    return str(DATA_ROOT.joinpath("pretrain_signals", *parts))

def harmonize_datasets(prefix=".", clean_ipa_only=False, dataset_name="g", sample_frac=1.0, random_seed=42):
    """
    Function to combine the pretraining dataset paths

    Args:
        prefix (String): Prefix for correct path
        clean_ipa_only (Boolean): To decide if we'd like to remove poor IPA signals
        dataset_name (String): Dataset combinations
    Returns:
        df (pandas.Dataframe): Dataframe with paths, frequency, etc. to feed into Torch Dataset
    """
    
    label = ['svri', 'skewness', 'ipa'] 
    df_vital = pd.read_csv(_pretrain_meta_csv("VitalDB", "train_clean.csv"), usecols=['VitalDBid', 'segments'] + label)
    df_mesa = pd.read_csv(_pretrain_meta_csv("MESA", "train_clean.csv"), usecols=['mesaid', 'segments'] + label)
    df_mimic = pd.read_csv(_pretrain_meta_csv("MIMIC", "train_clean.csv"), usecols=['SUBJECT_ID', 'segments'] + label)
    df_vg = pd.read_csv(_pretrain_meta_csv("VitalDB_g", "train_clean.csv"), usecols=['VitalDBid', 'segments'] + label)
    df_mag = pd.read_csv(_pretrain_meta_csv("MESA_g", "train_clean.csv"), usecols=['mesaid', 'segments'] + label)
    df_mig = pd.read_csv(_pretrain_meta_csv("MIMIC_g", "train_clean.csv"), usecols=['mesaid', 'segments'] + label)
    df_vg2 = pd.read_csv(_pretrain_meta_csv("VitalDB_g2", "train_clean.csv"), usecols=['VitalDBid', 'segments'] + label)
    df_mag2 = pd.read_csv(_pretrain_meta_csv("MESA_g2", "train_clean.csv"), usecols=['mesaid', 'segments'] + label)
    df_mig2 = pd.read_csv(_pretrain_meta_csv("MIMIC_g2", "train_clean.csv"), usecols=['SUBJECT_ID', 'segments'] + label)
    df_vg2_1 = pd.read_csv(_pretrain_meta_csv("VitalDB_g2_1", "train_clean.csv"), usecols=['mesaid', 'segments'] + label)
    df_mag_1 = pd.read_csv(_pretrain_meta_csv("MESA_g_1", "train_clean.csv"), usecols=['mesaid', 'segments'] + label)
    df_mag_2 = pd.read_csv(_pretrain_meta_csv("MESA_g_2", "train_clean.csv"), usecols=['mesaid', 'segments'] + label)
    df_mig2_1 = pd.read_csv(_pretrain_meta_csv("MIMIC_g2_1", "train_clean.csv"), usecols=['mesaid', 'segments'] + label)

        
    df_vital = df_vital.rename(columns={"VitalDBid": "case_id"})
    df_mesa = df_mesa.rename(columns={"mesaid": "case_id"})
    df_mimic = df_mimic.rename(columns={"SUBJECT_ID": "case_id"})
    df_vg = df_vg.rename(columns={"VitalDBid": "case_id"})
    df_mag = df_mag.rename(columns={"mesaid": "case_id"})
    df_mig = df_mig.rename(columns={"mesaid": "case_id"})
    df_vg2 = df_vg2.rename(columns={"VitalDBid": "case_id"})
    df_mag2 = df_mag2.rename(columns={"mesaid": "case_id"})
    df_mig2 = df_mig2.rename(columns={"SUBJECT_ID": "case_id"})
    df_vg2_1 = df_vg2_1.rename(columns={"mesaid": "case_id"})
    df_mag_1 = df_mag_1.rename(columns={"mesaid": "case_id"})
    df_mag_2 = df_mag_2.rename(columns={"mesaid": "case_id"})
    df_mig2_1 = df_mig2_1.rename(columns={"mesaid": "case_id"})

    df_vital.loc[:, 'case_id'] = df_vital.case_id.apply(lambda x: str(x).zfill(4))
    df_mesa.loc[:, 'case_id'] = df_mesa.case_id.apply(lambda x: str(x).zfill(4))
    df_mimic.loc[:, 'case_id'] = df_mimic.case_id.apply(lambda x: str(x).zfill(4))
    df_vg.loc[:, 'case_id'] = df_vg.case_id.apply(lambda x: str(x).zfill(4))
    df_mag.loc[:, 'case_id'] = df_mag.case_id.apply(lambda x: str(x).zfill(4))
    df_mig.loc[:, 'case_id'] = df_mig.case_id.apply(lambda x: str(x).zfill(4))
    df_vg2.loc[:, 'case_id'] = df_vg2.case_id.apply(lambda x: str(x).zfill(4))
    df_mag2.loc[:, 'case_id'] = df_mag2.case_id.apply(lambda x: str(x).zfill(4))
    df_mig2.loc[:, 'case_id'] = df_mig2.case_id.apply(lambda x: str(x).zfill(4))
    df_vg2_1.loc[:, 'case_id'] = df_vg2_1.case_id.apply(lambda x: str(x).zfill(4))
    df_mag_1.loc[:, 'case_id'] = df_mag_1.case_id.apply(lambda x: str(x).zfill(4))
    df_mag_2.loc[:, 'case_id'] = df_mag_2.case_id.apply(lambda x: str(x).zfill(4))
    df_mig2_1.loc[:, 'case_id'] = df_mig2_1.case_id.apply(lambda x: str(x).zfill(4))

    vital_path = _pretrain_signal_dir("PulseDB_Vital", "subject")
    mesa_path = _pretrain_signal_dir("MESA", "subject")
    mimic_path = _pretrain_signal_dir("PulseDB_MIMIC", "subject")
    vg_path = _pretrain_signal_dir("PulseDB_Vital", "generated")
    mag_path = _pretrain_signal_dir("MESA", "generated")
    mig_path = _pretrain_signal_dir("PulseDB_MIMIC", "generated")
    vg2_path = _pretrain_signal_dir("PulseDB_Vital", "generated_g2")
    mag2_path = _pretrain_signal_dir("MESA", "generated_g2")
    mig2_path = _pretrain_signal_dir("PulseDB_MIMIC", "generated_g2")
    vg2_1_path = _pretrain_signal_dir("PulseDB_Vital", "generated_g2_1")
    mag_1_path = _pretrain_signal_dir("MESA", "generated_g_1")
    mag_2_path = _pretrain_signal_dir("MESA", "generated_g_2")
    mig2_1_path = _pretrain_signal_dir("PulseDB_MIMIC", "generated_g2_1")
    
    
    df_vital.loc[:, 'path'] = np.repeat(vital_path, repeats=len(df_vital))
    df_mesa.loc[:, 'path'] = np.repeat(mesa_path, repeats=len(df_mesa))
    df_mimic.loc[:, 'path'] = np.repeat(mimic_path, repeats=len(df_mimic))
    df_vg.loc[:, 'path'] = np.repeat(vg_path, repeats=len(df_vg))
    df_mag.loc[:, 'path'] = np.repeat(mag_path, repeats=len(df_mag))
    df_mig.loc[:, 'path'] = np.repeat(mig_path, repeats=len(df_mig))
    df_vg2.loc[:, 'path'] = np.repeat(vg2_path, repeats=len(df_vg2))
    df_mag2.loc[:, 'path'] = np.repeat(mag2_path, repeats=len(df_mag2))
    df_mig2.loc[:, 'path'] = np.repeat(mig2_path, repeats=len(df_mig2))
    df_vg2_1.loc[:, 'path'] = np.repeat(vg2_1_path, repeats=len(df_vg2_1))
    df_mag_1.loc[:, 'path'] = np.repeat(mag_1_path, repeats=len(df_mag_1))
    df_mag_2.loc[:, 'path'] = np.repeat(mag_2_path, repeats=len(df_mag_2))
    df_mig2_1.loc[:, 'path'] = np.repeat(mig2_1_path, repeats=len(df_mig2_1))
    
    df_vital.loc[:, 'fs'] = np.repeat(500, repeats=len(df_vital))
    df_mesa.loc[:, 'fs'] = np.repeat(256, repeats=len(df_mesa))
    df_mimic.loc[:, 'fs'] = np.repeat(125, repeats=len(df_mimic))
    df_vg.loc[:, 'fs'] = np.repeat(500, repeats=len(df_vg))
    df_mag.loc[:, 'fs'] = np.repeat(256, repeats=len(df_mag))
    df_mig.loc[:, 'fs'] = np.repeat(125, repeats=len(df_mig))
    df_vg2.loc[:, 'fs'] = np.repeat(500, repeats=len(df_vg2))
    df_mag2.loc[:, 'fs'] = np.repeat(256, repeats=len(df_mag2))
    df_mig2.loc[:, 'fs'] = np.repeat(125, repeats=len(df_mig2))
    df_vg2_1.loc[:, 'fs'] = np.repeat(500, repeats=len(df_vg2_1))
    df_mag_1.loc[:, 'fs'] = np.repeat(256, repeats=len(df_mag_1))
    df_mag_2.loc[:, 'fs'] = np.repeat(256, repeats=len(df_mag_2))
    df_mig2_1.loc[:, 'fs'] = np.repeat(125, repeats=len(df_mig2_1))
    
    if sample_frac < 1.0 and dataset_name == "g":
        df_vg = df_vg.sample(frac=sample_frac, random_state=random_seed).reset_index(drop=True)
        df_mag = df_mag.sample(frac=sample_frac, random_state=random_seed).reset_index(drop=True)
        df_mig = df_mig.sample(frac=sample_frac, random_state=random_seed).reset_index(drop=True)
        print(f"Sampled {len(df_vg)} rows with sample_frac={sample_frac}, random_seed={random_seed}")
    
    if sample_frac > 1.0 and dataset_name == "g":
        df_vital = df_vital.sample(frac=1/sample_frac, random_state=random_seed).reset_index(drop=True)
        df_mesa = df_mesa.sample(frac=1/sample_frac, random_state=random_seed).reset_index(drop=True)
        df_mimic = df_mimic.sample(frac=1/sample_frac, random_state=random_seed).reset_index(drop=True)
        print(f"Sampled {len(df_vital)} rows with sample_frac={sample_frac}, random_seed={random_seed}")
    
    if dataset_name == "g_5":
        df = pd.concat((df_vital, df_mesa, df_mimic, df_vg, df_mag, df_mig, df_vg2, df_mag2, df_mig2, df_vg2_1, df_mag_1, df_mag_2, df_mig2_1))
    if dataset_name == "g_4":
        df = pd.concat((df_vital, df_mesa, df_mimic, df_vg2, df_mag2, df_mig2, df_vg2_1, df_mag_1, df_mag_2, df_mig2_1))   
    if dataset_name == "g_3":
        df = pd.concat((df_vital, df_mesa, df_mimic, df_vg, df_mag, df_mig, df_vg2, df_mag2, df_mig2))
    if dataset_name == "g":
        df = pd.concat((df_vital, df_mesa, df_mimic, df_vg, df_mag, df_mig))
    if dataset_name == "g_2":
        df = pd.concat((df_vital, df_mesa, df_mimic, df_vg2, df_mag2, df_mig2))
    if dataset_name == "all":
        df = pd.concat((df_vital, df_mesa, df_mimic))
    if dataset_name == "vital_mesa":
        df = pd.concat((df_vital, df_mesa))
    if dataset_name == "vital_mimic":
        df = pd.concat((df_vital, df_mimic))
    if dataset_name == "mesa_mimic":
        df = pd.concat((df_mesa, df_mimic))
    if dataset_name == "vital":
        df = df_vital
    if dataset_name == "mesa":
        df = df_mesa
    if dataset_name == "mimic":
        df = df_mimic
    if dataset_name == "pg":
        df = pd.concat((df_vg, df_mag, df_mig))
    df = df.reset_index()

    df = df[(df.svri > 0) & (df.svri < 2)]
    df = df[(df.ipa > -10) & (df.ipa < 10)]
    df = df[(df.skewness > -3) & (df.skewness < 3)]

    if clean_ipa_only:
        df = df[df.ipa != 0]
    
    # if sample_frac < 1.0:
    #     df = df.sample(frac=sample_frac, random_state=random_seed).reset_index(drop=True)
    #     print(f"Sampled {len(df)} rows with sample_frac={sample_frac}, random_seed={random_seed}")


    return df

def train_step(epoch, model, dataloader, criterion, optimizer, device, miner=None, use_sqi=True):

    """
    One training epoch for a model

    Args:
        epoch (int): Current step
        model (torch.nn.Module): Model to train
        dataloader (torch.utils.data.Dataloader): A training dataloader with signals
        criterion (torch.nn.<Loss>): Loss function to optimizer
        optimizer (torch.optim): Optimizer to modify weights
        device (string): training device; use GPU
        miner (pytorch metric learning miner): Use a hard sample mining method
        use_sqi (boolean): To use signal quality index for mining

    Returns:
        loss (float): The training loss for the step
    """
    
    model.to(device)
    model.train()
    dataloader.sampler.set_epoch(epoch)

    X, y = next(iter(dataloader))
    signal, svri, sqi = X.to(device), y[:, 0].to(device), y[:, 1].to(device)

    embeddings, _ = model(signal)

    # Use a miner?
    if miner:
        # Compute hard pairs using quality or svri?
        if use_sqi:
            hard_pairs = miner(embeddings, sqi)
        else:
            hard_pairs = miner(embeddings, svri)
        loss = criterion(embeddings, svri, hard_pairs)
    else:
        loss = criterion(embeddings, svri)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()

def training(model, epochs, train_dataloader, criterion, optimizer, device, directory, filename, miner=None, wandb=None):

    """
    Training a model with a different positive pair strategy.

    Args:
        model (torch.nn.Module): Model to train
        epochs (int): No. of epochs to train
        train_dataloader (torch.utils.data.Dataloader): A training dataloader with signals
        criterion (torch.nn.<Loss>): Loss function to optimizer
        optimizer (torch.optim): Optimizer to modify weights
        device (string): training device; use GPU
        directory (string): directory to save model
        filename (string): model name for saving
        miner (pytorch metric learning miner): Use a hard sample mining method
        wandb (wandb): wandb object for experiment tracking

    Returns:
        dict_log (dictionary): A dictionary log with metrics
    """

    dict_log = {'train_loss': []}
    best_loss = float('inf')
    
    for step in tqdm(range(epochs), desc="Training Progress"):
        epoch_loss = train_step(epoch=step,
                                model=model,
                                dataloader=train_dataloader,
                                criterion=criterion,
                                optimizer=optimizer,
                                device=device,
                                miner=miner)

        if wandb and device == "cuda:0":
            wandb.log({"Train Loss": epoch_loss})

        dict_log['train_loss'].append(epoch_loss)
        print(f"[{device}] Step: {step+1}/{epochs} | Train Loss: {epoch_loss:.4f}")

        if device == "cuda:0" and epoch_loss < best_loss:
            best_loss = epoch_loss
            print(f"Saving model to: {directory}")
            content = f"step{step+1}_loss{epoch_loss:.4f}"
            save_model(model, directory, filename, content)

        if device == "cuda:0" and step == epochs - 1:
            print(f"Saving model to: {directory}")
            content = f"step{step+1}_loss{epoch_loss:.4f}"
            save_model(model, directory, filename, content)

    return dict_log

def main(rank, world_size, epochs, batch_size):
    ddp_setup(rank, world_size)
    
    shuffle = True
    distributed = True
    lr = 0.0001
    prob_dictionary = {'g_p': 0.25, 'n_p': 0.0, 'w_p':0.0, 'f_p':0.0, 's_p':0.0, 'c_p':0.25}
    fs_target = 125
    bins_svri = 8
    bins_skewness = 5
    binary_ipa = False

    simclr_transform = augmentations.get_transformations(g_p=prob_dictionary['g_p'],
                                            n_p=prob_dictionary['n_p'],
                                            w_p=prob_dictionary['w_p'],
                                            f_p=prob_dictionary['f_p'],
                                            s_p=prob_dictionary['s_p'],
                                            c_p=prob_dictionary['c_p']) 
    train_transform = transforms.Compose(simclr_transform)

    df = harmonize_datasets()

    dataset = PPGDatasetLabelsArray(df=df,
                                fs_target=fs_target, 
                                transform=train_transform,
                                bins_svri=bins_svri,
                                bins_skewness=bins_skewness,
                                binary_ipa=binary_ipa)

    sampler = DistributedSampler(dataset, shuffle=shuffle)
    train_dataloader = DataLoader(dataset=dataset,
                    batch_size=batch_size,
                    num_workers=0,
                    sampler=sampler,
                    drop_last=True)

    # model_config = {'d_model': 1250,
    #            'nhead': 2,
    #            'dim_feedforward': 2048,
    #            'trans_dropout': 0.0,
    #            'proj_dropout': 0.0,
    #            'num_layers': 2,
    #            'h1': 1024,
    #            'embedding_size': 512}
    # model = TransformerSimple(model_config=model_config)

    model_config = {'base_filters': 32,
                    'kernel_size': 3,
                    'stride': 2,
                    'groups': 1,
                    'n_block': 18,
                    'n_classes': 512,
                    }

    model = ResNet1D(in_channels=1, 
                base_filters=model_config['base_filters'], 
                kernel_size=model_config['kernel_size'],
                stride=model_config['stride'],
                groups=model_config['groups'],
                n_block=model_config['n_block'],
                n_classes=model_config['n_classes'])

    # model_config = {'h1': 64,
    #                 'h2': 32,
    #                 'h3': 128,
    #                 'h4': 256,
    #                 'h5': 384,
    #                 'h6': 512,
    #                 'h7': 768,
    #                 'h8': 1024}

    # model = efficientnet.EfficientNetB0Base(in_channels=1, dict_channels=model_config)

    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    device = "cuda:" + str(rank) 
    print(device)
    model.to(device)
    model = DDP(model, device_ids=[rank], output_device=rank, find_unused_parameters=True)
    criterion = losses.NTXentLoss()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=lr)
    miner = miners.MultiSimilarityMiner()
    ### Experiment Tracking ###
    experiment_name = "resnet"
    name = "svri_skew"
    group_name = "PPG"

    config = {"learning_rate": lr, 
         "epochs": epochs,
         "batch_size": batch_size,
         "augmentations": prob_dictionary,
         "bins_svri": bins_svri,
         "bins_skewness": bins_skewness,
         "binary_ipa": binary_ipa}

    wandb.init(project=experiment_name,
            config=config | model_config, 
            name=name,
            group=group_name)

    run_id = wandb.run.id
    time = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    model_filename = f'{experiment_name}_{name}_{run_id}_{time}'

    dict_log = training(model=model, 
                   train_dataloader=train_dataloader,
                   epochs=epochs,
                   criterion=criterion,
                   optimizer=optimizer,
                   device=device,
                   directory=time,
                   filename=model_filename,
                   miner=miner,
                   wandb=wandb)
    wandb.finish()
    joblib.dump(dict_log, ensure_dir(FOUNDATION_CHECKPOINT_ROOT / time) / f"{model_filename}_log.p")
    
    destroy_process_group()

if __name__ == "__main__":
    torch.autograd.set_detect_anomaly(True)
    world_size = 8
    epochs = 15000
    batch_size = 128
    mp.spawn(main, args=(world_size, epochs, batch_size), nprocs=world_size)
