
import torch
import joblib
import numpy as np 
import os 
import pandas as pd
from tqdm import tqdm
from project_paths import DATA_ROOT


def _downstream_dir(dataset_name: str):
    return DATA_ROOT / "downstream" / dataset_name

def delete_from_dictionary(path, wave_list):
    """
    Delete the given waveforms in wave_list from dictionary
    Saves spaces for redundant data processing

    Args:
        path (string): directory containing all .p dictionaries
        wave_list (list of strings): keys to dictionary to be deleted
    """
    filenames = os.listdir(path)
    for i in tqdm(range(len(filenames))):
        f = filenames[i]
        try:
            data = joblib.load(os.path.join(path, f))
            for wave in wave_list:
                del data[wave]
            joblib.dump(data, os.path.join(path, f))
        except Exception as e:
            print(f"{f} | {e}")

def load_model(model, filepath):
    """
    Load a PyTorch model from a specified file path.

    Args:
    model (torch.nn.Module): The PyTorch model instance to load the state dictionary into.
    filepath (str): The path from which the model will be loaded.

    Returns:
    model (torch.nn.Module): The model with the loaded state dictionary.
    """
    model.load_state_dict(torch.load(filepath))
    model.eval()  # Set the model to evaluation mode
    print(f"Model loaded from {filepath}")
    return model

def get_random_consecutive_files(directory, num_files):

    """
    Helper func to select files to merge.

    Args:
        directory (string): path to directory with p files
        num_files (int): number of files to combine
        
    Returns:
        selected_files (list): list of selected files to merge
    """
    
    all_files = [f for f in os.listdir(directory) if f.endswith('.p')]
    
    # Sort files
    file_numbers = sorted(int(f.split('.')[0]) for f in all_files if f.endswith('.p'))
    
    # Check if there are enough files to select the requested number of consecutive files
    if len(file_numbers) < num_files:
        raise ValueError("Not enough files to select the requested number of consecutive files.")
    
    # Select a random start index, ensuring there's enough room for the consecutive sequence
    # valid start ensures that the samples joined are disjoint. 
    # E.g., if 2 segments are to be joined, then 0-1 or 2-3 is valid but not 1-2. 
    valid_starts = np.arange(0, len(file_numbers), num_files)[:-1]
    start_index = np.random.choice(valid_starts, size=1)[0]
    
    # Find the corresponding files from the start index
    selected_files = [f"{file_numbers[i]}.p" for i in range(start_index, start_index + num_files)]
    
    return selected_files

def load_and_generate_longer_signal(directory, no_of_segments):

    """
    Merge selected segments to form longer signal.

    Args:
        directory (string): path to directory with p files
        no_of_segments (int): number of files to combine

    Returns:
        signal (np.array): Merged signal 
    """

    files = get_random_consecutive_files(directory=directory,
                                    num_files=no_of_segments)

    signal = [joblib.load(os.path.join(directory, f)) for f in files]

    return np.hstack(signal)

def get_data_info(dataset_name, prefix="", usecolumns=None):
    """
    This function returns meta data about the dataset such as user/ppg dataframes,
    column name of user_id, and the raw ppg directory.

    Args:
        dataset_name (string): string for selecting the dataset
        prefix (string): prefix for correct path
        usecolumns (list): quick loading if the .csv files contains many columns or if > 0.5GB

    Returns:
        df_train (pandas.DataFrame): training dataframe containing user id and segment id 
        df_val (pandas.DataFrame): validation dataframe containing user id and segment id 
        df_test (pandas.DataFrame): test dataframe containing user id and segment id 
        case_name (string): column name containing user id
        path (string): path to ppg directory
    """

    if dataset_name == "mesa":
        case_name = "mesaid"
        path = str(_downstream_dir("mesa") / "mesappg")

        if usecolumns is not None:
            usecols = np.concatenate([[case_name], usecolumns])
        else:
            usecols = None 

        df_train = pd.read_csv(_downstream_dir("mesa") / "train_clean.csv", usecols=usecols)
        df_val = pd.read_csv(_downstream_dir("mesa") / "val_clean.csv", usecols=usecols)
        df_test = pd.read_csv(_downstream_dir("mesa") / "test_clean.csv", usecols=usecols)

        df_train.loc[:, 'mesaid'] = df_train.mesaid.apply(lambda x: str(x).zfill(4))
        df_val.loc[:, 'mesaid'] = df_val.mesaid.apply(lambda x: str(x).zfill(4))
        df_test.loc[:, 'mesaid'] = df_test.mesaid.apply(lambda x: str(x).zfill(4))
        
    if dataset_name == "vital":
        path = str(_downstream_dir("vital") / "ppg")
        case_name = "subject_id"
        if usecolumns is not None:
            usecols = np.concatenate([[case_name], usecolumns])
        else:
            usecols = None 
        case_name = "subject_id"
        df_train = pd.read_csv(_downstream_dir("vital") / "train.csv", usecols=usecols)
        df_val = pd.read_csv(_downstream_dir("vital") / "val.csv", usecols=usecols)
        df_test = pd.read_csv(_downstream_dir("vital") / "test.csv", usecols=usecols)

        df_train.loc[:, 'subject_id'] = df_train[case_name].apply(lambda x: str(x))
        df_val.loc[:, 'subject_id'] = df_val[case_name].apply(lambda x: str(x))
        df_test.loc[:, 'subject_id'] = df_test[case_name].apply(lambda x: str(x))

    if dataset_name == "mimic":
        case_name = "SUBJECT_ID"
        path = str(_downstream_dir("mimic") / "ppg")
        if usecolumns is not None:
            usecols = np.concatenate([[case_name], usecolumns])
        else:
            usecols = None 
        df_train = pd.read_csv(_downstream_dir("mimic") / "train_clean.csv", usecols=usecols)
        df_val = pd.read_csv(_downstream_dir("mimic") / "val_clean.csv", usecols=usecols)
        df_test = pd.read_csv(_downstream_dir("mimic") / "test_clean.csv", usecols=usecols)


    if dataset_name == "sdb":
        case_name = "subject_id"
        path = str(_downstream_dir("sdb") / "ppg")
        if usecolumns is not None:
            usecols = np.concatenate([[case_name], usecolumns])
        else:
            usecols = None 

        df_train = pd.read_csv(_downstream_dir("sdb") / "train.csv", usecols=usecols)
        df_val = pd.read_csv(_downstream_dir("sdb") / "val.csv", usecols=usecols)
        df_test = pd.read_csv(_downstream_dir("sdb") / "test.csv", usecols=usecols)

        df_train.loc[:, case_name] = df_train[case_name].apply(lambda x:str(x).zfill(4))
        df_val.loc[:, case_name] = df_val[case_name].apply(lambda x:str(x).zfill(4))
        df_test.loc[:, case_name] = df_test[case_name].apply(lambda x:str(x).zfill(4))
    
    if dataset_name == "ppg-bp":
        case_name = "subject_id"
        path = str(_downstream_dir("ppg-bp") / "ppg")
        if usecolumns is not None:
            usecols = np.concatenate([[case_name], usecolumns])
        else:
            usecols = None 

        df_train = pd.read_csv(_downstream_dir("ppg-bp") / "train.csv", usecols=usecols)
        df_val = pd.read_csv(_downstream_dir("ppg-bp") / "val.csv", usecols=usecols)
        df_test = pd.read_csv(_downstream_dir("ppg-bp") / "test.csv", usecols=usecols)

        df_train.loc[:, case_name] = df_train[case_name].apply(lambda x:str(x).zfill(4))
        df_val.loc[:, case_name] = df_val[case_name].apply(lambda x:str(x).zfill(4))
        df_test.loc[:, case_name] = df_test[case_name].apply(lambda x:str(x).zfill(4))
        
    if dataset_name == "lexin_af":
        case_name = "subject_id"
        path = str(_downstream_dir("lexin_af") / "ppg")
        if usecolumns is not None:
            usecols = np.concatenate([[case_name], usecolumns])
        else:
            usecols = None 

        df_train = pd.read_csv(_downstream_dir("lexin_af") / "train.csv", usecols=usecols)
        df_val = pd.read_csv(_downstream_dir("lexin_af") / "val.csv", usecols=usecols)
        df_test = pd.read_csv(_downstream_dir("lexin_af") / "test.csv", usecols=usecols)

        df_train.loc[:, case_name] = df_train[case_name]
        df_val.loc[:, case_name] = df_val[case_name]
        df_test.loc[:, case_name] = df_test[case_name]
    
    if dataset_name == "bcg":
        case_name = "subject_id"
        path = str(_downstream_dir("bcg") / "ppg")
        if usecolumns is not None:
            usecols = np.concatenate([[case_name], usecolumns])
        else:
            usecols = None 

        df_train = pd.read_csv(_downstream_dir("bcg") / "train.csv", usecols=usecols)
        df_val = pd.read_csv(_downstream_dir("bcg") / "val.csv", usecols=usecols)
        df_test = pd.read_csv(_downstream_dir("bcg") / "test.csv", usecols=usecols)

        df_train.loc[:, case_name] = df_train[case_name]
        df_val.loc[:, case_name] = df_val[case_name]
        df_test.loc[:, case_name] = df_test[case_name]
    
    if dataset_name == "mimic_af":
        case_name = "subject_id"
        path = str(_downstream_dir("mimic_af") / "ppg")
        if usecolumns is not None:
            usecols = np.concatenate([[case_name], usecolumns])
        else:
            usecols = None 

        df_train = pd.read_csv(_downstream_dir("mimic_af") / "train.csv", usecols=usecols)
        df_val = pd.read_csv(_downstream_dir("mimic_af") / "val.csv", usecols=usecols)
        df_test = pd.read_csv(_downstream_dir("mimic_af") / "test.csv", usecols=usecols)

        df_train.loc[:, case_name] = df_train[case_name].apply(lambda x:str(x).zfill(4))
        df_val.loc[:, case_name] = df_val[case_name].apply(lambda x:str(x).zfill(4))
        df_test.loc[:, case_name] = df_test[case_name].apply(lambda x:str(x).zfill(4))
    
    if dataset_name == "ecsmp":
        case_name = "ID"
        path = str(_downstream_dir("ecsmp") / "ppg")
        if usecolumns is not None:
            usecols = np.concatenate([[case_name], usecolumns])
        else:
            usecols = None 
        
        df_train = pd.read_csv(_downstream_dir(dataset_name) / "train.csv", usecols=usecols)
        df_val = pd.read_csv(_downstream_dir(dataset_name) / "val.csv", usecols=usecols)
        df_test = pd.read_csv(_downstream_dir(dataset_name) / "test.csv", usecols=usecols)

        df_train.loc[:, case_name] = df_train[case_name].apply(lambda x:str(x).zfill(4))
        df_val.loc[:, case_name] = df_val[case_name].apply(lambda x:str(x).zfill(4))
        df_test.loc[:, case_name] = df_test[case_name].apply(lambda x:str(x).zfill(4))
    
    if dataset_name == "wesad":
        case_name = "subject_id"
        path = str(_downstream_dir("wesad") / "ppg")
        if usecolumns is not None:
            usecols = np.concatenate([[case_name], usecolumns])
        else:
            usecols = None 
        
        df_train = pd.read_csv(_downstream_dir(dataset_name) / "train.csv", usecols=usecols)
        df_val = pd.read_csv(_downstream_dir(dataset_name) / "val.csv", usecols=usecols)
        df_test = pd.read_csv(_downstream_dir(dataset_name) / "test.csv", usecols=usecols)
         # 关键修复：转换为字符串并补齐到4位（与 convert_keys_to_strings 逻辑一致）
        df_train.loc[:, case_name] = df_train[case_name].apply(lambda x:str(x).zfill(4))
        df_val.loc[:, case_name] = df_val[case_name].apply(lambda x:str(x).zfill(4))
        df_test.loc[:, case_name] = df_test[case_name].apply(lambda x:str(x).zfill(4))
    
    
    if dataset_name == "dalia":
        case_name = "subject_id"
        path = str(_downstream_dir("dalia") / "ppg")
        if usecolumns is not None:
            usecols = np.concatenate([[case_name], usecolumns])
        else:
            usecols = None 
        
        df_train = pd.read_csv(_downstream_dir(dataset_name) / "train.csv", usecols=usecols)
        df_val = pd.read_csv(_downstream_dir(dataset_name) / "val.csv", usecols=usecols)
        df_test = pd.read_csv(_downstream_dir(dataset_name) / "test.csv", usecols=usecols)
    
    if dataset_name == "marsh":
        case_name = "subjects"
        path = str(_downstream_dir("marsh") / "ppg")
        if usecolumns is not None:
            usecols = np.concatenate([[case_name], usecolumns])
        else:
            usecols = None 
        
        df_train = pd.read_csv(_downstream_dir(dataset_name) / "train.csv", usecols=usecols)
        df_val = pd.read_csv(_downstream_dir(dataset_name) / "val.csv", usecols=usecols)
        df_test = pd.read_csv(_downstream_dir(dataset_name) / "test.csv", usecols=usecols)

        df_train.loc[:, case_name] = df_train[case_name].apply(lambda x:str(x).zfill(4))
        df_val.loc[:, case_name] = df_val[case_name].apply(lambda x:str(x).zfill(4))
        df_test.loc[:, case_name] = df_test[case_name].apply(lambda x:str(x).zfill(4))
        
    if dataset_name == "numom2b":
        case_name = "subjects"
        path = str(_downstream_dir("numom2b") / "ppg")
        if usecolumns is not None:
            usecols = np.concatenate([[case_name], usecolumns])
        else:
            usecols = None 
        
        df_train = pd.read_csv(_downstream_dir(dataset_name) / "train.csv", usecols=usecols)
        df_val = pd.read_csv(_downstream_dir(dataset_name) / "val.csv", usecols=usecols)
        df_test = pd.read_csv(_downstream_dir(dataset_name) / "test.csv", usecols=usecols)
    
    if dataset_name == "bidmc":
        case_name = "subjects"
        path = str(_downstream_dir("bidmc") / "ppg")
        if usecolumns is not None:
            usecols = np.concatenate([[case_name], usecolumns])
        else:
            usecols = None 
        
        df_train = pd.read_csv(_downstream_dir(dataset_name) / "train.csv", usecols=usecols)
        df_val = pd.read_csv(_downstream_dir(dataset_name) / "val.csv", usecols=usecols)
        df_test = pd.read_csv(_downstream_dir(dataset_name) / "test.csv", usecols=usecols)

        df_train.loc[:, case_name] = df_train[case_name].apply(lambda x:str(x).zfill(2))
        df_val.loc[:, case_name] = df_val[case_name].apply(lambda x:str(x).zfill(2))
        df_test.loc[:, case_name] = df_test[case_name].apply(lambda x:str(x).zfill(2))
    
    if dataset_name == "mimicAF":
        case_name = "subjects"
        path = str(_downstream_dir("mimicAF") / "ppg")
        if usecolumns is not None:
            usecols = np.concatenate([[case_name], usecolumns])
        else:
            usecols = None 
        
        df_train = pd.read_csv(_downstream_dir(dataset_name) / "train.csv", usecols=usecols)
        df_val = pd.read_csv(_downstream_dir(dataset_name) / "val.csv", usecols=usecols)
        df_test = pd.read_csv(_downstream_dir(dataset_name) / "test.csv", usecols=usecols)
    
    if dataset_name == "vv":
        case_name = "subjects"
        path = str(_downstream_dir("vv") / "ppg")
        if usecolumns is not None:
            usecols = np.concatenate([[case_name], usecolumns])
        else:
            usecols = None 
        
        df_train = pd.read_csv(_downstream_dir(dataset_name) / "train.csv", usecols=usecols)
        df_val = pd.read_csv(_downstream_dir(dataset_name) / "val.csv", usecols=usecols)
        df_test = pd.read_csv(_downstream_dir(dataset_name) / "test.csv", usecols=usecols)

    return df_train, df_val, df_test, case_name, path
