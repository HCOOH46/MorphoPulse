"""
Generate synthetic time series dataset using trained TimeVQVAE model

Usage: python generate_dataset.py --dataset_names Mesa_ppg_svri --n_samples 1000 --output_path ./generated_data.pt
"""
import argparse
from argparse import ArgumentParser
from typing import Union
import random

import torch
import wandb
import numpy as np
import os
from torch.utils.data import DataLoader
from preprocessing.data_pipeline import build_data_pipeline, build_custom_data_pipeline
from preprocessing.preprocess_ucr import DatasetImporterUCR, DatasetImporterCustom, DatasetImporterCustomTest

from evaluation.evaluation import Evaluation
from utils import get_root_dir, load_yaml_param_settings, str2bool

os.environ["WANDB_MODE"] = "offline"


def load_args():
    parser = ArgumentParser()
    parser.add_argument('--config', type=str, help="Path to the config data file.",
                        default=get_root_dir().joinpath('configs', 'config.yaml'))
    parser.add_argument('--dataset_names', nargs='+', help="e.g., Mesa_ppg_svri", required=True)
    parser.add_argument('--gpu_device_ind', default=0, type=int)
    parser.add_argument('--use_neural_mapper', type=str2bool, default=False, help='Use the neural mapper')
    parser.add_argument('--feature_extractor_type', type=str, default='papagei', help='papagei | supervised_fcn | rocket for evaluation.')
    parser.add_argument('--use_custom_dataset', type=str2bool, default=True, help='Using a custom dataset, then set it to True.')
    parser.add_argument('--sampling_batch_size', type=int, default=32, help='batch size when sampling.')
    parser.add_argument('--DDiT', type=str2bool, default=False, help='Use DDiT as the backbone of stage 2.')
    parser.add_argument('--n_samples', type=int, default=1000, help='Number of samples to generate')
    parser.add_argument('--output_path', type=str, default='./generated_dataset.pt', help='Output path for generated dataset')
    parser.add_argument('--generation_mode', type=str, default='conditional', choices=['conditional', 'unconditional', 'mixed'], 
                        help='Generation mode: conditional (class-balanced), unconditional, or mixed')
    parser.add_argument('--rand_seed', type=int, default=42, help='Random seed for reproducibility')
    return parser.parse_args()


def generate_dataset(config: dict,
                    dataset_name: str,
                    train_data_loader: DataLoader,
                    gpu_device_idx: int,
                    use_neural_mapper: bool,
                    feature_extractor_type: str,
                    use_custom_dataset: bool = False,
                    sampling_batch_size: int = 32,
                    DDiT: bool = False,
                    n_samples: int = 1000,
                    output_path: str = './generated_dataset.pt',
                    generation_mode: str = 'conditional',
                    rand_seed: int = 42):
    """
    Generate synthetic time series dataset and save as .pt file
    
    Args:
        generation_mode: 'conditional' (class-balanced), 'unconditional', or 'mixed'
    """
    # Set random seeds
    np.random.seed(rand_seed)
    torch.manual_seed(rand_seed)
    random.seed(rand_seed)
    
    n_classes = len(np.unique(train_data_loader.dataset.Y))
    print(f'Number of classes: {n_classes}')
    _, in_channels, input_length = train_data_loader.dataset.X.shape
    print(f'Data shape: channels={in_channels}, length={input_length}')
    
    # Initialize evaluation object
    print('Initializing model...')
    evaluation = Evaluation(dataset_name, in_channels, input_length, n_classes, gpu_device_idx, config, 
                          use_neural_mapper=use_neural_mapper,
                          feature_extractor_type=feature_extractor_type,
                          use_custom_dataset=use_custom_dataset, test=True, DDiT=DDiT).to(gpu_device_idx)
    
    generated_data = []
    generated_labels = []
    
    print(f'Generating {n_samples} samples using {generation_mode} mode...')
    
    if generation_mode == 'unconditional':
        # Generate all samples unconditionally
        (_, _, xhat), xhat_R = evaluation.sample(n_samples, 'unconditional', batch_size=sampling_batch_size)
        
        # Use neural mapper if available
        final_data = xhat_R if use_neural_mapper and xhat_R is not None else xhat
        generated_data = final_data
        
        # Assign random labels for unconditional generation
        generated_labels = np.random.randint(0, n_classes, size=n_samples)
        
    elif generation_mode == 'conditional':
        # Generate class-balanced samples
        samples_per_class = n_samples // n_classes
        remainder = n_samples % n_classes
        
        for cls_idx in range(n_classes):
            # Calculate samples for this class
            cls_samples = samples_per_class + (1 if cls_idx < remainder else 0)
            
            if cls_samples > 0:
                print(f'Generating {cls_samples} samples for class {cls_idx}...')
                (_, _, xhat_c), xhat_c_R = evaluation.sample(cls_samples, kind='conditional', 
                                                           class_index=cls_idx, batch_size=sampling_batch_size)
                
                # Use neural mapper if available
                final_data = xhat_c_R if use_neural_mapper and xhat_c_R is not None else xhat_c
                
                generated_data.append(final_data)
                generated_labels.extend([cls_idx] * cls_samples)
        
        # Concatenate all class data
        generated_data = np.concatenate(generated_data, axis=0)
        generated_labels = np.array(generated_labels)
        
    elif generation_mode == 'mixed':
        # Generate half conditional, half unconditional
        n_conditional = n_samples // 2
        n_unconditional = n_samples - n_conditional
        
        # Conditional part
        if n_conditional > 0:
            samples_per_class = n_conditional // n_classes
            remainder = n_conditional % n_classes
            
            conditional_data = []
            conditional_labels = []
            
            for cls_idx in range(n_classes):
                cls_samples = samples_per_class + (1 if cls_idx < remainder else 0)
                
                if cls_samples > 0:
                    print(f'Generating {cls_samples} conditional samples for class {cls_idx}...')
                    (_, _, xhat_c), xhat_c_R = evaluation.sample(cls_samples, kind='conditional', 
                                                               class_index=cls_idx, batch_size=sampling_batch_size)
                    
                    final_data = xhat_c_R if use_neural_mapper and xhat_c_R is not None else xhat_c
                    conditional_data.append(final_data)
                    conditional_labels.extend([cls_idx] * cls_samples)
            
            conditional_data = np.concatenate(conditional_data, axis=0)
            conditional_labels = np.array(conditional_labels)
        
        # Unconditional part
        if n_unconditional > 0:
            print(f'Generating {n_unconditional} unconditional samples...')
            (_, _, xhat), xhat_R = evaluation.sample(n_unconditional, 'unconditional', batch_size=sampling_batch_size)
            
            unconditional_data = xhat_R if use_neural_mapper and xhat_R is not None else xhat
            unconditional_labels = np.random.randint(0, n_classes, size=n_unconditional)
        
        # Combine conditional and unconditional
        if n_conditional > 0 and n_unconditional > 0:
            generated_data = np.concatenate([conditional_data, unconditional_data], axis=0)
            generated_labels = np.concatenate([conditional_labels, unconditional_labels], axis=0)
        elif n_conditional > 0:
            generated_data = conditional_data
            generated_labels = conditional_labels
        else:
            generated_data = unconditional_data
            generated_labels = unconditional_labels
    
    # Convert to torch tensors
    print('Converting to torch tensors...')
    try:        
        generated_data_tensor = torch.from_numpy(generated_data).float()  # [n_samples, 1, 2560]
    except:
        generated_data_tensor = generated_data
    try:
        generated_labels_tensor = torch.from_numpy(generated_labels).long()  # [n_samples]
    except:
        generated_labels_tensor = generated_labels
    
    # Ensure correct shape
    if generated_data_tensor.dim() == 2:
        generated_data_tensor = generated_data_tensor.unsqueeze(1)  # Add channel dimension
    
    # Shuffle the data
    if n_samples > 5000000:
        print('Dataset too large for in-memory shuffling. Skipping shuffle to prevent OOM.')
        print('Data is grouped by class. Please shuffle during loading (e.g. DataLoader shuffle=True).')
    else:
        print('Shuffling data...')
        indices = torch.randperm(generated_data_tensor.shape[0])
        generated_data_tensor = generated_data_tensor[indices]
        generated_labels_tensor = generated_labels_tensor[indices]

    
    # Print statistics
    print(f'\nGenerated dataset statistics:')
    print(f'Data shape: {generated_data_tensor.shape}')
    print(f'Labels shape: {generated_labels_tensor.shape}')
    print(f'Data type: {generated_data_tensor.dtype}')
    print(f'Labels type: {generated_labels_tensor.dtype}')
    print(f'Data range: [{generated_data_tensor.min().item():.4f}, {generated_data_tensor.max().item():.4f}]')
    
    # Print class distribution
    unique_labels, counts = torch.unique(generated_labels_tensor, return_counts=True)
    print(f'\nClass distribution:')
    for label, count in zip(unique_labels.tolist(), counts.tolist()):
        print(f'  Class {label}: {count} samples ({count/len(generated_labels_tensor)*100:.1f}%)')
    
    # Save dataset
    print(f'\nSaving dataset to {output_path}...')
    dataset_dict = {
        'data': generated_data_tensor,
        'labels': generated_labels_tensor,
        'metadata': {
            'dataset_name': dataset_name,
            'n_samples': len(generated_data_tensor),
            'n_classes': n_classes,
            'input_length': input_length,
            'in_channels': in_channels,
            'generation_mode': generation_mode,
            'use_neural_mapper': use_neural_mapper,
            'DDiT': DDiT,
            'rand_seed': rand_seed,
            'model_config': config
        }
    }
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    
    torch.save(dataset_dict, output_path)
    print(f'Dataset saved successfully!')
    
    # Verify saved data
    print('\nVerifying saved dataset...')
    loaded_data = torch.load(output_path)
    print(f'Loaded data shape: {loaded_data["data"].shape}')
    print(f'Loaded labels shape: {loaded_data["labels"].shape}')
    print(f'Metadata keys: {list(loaded_data["metadata"].keys())}')
    
    return output_path


if __name__ == '__main__':
    # Load arguments
    args = load_args()
    config = load_yaml_param_settings(args.config)
    
    for dataset_name in args.dataset_names:
        print(f'\n{"="*50}')
        print(f'Processing dataset: {dataset_name}')
        print(f'{"="*50}')
        
        # Data pipeline
        batch_size = config['evaluation']['batch_size']
        if not args.use_custom_dataset:
            dataset_importer = DatasetImporterUCR(dataset_name, **config['dataset'])
            train_data_loader, test_data_loader = [build_data_pipeline(batch_size, dataset_importer, config, kind) 
                                                 for kind in ['train', 'test']]
        else:
            dataset_importer = DatasetImporterCustomTest(dataset_name, **config['dataset'])
            train_data_loader, test_data_loader = [build_custom_data_pipeline(batch_size, dataset_importer, config, kind) 
                                                 for kind in ['train', 'test']]
        
        # Generate output filename
        # if len(args.dataset_names) > 1:
        base_name = os.path.splitext(args.output_path)[0]
        ext = os.path.splitext(args.output_path)[1]
        output_path = f"{base_name}_{dataset_name}{ext}"
        # else:
        #     output_path = args.output_path
        
        # Generate dataset
        try:
            generated_path = generate_dataset(
                config=config,
                dataset_name=dataset_name,
                train_data_loader=train_data_loader,
                gpu_device_idx=args.gpu_device_ind,
                use_neural_mapper=args.use_neural_mapper,
                feature_extractor_type=args.feature_extractor_type,
                use_custom_dataset=args.use_custom_dataset,
                sampling_batch_size=args.sampling_batch_size,
                DDiT=args.DDiT,
                n_samples=args.n_samples,
                output_path=output_path,
                generation_mode=args.generation_mode,
                rand_seed=args.rand_seed
            )
            
            print(f'\n✅ Successfully generated dataset for {dataset_name}')
            print(f'   Saved to: {generated_path}')
            
        except Exception as e:
            print(f'\n❌ Error generating dataset for {dataset_name}: {str(e)}')
            import traceback
            traceback.print_exc()
        
        # Clean memory
        torch.cuda.empty_cache()
    
    print(f'\n{"="*50}')
    print('Generation completed!')
    print(f'{"="*50}')