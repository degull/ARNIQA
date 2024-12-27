""" import torch
from pathlib import Path
from scipy import stats
import argparse
from dotmap import DotMap
import yaml
import numpy as np
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[1]

print(f"PROJECT_ROOT set to: {PROJECT_ROOT}")

synthetic_datasets = ["live", "csiq", "tid2013", "kadid10k"]
authentic_datasets = ["flive", "spaq"]


def parse_args():
    parser = argparse.ArgumentParser(description="Training configuration")
    parser.add_argument('--config', type=str, required=True, help="Path to configuration YAML file")
    args = parser.parse_args()

    with open(args.config, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    return DotMap(config)

def parse_command_line_args(config: DotMap) -> DotMap:
    parser = argparse.ArgumentParser()

    def add_arguments(section, prefix=''):
        for key, value in section.items():
            full_key = f'{prefix}.{key}' if prefix else key
            if isinstance(value, dict):
                add_arguments(value, prefix=full_key)
            else:
                parser.add_argument(f'--{full_key}', type=type(value), default=value)

    add_arguments(config)
    args, _ = parser.parse_known_args()
    return DotMap(vars(args))

def parse_config(config_file_path: str) -> DotMap:
    with open(config_file_path, 'r', encoding='utf-8') as file:
        return DotMap(yaml.safe_load(file))

def merge_configs(config: DotMap, args: DotMap) -> DotMap:
    for key, value in args.items():
        keys = key.split('.')
        temp_config = config
        for subkey in keys[:-1]:
            temp_config = temp_config[subkey]
        temp_config[keys[-1]] = value
    return config

def save_checkpoint(model, path, epoch, srocc):
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "srocc": srocc,
    }
    torch.save(checkpoint, path / f"epoch_{epoch}_srocc_{srocc:.4f}.pth")

def calculate_srcc_plcc(proj_A, proj_B):
    proj_A = proj_A.detach().cpu().numpy()
    proj_B = proj_B.detach().cpu().numpy()
    assert proj_A.shape == proj_B.shape, "Shape mismatch between proj_A and proj_B"
    srocc, _ = stats.spearmanr(proj_A.flatten(), proj_B.flatten())
    plcc, _ = stats.pearsonr(proj_A.flatten(), proj_B.flatten())
    return srocc, plcc

def configure_degradation_model(batch, device):
    img_A_orig = batch["img_A_orig"].to(device)
    img_B_orig = batch["img_B_orig"].to(device)
    scale_factor = 0.5
    img_A_ds = F.interpolate(img_A_orig, scale_factor=scale_factor, mode='bilinear', align_corners=False)
    img_B_ds = F.interpolate(img_B_orig, scale_factor=scale_factor, mode='bilinear', align_corners=False)
    return img_A_orig, img_A_ds, img_B_orig, img_B_ds

import torch

def get_results(model, data_base_path, datasets, num_splits, phase, alpha, grid_search, crop_size, batch_size, num_workers, device):

    # Initialize empty results
    results = {
        "srocc": [],
        "plcc": []
    }

    # Iterate over datasets
    for dataset_name in datasets:
        print(f"Evaluating dataset: {dataset_name}")
        # Implement logic to evaluate the dataset
        # For example, load dataset, calculate SRCC and PLCC
        # Append results to `results`
    
    return results["srocc"], results["plcc"], None, None, None
 """

import argparse
import yaml
from dotmap import DotMap
from functools import reduce
from operator import getitem
from distutils.util import strtobool
from pathlib import Path
import re

PROJECT_ROOT = Path(__file__).absolute().parents[1].absolute()

import yaml

def parse_config(config_path):
    """
    Parse the YAML configuration file.
    Args:
        config_path (str): Path to the configuration file.
    Returns:
        dict: Parsed configuration.
    """
    with open(config_path, "r", encoding="utf-8") as file:  # UTF-8 인코딩 추가
        config = yaml.safe_load(file)
    return config



def parse_command_line_args(config: DotMap) -> DotMap:
    """Parse the command-line arguments"""
    parser = argparse.ArgumentParser()

    # Automatically add command-line arguments based on the config structure
    def add_arguments(section, prefix=''):
        for key, value in section.items():
            full_key = f'{prefix}.{key}' if prefix else key
            if isinstance(value, dict):
                add_arguments(value, prefix=full_key)
            else:
                # Check if the value is a list
                if isinstance(value, list):
                    # Convert list to comma-separated string
                    parser.add_argument(f'--{full_key}', default=value, type=type(value[0]), nargs='+', help=f'Value for {full_key}')
                else:
                    if type(value) == bool:
                        parser.add_argument(f'--{full_key}', default=value, type=strtobool, help=f'Value for {full_key}')
                    else:
                        parser.add_argument(f'--{full_key}', default=value, type=type(value), help=f'Value for {full_key}')

    add_arguments(config)

    args, _ = parser.parse_known_args()
    args = DotMap(vars(args), _dynamic=False)
    return args


def merge_configs(config: DotMap, args: DotMap) -> DotMap:
    """Merge the command-line arguments into the config. The command-line arguments take precedence over the config file
    :rtype: object
    """
    keys_to_modify = []

    def update_config(config, key, value):
        *keys, last_key = key.split('.')
        reduce(getitem, keys, config)[last_key] = value

    # Recursively merge command-line parameters into the config
    def get_updates(section, args, prefix=''):
        for key, value in section.items():
            full_key = f'{prefix}.{key}' if prefix else key
            if isinstance(value, dict):
                get_updates(value, args, prefix=full_key)
            elif getattr(args, full_key, None) or getattr(args, full_key, None) != getattr(section, key, None):
                keys_to_modify.append((full_key, getattr(args, full_key)))

    get_updates(config, args)

    for key, value in keys_to_modify:
        update_config(config, key, value)

    return config
