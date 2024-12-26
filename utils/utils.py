import torch
from pathlib import Path
from scipy import stats
import argparse
from dotmap import DotMap
import yaml
import numpy as np
import torch.nn.functional as F

# 프로젝트 루트 디렉토리 정의
PROJECT_ROOT = Path(__file__).resolve().parents[1]

print(f"PROJECT_ROOT set to: {PROJECT_ROOT}")


def parse_args():
    """
    Configuration parser for training and testing.
    """
    parser = argparse.ArgumentParser(description="Training configuration")
    parser.add_argument('--config', type=str, required=True, help="Path to configuration YAML file")

    args = parser.parse_args()

    # YAML 파일에서 설정 읽기 (UTF-8 인코딩 명시)
    with open(args.config, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    # DotMap으로 변환
    config = DotMap(config)
    return config


def parse_command_line_args(config: DotMap) -> DotMap:
    """
    커맨드 라인 인수를 설정에 병합.
    """
    parser = argparse.ArgumentParser()

    def add_arguments(section, prefix=''):
        for key, value in section.items():
            full_key = f'{prefix}.{key}' if prefix else key
            if isinstance(value, dict):
                add_arguments(value, prefix=full_key)
            else:
                parser.add_argument(f'--{full_key}', type=type(value), default=value, help=f'{full_key} parameter')

    add_arguments(config)

    args, _ = parser.parse_known_args()
    args = DotMap(vars(args))
    return args


def parse_config(config_file_path: str) -> DotMap:
    """
    YAML 설정 파일을 DotMap으로 파싱.
    """
    with open(config_file_path, 'r', encoding='utf-8') as file:
        config = yaml.safe_load(file)
    return DotMap(config)


def merge_configs(config: DotMap, args: DotMap) -> DotMap:
    """
    커맨드 라인 인수를 YAML 설정에 병합.
    """
    for key, value in args.items():
        keys = key.split('.')
        temp_config = config
        for subkey in keys[:-1]:
            temp_config = temp_config[subkey]
        temp_config[keys[-1]] = value
    return config


def save_checkpoint(model, path, epoch, srocc):
    """
    모델의 체크포인트를 저장합니다.
    모델의 상태 및 epoch, SROCC 값을 포함하여 저장합니다.
    
    Args:
        model (nn.Module): 저장할 모델.
        path (Path): 체크포인트를 저장할 경로.
        epoch (int): 현재 에폭.
        srocc (float): SRCC 값.
    """
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "srocc": srocc,
    }
    torch.save(checkpoint, path / f"epoch_{epoch}_srocc_{srocc:.4f}.pth")


""" def calculate_srcc_plcc(proj_A, proj_B, mos):
    proj_A = proj_A.view(-1, proj_A.shape[-1]).detach().cpu().numpy()  # [batch_size * num_crops, 128]
    proj_B = proj_B.view(-1, proj_B.shape[-1]).detach().cpu().numpy()  # [batch_size * num_crops, 128]

    batch_size = mos.size(0)
    num_crops = proj_A.shape[0] // batch_size  # Number of crops per batch
    mos = mos.unsqueeze(1).repeat(1, num_crops).view(-1).detach().cpu().numpy()  # [batch_size * num_crops]

    srocc, _ = stats.spearmanr(proj_A.flatten(), mos.flatten())

    plcc, _ = stats.pearsonr(proj_B.flatten(), mos.flatten())

    return srocc, plcc
 """


def calculate_srcc_plcc(proj_A, proj_B):
    proj_A = proj_A.detach().cpu().numpy()
    proj_B = proj_B.detach().cpu().numpy()
    assert proj_A.shape == proj_B.shape, "Shape mismatch between proj_A and proj_B"
    srocc, _ = stats.spearmanr(proj_A.flatten(), proj_B.flatten())
    plcc, _ = stats.pearsonr(proj_A.flatten(), proj_B.flatten())
    return srocc, plcc


def configure_degradation_model(batch, device):
    """
    배치 데이터에서 원본 및 다운샘플링 이미지를 생성합니다.
    """
    img_A_orig = batch["img_A_orig"].to(device)
    img_B_orig = batch["img_B_orig"].to(device)

    # 다운샘플링된 이미지 생성
    if "img_A_ds" in batch and "img_B_ds" in batch:
        img_A_ds = batch["img_A_ds"].to(device)
        img_B_ds = batch["img_B_ds"].to(device)
    else:
        # 만약 다운샘플링된 이미지가 없으면 직접 생성
        scale_factor = 0.5
        img_A_ds = F.interpolate(img_A_orig, scale_factor=scale_factor, mode='bilinear', align_corners=False)
        img_B_ds = F.interpolate(img_B_orig, scale_factor=scale_factor, mode='bilinear', align_corners=False)

    return img_A_orig, img_A_ds, img_B_orig, img_B_ds
