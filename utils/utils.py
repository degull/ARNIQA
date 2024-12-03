import torch
from pathlib import Path
from scipy import stats
import argparse
from dotmap import DotMap
import yaml


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
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "srocc": srocc,
    }
    torch.save(checkpoint, path / f"epoch_{epoch}_srocc_{srocc:.4f}.pth")


def calculate_srcc_plcc(proj_A, proj_B):
    proj_A = proj_A.detach().cpu().numpy()
    proj_B = proj_B.detach().cpu().numpy()
    srocc, _ = stats.spearmanr(proj_A.flatten(), proj_B.flatten())
    plcc, _ = stats.pearsonr(proj_A.flatten(), proj_B.flatten())
    return srocc, plcc



def configure_degradation_model(batch, device):
    """
    입력 데이터를 GPU로 이동하고 모델과 동일한 장치에서 처리하도록 설정.
    """
    img_A_orig = batch["img_A_orig"].to(device)
    img_A_ds = batch["img_A_ds"].to(device)
    img_B_orig = batch["img_B_orig"].to(device)
    img_B_ds = batch["img_B_ds"].to(device)
    return img_A_orig, img_A_ds, img_B_orig, img_B_ds
