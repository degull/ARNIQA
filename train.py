
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import numpy as np
from dotmap import DotMap
import openpyxl
import pandas
from openpyxl.styles import Alignment
import pickle
from PIL import Image
from pathlib import Path
from typing import List, Tuple, Optional
from datetime import datetime
from einops import rearrange
from sklearn.linear_model import Ridge
from scipy import stats
from PIL import ImageFile
import argparse
from tqdm import tqdm
from data import LIVEDataset, CSIQDataset, TID2013Dataset, KADID10KDataset, FLIVEDataset, SPAQDataset
from utils.utils import PROJECT_ROOT, parse_command_line_args, merge_configs, parse_config
from models.simclr import SimCLR
from utils.visualization import visualize_tsne_umap_mos
import yaml
import matplotlib.pyplot as plt

ImageFile.LOAD_TRUNCATED_IMAGES = True

synthetic_datasets = ["live", "csiq", "tid2013", "kadid10k"]
authentic_datasets = ["flive", "spaq"]



def save_checkpoint(model: nn.Module, checkpoint_path: Path, epoch: int, srocc: float) -> None:
    """모델 체크포인트 저장 함수."""
    filename = f"epoch_{epoch}_srocc_{srocc:.3f}.pth"
    torch.save(model.state_dict(), checkpoint_path / filename)



def calculate_srcc_plcc(proj_A, proj_B):
    """SRCC와 PLCC 계산"""
    # 이미 NumPy 배열인지 확인 후 변환
    if isinstance(proj_A, torch.Tensor):
        proj_A = proj_A.detach().cpu().numpy()
    if isinstance(proj_B, torch.Tensor):
        proj_B = proj_B.detach().cpu().numpy()

    # 크기 확인 및 일치하지 않을 경우 예외 처리
    if proj_A.shape != proj_B.shape:
        raise ValueError(f"Shape mismatch: proj_A {proj_A.shape}, proj_B {proj_B.shape}")

    # SRCC 계산
    srocc, _ = stats.spearmanr(proj_A.flatten(), proj_B.flatten())
    # PLCC 계산
    plcc, _ = stats.pearsonr(proj_A.flatten(), proj_B.flatten())
    return srocc, plcc



def train(args: DotMap, model: nn.Module, train_dataloader: DataLoader, optimizer, lr_scheduler, scaler, device):
    checkpoint_path = Path(args.checkpoint_base_path) / args.experiment_name / "pretrain"
    checkpoint_path.mkdir(parents=True, exist_ok=True)
    print("Saving checkpoints in folder: ", checkpoint_path)

    best_srocc = 0
    max_epochs = args.training.epochs

    for epoch in range(max_epochs):
        model.train()
        running_loss = 0.0
        srocc_list, plcc_list = [], []

        progress_bar = tqdm(train_dataloader, desc=f"Epoch [{epoch + 1}/{max_epochs}]")

        for batch in progress_bar:
            inputs_A = batch["img_A_orig"].to(device)
            inputs_B = batch["img_B_orig"].to(device)

            optimizer.zero_grad()

            with torch.amp.autocast(device_type='cuda'):
                proj_A, proj_B = model(inputs_A, inputs_B)
                print(f"proj_A shape: {proj_A.shape}, proj_B shape: {proj_B.shape}")  # 디버깅용 출력
                loss = model.compute_loss(proj_A, proj_B)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()

            # SRCC, PLCC 계산
            srocc, plcc = calculate_srcc_plcc(proj_A, proj_B)
            srocc_list.append(srocc)
            plcc_list.append(plcc)

            # Progress bar 업데이트
            progress_bar.set_postfix(loss=running_loss / len(train_dataloader), srcc=srocc, plcc=plcc)

        # 평균 SRCC 및 PLCC 출력
        avg_srocc = np.mean(srocc_list)
        avg_plcc = np.mean(plcc_list)
        print(f"Epoch {epoch + 1}/{max_epochs}: Avg SRCC: {avg_srocc:.4f}, Avg PLCC: {avg_plcc:.4f}")

        # Validation
        if (epoch + 1) % args.validation.frequency == 0:
            validate(args, model, device)

        # Checkpoint 저장
        if (epoch + 1) % args.checkpoint_frequency == 0:
            save_checkpoint(model, checkpoint_path, epoch, avg_srocc)


def validate(args: DotMap, model: nn.Module, device: torch.device):
    """검증 루프"""
    model.eval()
    dataset = KADID10KDataset(args.data_base_path / "KADID10K", phase="val")
    dataloader = DataLoader(dataset, batch_size=args.test.batch_size, shuffle=False, num_workers=args.test.num_workers)

    all_preds, all_targets = [], []

    with torch.no_grad():
        for batch in dataloader:
            inputs_A = batch["img_A_orig"].to(device)
            inputs_B = batch["img_B_orig"].to(device)
            targets = batch["mos"].to(device)  # MOS 점수 (1D)

            with torch.amp.autocast(device_type='cuda'):
                proj_A, proj_B = model(inputs_A, inputs_B)

                # proj_A와 proj_B를 1D 스칼라로 변환 (평균 계산)
                preds_A = proj_A.view(inputs_A.size(0), -1, proj_A.size(-1)).mean(dim=(1, 2))  # 1D로 변환
                preds_B = proj_B.view(inputs_B.size(0), -1, proj_B.size(-1)).mean(dim=(1, 2))  # 1D로 변환

                # 두 특징의 평균값을 최종 예측으로 사용
                preds = (preds_A + preds_B) / 2

            # 결과 추가
            all_preds.append(preds.cpu().numpy())
            all_targets.append(targets.cpu().numpy())

    # Flatten predictions and targets
    all_preds = np.concatenate(all_preds, axis=0)  # 1D 배열로 합치기
    all_targets = np.concatenate(all_targets, axis=0)  # 1D 배열로 합치기

    # 크기 확인
    if all_preds.shape != all_targets.shape:
        raise ValueError(f"Mismatch between predictions and targets: {all_preds.shape} vs {all_targets.shape}")

    # SRCC와 PLCC 계산
    srocc, plcc = calculate_srcc_plcc(all_preds, all_targets)
    print(f"Validation - SRCC: {srocc:.4f}, PLCC: {plcc:.4f}")
    return srocc, plcc


def get_results(model: nn.Module,
                data_base_path: Path,
                datasets: List[str],
                num_splits: int,
                phase: str,
                alpha: float,
                grid_search: bool,
                crop_size: int,
                batch_size: int,
                num_workers: int,
                device: torch.device,
                eval_type: str = "scratch") -> Tuple[dict, dict, dict, dict, dict]:
    srocc_all = {}
    plcc_all = {}
    regressors = {}
    alphas = {}
    best_worst_results_all = {}

    assert phase in ["val", "test"], "Phase must be in ['val', 'test']"

    print(f"{datetime.now().strftime('%d/%m/%Y %H:%M:%S')} - Starting {phase} phase")
    for d in datasets:
        if d == "live":
            dataset = LIVEDataset(data_base_path / "LIVE", phase="all", crop_size=crop_size)
        elif d == "csiq":
            dataset = CSIQDataset(data_base_path / "CSIQ", phase="all", crop_size=crop_size)
        elif d == "tid2013":
            dataset = TID2013Dataset(data_base_path / "TID2013", phase="all", crop_size=crop_size)
        elif d == "kadid10k":
            dataset = KADID10KDataset(data_base_path / "KADID10K", phase="all", crop_size=crop_size)
        elif d == "flive":
            dataset = FLIVEDataset(data_base_path / "FLIVE", phase="all", crop_size=crop_size)
        elif d == "spaq":
            dataset = SPAQDataset(data_base_path / "SPAQ", phase="all", crop_size=crop_size)
        else:
            raise ValueError(f"Dataset {d} not supported")

        # 결과 계산
        srocc_dataset, plcc_dataset, regressor, alpha_value, best_worst_results = compute_metrics(model, dataset,
                                                                                                num_splits, phase,
                                                                                                alpha, grid_search,
                                                                                                batch_size, num_workers,
                                                                                                device, eval_type)
        srocc_all[d] = srocc_dataset
        plcc_all[d] = plcc_dataset
        regressors[d] = regressor
        alphas[d] = alpha_value
        best_worst_results_all[d] = best_worst_results
        print(f"{datetime.now().strftime('%d/%m/%Y %H:%M:%S')} - {d}:" f" SRCC: {np.median(srocc_dataset['global']):.3f} - PLCC: {np.median(plcc_dataset['global']):.3f}")

    return srocc_all, plcc_all, regressors, alphas, best_worst_results_all

""" def compute_metrics(model: nn.Module,
                    dataset: DataLoader,
                    num_splits: int,
                    phase: str,
                    alpha: float,
                    grid_search: bool,
                    batch_size: int,
                    num_workers: int,
                    device: torch.device,
                    eval_type: str = "scratch") -> Tuple[dict, dict, Ridge, float, dict]:
    srocc_dataset = {"global": []}
    plcc_dataset = {"global": []}
    best_worst_results = {}

    # DataLoader 설정
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    # features 및 scores 가져오기
    features, scores = get_features_scores(model, dataloader, device, eval_type)

    # 배치 형태 출력
    for i, batch in enumerate(dataloader):
        print(f"Batch {i} shapes:")
        print(f"img_A_orig shape: {batch['img_A_orig'].shape}")
        print(f"img_B_orig shape: {batch['img_B_orig'].shape}")
        print(f"mos shape: {batch['mos'].shape}")
        print("-" * 40)

    # Debugging: scores shape 확인
    print(f"Features shape: {features.shape}, Scores shape: {scores.shape}")

    # Grid search 또는 alpha 값을 사용하여 회귀 모델 학습
    if phase == "test" and grid_search:
        best_alpha = alpha_grid_search(dataset=dataset, features=features, scores=scores, num_splits=num_splits)
    else:
        best_alpha = alpha

    for i in range(num_splits):
        # Split 데이터셋
        train_indices = dataset.get_split_indices(split=i, phase="train")
        test_indices = dataset.get_split_indices(split=i, phase=phase)

        # Train features 및 scores 가져오기
        train_features = features[train_indices]
        train_scores = scores[train_indices]

        # Reshape to 2D if necessary
        if train_features.ndim == 1:
            train_features = train_features.reshape(-1, 1)
        if train_scores.ndim == 1:
            train_scores = train_scores.reshape(-1, 1)

        regressor = Ridge(alpha=best_alpha).fit(train_features, train_scores)

        # Test features 및 scores 가져오기
        test_features = features[test_indices]
        test_scores = scores[test_indices]

        # Reshape to 2D if necessary
        if test_features.ndim == 1:
            test_features = test_features.reshape(-1, 1)
        if test_scores.ndim == 1:
            test_scores = test_scores.reshape(-1, 1)

        preds = regressor.predict(test_features)

        # Reshape preds to 1D
        preds = preds.flatten()  # Make preds a 1D array

        # Debugging: preds와 test_scores의 shape 확인
        print(f"Preds shape: {preds.shape}, Test scores shape: {test_scores.shape}")

        # Check if preds needs to be reshaped to match test_scores
        if preds.shape[0] != test_scores.shape[0]:
            print(f"Mismatch in shapes: preds {preds.shape}, test_scores {test_scores.shape}")
            preds = preds[:test_scores.shape[0]]  # Adjust preds to match test_scores if necessary

        # Compute SROCC 및 PLCC
        srocc_dataset["global"].append(stats.spearmanr(preds, test_scores.flatten())[0])  # Flatten for correlation
        plcc_dataset["global"].append(stats.pearsonr(preds, test_scores.flatten())[0])  # Flatten for correlation

        # Compute best and worst results
        if i == 0:
            diff = np.abs(preds - test_scores.flatten())
            sorted_diff_indices = np.argsort(diff)
            best_indices = sorted_diff_indices[:16]
            worst_indices = sorted_diff_indices[-16:][::-1]
            best_worst_results["best"] = {"images": dataset.images[test_indices[best_indices]], "gts": test_scores[best_indices], "preds": preds[best_indices]}
            best_worst_results["worst"] = {"images": dataset.images[test_indices[worst_indices]], "gts": test_scores[worst_indices], "preds": preds[worst_indices]}

    return srocc_dataset, plcc_dataset, regressor, best_alpha, best_worst_results
 """


def compute_metrics(model: nn.Module,
                    dataset: DataLoader,
                    num_splits: int,
                    phase: str,
                    alpha: float,
                    grid_search: bool,
                    batch_size: int,
                    num_workers: int,
                    device: torch.device,
                    eval_type: str = "scratch") -> Tuple[dict, dict, Optional[Ridge], float, dict]:
    srocc_dataset = {"global": []}
    plcc_dataset = {"global": []}
    best_worst_results = {}
    regressor = None  # 초기화

    # DataLoader 설정
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    # features 및 scores 가져오기
    features, scores = get_features_scores(model, dataloader, device, eval_type)

    # 디버깅: features와 scores의 크기 확인
    print(f"Features shape: {features.shape}")
    print(f"Scores shape: {scores.shape}")

    if len(features) == 0 or len(scores) == 0:
        raise ValueError("Features or Scores are empty. Please check the dataset or feature extraction process.")

    # Grid search 또는 alpha 값 확인
    best_alpha = alpha
    if phase == "test" and grid_search:
        best_alpha = alpha_grid_search(dataset=dataset, features=features, scores=scores, num_splits=num_splits)

    for i in range(num_splits):
        # 데이터셋 인덱스 분할
        train_indices = dataset.get_split_indices(split=i, phase="train")
        test_indices = dataset.get_split_indices(split=i, phase=phase)

        # 인덱스 범위 필터링
        train_indices = [idx for idx in train_indices if idx < len(features)]
        test_indices = [idx for idx in test_indices if idx < len(features)]

        # 디버깅: 데이터셋 인덱스 상태 확인
        print(f"Split {i}: Train indices length: {len(train_indices)}, Test indices length: {len(test_indices)}")

        if len(train_indices) == 0 or len(test_indices) == 0:
            print(f"Split {i} has empty indices. Skipping this split.")
            continue

        # Train 데이터 준비
        train_features = features[train_indices]
        train_scores = scores[train_indices]

        # Test 데이터 준비
        test_features = features[test_indices]
        test_scores = scores[test_indices]

        if len(train_features) == 0 or len(train_scores) == 0:
            print(f"Split {i} has no valid train data. Skipping this split.")
            continue

        # Ridge Regressor 훈련
        regressor = Ridge(alpha=best_alpha).fit(train_features, train_scores)

        # Regressor 예측
        preds = regressor.predict(test_features)
        preds = preds.flatten()
        true_scores = test_scores.flatten()

        # SROCC 및 PLCC 계산
        srocc_value = stats.spearmanr(preds, true_scores)[0]
        plcc_value = stats.pearsonr(preds, true_scores)[0]
        print(f"Split {i}: SROCC: {srocc_value:.4f}, PLCC: {plcc_value:.4f}")

        srocc_dataset["global"].append(srocc_value)
        plcc_dataset["global"].append(plcc_value)

        # 그래프 생성
        plt.figure(figsize=(8, 8))
        plt.scatter(true_scores, preds, alpha=0.7, color='blue', label="Predictions")
        plt.plot([min(true_scores), max(true_scores)], [min(true_scores), max(true_scores)], '--', color='red', label="Ideal Line")
        plt.xlabel("True MOS")
        plt.ylabel("Predicted MOS")
        plt.title(f"True vs Predicted MOS (Split {i+1}) - KADID")
        plt.legend()
        plt.grid()
        plt.show()

        # 잔차 분석 그래프 생성
        residuals = true_scores - preds
        plt.figure(figsize=(8, 6))
        plt.scatter(true_scores, residuals, alpha=0.7, color='green')
        plt.axhline(0, color='red', linestyle='--', label="Zero Residual Line")
        plt.xlabel("True MOS")
        plt.ylabel("Residuals")
        plt.title(f"Residuals (Split {i+1}) - KADID")
        plt.legend()
        plt.grid()
        plt.show()

    # 모든 Split을 건너뛰었다면 regressor가 None일 수 있음
    if regressor is None:
        print("No valid splits processed. Please check the dataset or split configuration.")
        return srocc_dataset, plcc_dataset, None, best_alpha, best_worst_results

    return srocc_dataset, plcc_dataset, regressor, best_alpha, best_worst_results


def get_features_scores(model, dataloader, device, eval_type):
    scores = np.array([])  # 초기화
    mos = np.array([])  # 초기화

    model.eval()  # 모델을 평가 모드로 설정
    with torch.no_grad():  # 그래디언트 계산 비활성화
        for i, batch in enumerate(dataloader):
            print(f"Batch {i} keys: {batch.keys()}")

            # Check if the expected keys are present in the batch
            if not all(key in batch for key in ['img_A_orig', 'img_B_orig']):
                print(f"Missing keys in batch {i}: {[key for key in ['img_A_orig', 'img_B_orig'] if key not in batch]}")
                continue

            print(f"Batch {i} mos: {batch['mos']}")  # Debugging: 'mos'의 내용 확인

            # Convert 'mos' to numpy array if it is a list
            if isinstance(batch['mos'], list):
                mos_batch = np.array(batch['mos'])
            else:
                mos_batch = batch['mos'].cpu().numpy()  # Ensure it is on CPU and convert to numpy

            mos = np.concatenate((mos, mos_batch), axis=0)  # Concatenate the mos

            # 이미지 데이터 가져오기
            img_A_orig = batch["img_A_orig"].to(device)
            img_B_orig = batch["img_B_orig"].to(device)

            # 모델에 대한 피처 추출
            with torch.amp.autocast(device_type='cuda'):
                feature_A, feature_B = model(img_A_orig, img_B_orig)  # 모델에서 두 개의 피처를 얻습니다.

            # Feature와 MOS를 병합
            features = np.concatenate((feature_A.detach().cpu().numpy(), feature_B.detach().cpu().numpy()), axis=0)
            scores = mos  # MOS를 그대로 scores에 저장


    # 반환값 추가
    return features, scores

def get_split_indices(dataset_size, num_splits, split, phase):
    """
    데이터셋을 균등하게 나누기 위한 인덱스 생성 함수.
    """
    if dataset_size < num_splits:
        raise ValueError("데이터셋 크기가 분할 수보다 작습니다. 데이터를 확인하세요.")

    indices = np.arange(dataset_size)
    np.random.shuffle(indices)

    split_size = dataset_size // num_splits
    remainder = dataset_size % num_splits

    # 각 split의 시작과 끝 인덱스를 정의
    start = split * split_size + min(split, remainder)
    end = start + split_size + (1 if split < remainder else 0)

    val_indices = indices[start:end]
    train_indices = np.setdiff1d(indices, val_indices, assume_unique=True)

    if phase == "train":
        return train_indices
    elif phase == "val":
        return val_indices
    else:
        raise ValueError("Phase must be 'train' or 'val'")


def validate_batch(batch, required_keys):
    """
    배치가 유효한지 검증합니다.
    """
    for key in required_keys:
        if key not in batch or batch[key] is None or len(batch[key]) == 0:
            print(f"배치에 '{key}'가 없습니다.")
            return False
    return True

def alpha_grid_search(dataset, features, scores, num_splits):
    """
    alpha의 최적값을 찾기 위한 Grid Search
    """
    alphas = np.geomspace(1e-3, 1e3, 100)
    best_alpha = None
    best_srocc = -1

    for alpha in alphas:
        srocc_list = []
        for split in range(num_splits):
            train_indices = get_split_indices(len(features), num_splits, split, 'train')
            val_indices = get_split_indices(len(features), num_splits, split, 'val')

            # Validation 데이터가 없는 split은 건너뜁니다.
            if len(train_indices) == 0 or len(val_indices) == 0:
                continue

            train_features = features[train_indices]
            train_scores = scores[train_indices]
            val_features = features[val_indices]
            val_scores = scores[val_indices]

            regressor = Ridge(alpha=alpha).fit(train_features, train_scores)
            preds = regressor.predict(val_features)

            srocc, _ = stats.spearmanr(preds, val_scores)
            srocc_list.append(srocc)

        avg_srocc = np.mean(srocc_list)
        if avg_srocc > best_srocc:
            best_srocc = avg_srocc
            best_alpha = alpha

    print(f"Grid Search 완료 - Best Alpha: {best_alpha}, Best SROCC: {best_srocc}")
    return best_alpha


def train_ridge_regressor(features, scores, alpha):
    """
    Ridge Regressor 학습 및 테스트
    """
    regressor = Ridge(alpha=alpha)
    regressor.fit(features, scores)
    return regressor



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Configuration file path")
    args = parser.parse_args()

    # Config 파일 로드
    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    args = DotMap(config)
    args.data_base_path = Path(args.data_base_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 데이터 로더 준비
    train_dataset = KADID10KDataset(args.data_base_path / "KADID10K", phase="train")
    train_dataloader = DataLoader(train_dataset, batch_size=args.training.batch_size, shuffle=True, num_workers=args.training.num_workers)

    # 모델 초기화
    model = SimCLR(encoder_params=args.model.encoder, temperature=args.model.temperature).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.training.learning_rate)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.training.step_size, gamma=args.training.gamma)
    scaler = torch.cuda.amp.GradScaler()

    # 학습 시작
    train(args, model, train_dataloader, optimizer, lr_scheduler, scaler, device)