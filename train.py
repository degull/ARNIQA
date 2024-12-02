
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
    # 모델 출력값을 넘파이 배열로 변환
    proj_A = proj_A.detach().cpu().numpy()
    proj_B = proj_B.detach().cpu().numpy()

    # SRCC 계산
    srocc, _ = stats.spearmanr(proj_A.flatten(), proj_B.flatten())

    # PLCC 계산
    plcc, _ = stats.pearsonr(proj_A.flatten(), proj_B.flatten())

    return srocc, plcc

def train(args: DotMap,
          model: nn.Module,
          train_dataloader: DataLoader,
          optimizer: torch.optim.Optimizer,
          lr_scheduler: Optional[torch.optim.lr_scheduler.LRScheduler],
          scaler: torch.cuda.amp.GradScaler,
          device: torch.device) -> None:
    checkpoint_path = Path(args.checkpoint_base_path) / args.experiment_name / "pretrain"
    checkpoint_path.mkdir(parents=True, exist_ok=True)
    print("Saving checkpoints in folder: ", checkpoint_path)

    config_file_path = checkpoint_path / "config.yaml"
    dumpable_args = args.toDict()
    config_file_path = checkpoint_path / "config.yaml"
    with open(config_file_path, "w", encoding="utf-8") as f:
        yaml.dump(dumpable_args, f)

    start_epoch = 0 if not args.training.resume_training else args.training.start_epoch
    max_epochs = args.training.epochs
    best_srocc = 0 if not args.training.resume_training else args.best_srocc

    last_srocc = 0
    last_plcc = 0
    best_model_filename = ""

    for epoch in range(start_epoch, max_epochs):
        model.train()
        running_loss = 0.0
        srocc_list = []
        plcc_list = []

        progress_bar = tqdm(train_dataloader, desc=f"Epoch [{epoch + 1}/{max_epochs}]")

        for i, batch in enumerate(progress_bar):
            inputs_A_orig = batch["img_A_orig"].to(device=device, non_blocking=True)
            inputs_A_ds = batch["img_A_ds"].to(device=device, non_blocking=True)
            inputs_A = torch.cat((inputs_A_orig, inputs_A_ds), dim=0)

            inputs_B_orig = batch["img_B_orig"].to(device=device, non_blocking=True)
            inputs_B_ds = batch["img_B_ds"].to(device=device, non_blocking=True)
            inputs_B = torch.cat((inputs_B_orig, inputs_B_ds), dim=0)

            optimizer.zero_grad()

            with torch.amp.autocast(device_type='cuda'):
                proj_A, proj_B = model(inputs_A, inputs_B)
                loss = model.compute_loss(proj_A, proj_B)

            if torch.isnan(loss):
                raise ValueError("Loss is NaN")

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()

            # SRCC, PLCC 계산
            srocc, plcc = calculate_srcc_plcc(proj_A, proj_B)
            srocc_list.append(srocc)
            plcc_list.append(plcc)

            # Progress bar 업데이트
            progress_bar.set_postfix(loss=running_loss / (i + 1), srcc=srocc, plcc=plcc)

        # 에포크 단위로 평균 SRCC, PLCC 계산
        avg_srocc = np.mean(srocc_list)
        avg_plcc = np.mean(plcc_list)
        print(f"Epoch [{epoch + 1}/{max_epochs}] - Avg SRCC: {avg_srocc:.4f}, Avg PLCC: {avg_plcc:.4f}")

        # Validation
        if epoch % args.validation.frequency == 0:
            print("Starting validation...")
            last_srocc, last_plcc = validate(args, model, device)

        # Save checkpoints
        if epoch % args.checkpoint_frequency == 0:
            print("Saving checkpoint...")
            save_checkpoint(model, checkpoint_path, epoch, last_srocc)

    print("Finished training")



def validate(args: DotMap,
             model: nn.Module,
             device: torch.device) -> Tuple[float, float]:
    model.eval()

    srocc_all, plcc_all, _, _, _ = get_results(model=model,
                                               data_base_path=args.data_base_path,
                                               datasets=["kadid10k"],
                                               num_splits=args.validation.num_splits,
                                               phase="val",
                                               alpha=args.validation.alpha,
                                               grid_search=False,
                                               crop_size=args.test.crop_size,
                                               batch_size=args.test.batch_size,
                                               num_workers=args.test.num_workers,
                                               device=device)

    srocc_avg = np.mean([np.median(srocc["global"]) for srocc in srocc_all.values()])
    plcc_avg = np.mean([np.median(plcc["global"]) for plcc in plcc_all.values()])
    return srocc_avg, plcc_avg



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
                    eval_type: str = "scratch") -> Tuple[dict, dict, Ridge, float, dict]:
    srocc_dataset = {"global": []}
    plcc_dataset = {"global": []}
    best_worst_results = {}

    # DataLoader 설정
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    # features 및 scores 가져오기
    # features : 이미지에서 추출된 벡터 (입력값)
    # scores : 각 이미지의 실제 MOS
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
            raise ValueError(f"Train features or scores are empty at split {i}. Please check data preparation.")

        # Ridge Regressor 훈련
        # alpha : L2 규제 강도 -> 과적합 방지
        # fit : 주어진 학습 데이터 -> 가중치 w 최적화
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


def alpha_grid_search(dataset: Dataset,
                      features: np.ndarray,
                      scores: np.ndarray,
                      num_splits: int) -> float:
    """
    Perform a grid search over the validation splits to find the best alpha value for the regression based on the SROCC
    metric. The grid search is performed over the range [1-e3, 1e3, 100].

    Args:
        dataset (Dataset): dataset to use
        features (np.ndarray): features extracted with the model to test
        scores (np.ndarray): ground-truth MOS scores
        num_splits (int): number of splits to use

    Returns:
        alpha (float): best alpha value
    """

    grid_search_range = [1e-3, 1e3, 100]
    alphas = np.geomspace(*grid_search_range, endpoint=True)
    srocc_all = [[] for _ in range(len(alphas))]

    for i in range(num_splits):
        train_indices = dataset.get_split_indices(split=i, phase="train")
        val_indices = dataset.get_split_indices(split=i, phase="val")

        # for each index generate 5 indices (one for each crop)
        train_indices = np.repeat(train_indices * 5, 5) + np.tile(np.arange(5), len(train_indices))
        val_indices = np.repeat(val_indices * 5, 5) + np.tile(np.arange(5), len(val_indices))

        train_features = features[train_indices]
        train_scores = scores[train_indices]

        val_features = features[val_indices]
        val_scores = scores[val_indices]
        val_scores = val_scores[::5]  # Scores are repeated for each crop, so we only keep the first one

        for idx, alpha in enumerate(alphas):
            regressor = Ridge(alpha=alpha).fit(train_features, train_scores)
            preds = regressor.predict(val_features)
            preds = np.mean(np.reshape(preds, (-1, 5)), 1)  # Average the predictions of the 5 crops of the same image
            srocc_all[idx].append(stats.spearmanr(preds, val_scores)[0])

    srocc_all_median = [np.median(srocc) for srocc in srocc_all]
    srocc_all_median = np.array(srocc_all_median)
    best_alpha_idx = np.argmax(srocc_all_median)
    best_alpha = alphas[best_alpha_idx]

    return best_alpha


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Configuration file')
    parsed_args = parser.parse_args()

    # YAML 파일 로드 시 인코딩 문제 방지
    with open(parsed_args.config, encoding="utf-8") as f:
        config = yaml.safe_load(f)
    print("Loaded configuration:", config)

    # DotMap 객체로 변환
    args = DotMap(config)

    # 경로 변환
    args.data_base_path = Path(args.data_base_path)
    args.checkpoint_base_path = Path(args.checkpoint_base_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 데이터셋 및 데이터로더 설정
    train_dataset = KADID10KDataset(args.data_base_path / "KADID10K", phase="train")
    train_dataloader = DataLoader(train_dataset, batch_size=args.training.batch_size, shuffle=True,
                                  num_workers=args.training.num_workers)

    # 모델, 옵티마이저, 스케줄러, 스케일러 초기화
    model = SimCLR(encoder_params=args.model.encoder, temperature=args.model.temperature)
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.training.learning_rate)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.training.step_size,
                                                    gamma=args.training.gamma)
    scaler = torch.cuda.amp.GradScaler()

    # 학습 시작
    train(args, model, train_dataloader, optimizer, lr_scheduler, scaler, device)
