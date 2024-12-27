import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import numpy as np
from dotmap import DotMap
import wandb
from wandb.wandb_run import Run
import openpyxl
from openpyxl.styles import Alignment
import pickle
from PIL import Image
from pathlib import Path
from typing import List, Tuple
from datetime import datetime
from einops import rearrange
from sklearn.linear_model import Ridge
from scipy import stats
import argparse

from data import LIVEDataset, CSIQDataset, TID2013Dataset, KADID10KDataset, FLIVEDataset, SPAQDataset
from utils.utils import PROJECT_ROOT, parse_command_line_args, merge_configs, parse_config
from models.simclr import SimCLR


synthetic_datasets = ["live", "csiq", "tid2013", "kadid10k"]
authentic_datasets = ["flive", "spaq"]


def test(args: DotMap, model: nn.Module, logger: Run, device: torch.device) -> None:
    checkpoint_base_path = PROJECT_ROOT / "experiments"
    checkpoint_path = checkpoint_base_path / args.experiment_name
    regressor_path = checkpoint_path / "regressors"
    regressor_path.mkdir(parents=True, exist_ok=True)

    eval_type = args.get("eval_type", "scratch")
    model.eval()

    srocc_all, plcc_all, regressors, alphas, best_worst_results_all = get_results(
        model=model,
        data_base_path=args.data_base_path,
        datasets=args.test.datasets,
        num_splits=args.test.num_splits,
        phase="test",
        alpha=args.test.alpha,
        grid_search=args.test.grid_search,
        crop_size=args.test.crop_size,
        batch_size=args.test.batch_size,
        num_workers=args.test.num_workers,
        device=device,
        eval_type=eval_type,
    )

    # Ensure "global" key exists
    for key in synthetic_datasets + authentic_datasets:
        if key not in srocc_all:
            srocc_all[key] = {"global": [0.0]}
            plcc_all[key] = {"global": [0.0]}

    # Log results
    srocc_all_median = {key: np.median(value["global"]) for key, value in srocc_all.items()}
    plcc_all_median = {key: np.median(value["global"]) for key, value in plcc_all.items()}

    srocc_synthetic_avg = np.mean([srocc_all_median.get(key, 0) for key in synthetic_datasets])
    plcc_synthetic_avg = np.mean([plcc_all_median.get(key, 0) for key in synthetic_datasets])
    srocc_authentic_avg = np.mean([srocc_all_median.get(key, 0) for key in authentic_datasets])
    plcc_authentic_avg = np.mean([plcc_all_median.get(key, 0) for key in authentic_datasets])

    print(f"{'Dataset':<15} {'Alpha':<15} {'SROCC':<15} {'PLCC':<15}")
    for dataset in srocc_all_median.keys():
        print(f"{dataset:<15} {alphas.get(dataset, 0):<15.4f} {srocc_all_median[dataset]:<15.4f} {plcc_all_median[dataset]:<15.4f}")
    print(f"{'Synthetic avg':<15} {srocc_synthetic_avg:<15.4f} {plcc_synthetic_avg:<15.4f}")
    print(f"{'Authentic avg':<15} {srocc_authentic_avg:<15.4f} {plcc_authentic_avg:<15.4f}")

    # Save results
    workbook = openpyxl.Workbook()
    median_sheet = workbook.create_sheet('Median', 0)
    median_sheet.append(['Dataset', 'Alpha', 'SROCC', 'PLCC'])
    for dataset in srocc_all_median.keys():
        median_sheet.append([dataset, alphas.get(dataset, 0), srocc_all_median[dataset], plcc_all_median[dataset]])
    workbook.save(checkpoint_path / 'results.xlsx')


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

    print(f"{datetime.now().strftime('%d/%m/%Y %H:%M:%S')} - Starting {phase} phase")
    for d in datasets:
        if d in ["train", "val", "test"]:
            dataset = KADID10KDataset(data_base_path / "KADID10K", phase=d, crop_size=crop_size)
        elif d == "live":
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

        srocc_dataset, plcc_dataset, regressor, alpha, best_worst_results = compute_metrics(
            model, dataset, num_splits, phase, alpha, grid_search, batch_size, num_workers, device, eval_type
        )

        if "global" not in srocc_dataset:
            srocc_dataset["global"] = [0.0] * num_splits
        if "global" not in plcc_dataset:
            plcc_dataset["global"] = [0.0] * num_splits

        srocc_all[d] = srocc_dataset
        plcc_all[d] = plcc_dataset
        regressors[d] = regressor
        alphas[d] = alpha
        best_worst_results_all[d] = best_worst_results

    return srocc_all, plcc_all, regressors, alphas, best_worst_results_all



""" def compute_metrics(model, dataloader, num_splits, phase=None, alpha=None, grid_search=False, batch_size=None, num_workers=None, device=None, eval_type="scratch"):
    features, scores = get_features_scores(model, dataloader, device)

    srocc_all, plcc_all = {"global": []}, {"global": []}
    for i in range(num_splits):
        train_indices = np.arange(0, len(features) // 2)
        test_indices = np.arange(len(features) // 2, len(features))

        train_features = features[train_indices]
        train_scores = scores[train_indices]
        test_features = features[test_indices]
        test_scores = scores[test_indices]

        regressor = Ridge().fit(train_features, train_scores)
        preds = regressor.predict(test_features)

        srocc = stats.spearmanr(preds, test_scores)[0]
        plcc = stats.pearsonr(preds, test_scores)[0]

        srocc_all["global"].append(srocc if not np.isnan(srocc) else 0.0)
        plcc_all["global"].append(plcc if not np.isnan(plcc) else 0.0)

    return srocc_all, plcc_all, None, None, None
 """



def compute_metrics(model, dataloader, num_splits, phase=None, alpha=None, grid_search=False,
                    batch_size=None, num_workers=None, device=None, eval_type="scratch"):
    features, scores = get_features_scores(model, dataloader, device)

    srocc_all, plcc_all = {"global": []}, {"global": []}
    for i in range(num_splits):
        train_indices = np.arange(0, len(features) // 2)
        test_indices = np.arange(len(features) // 2, len(features))

        train_features = features[train_indices]
        train_scores = scores[train_indices]
        test_features = features[test_indices]
        test_scores = scores[test_indices]

        if len(train_features) == 0 or len(test_features) == 0:
            srocc_all["global"].append(0.0)
            plcc_all["global"].append(0.0)
            continue

        regressor = Ridge().fit(train_features, train_scores)
        preds = regressor.predict(test_features)

        srocc = stats.spearmanr(preds, test_scores)[0]
        plcc = stats.pearsonr(preds, test_scores)[0]

        srocc_all["global"].append(srocc if not np.isnan(srocc) else 0.0)
        plcc_all["global"].append(plcc if not np.isnan(plcc) else 0.0)

    return srocc_all, plcc_all, None, None, None


def get_features_scores(model, dataloader, device):
    features, scores = [], []

    for batch in dataloader:
        img = batch["img"].to(device)
        mos = batch["mos"]

        # Ensure img is 4D (batch_size, channels, height, width)
        if img.ndim == 3:  # If a single image is passed (C, H, W)
            img = img.unsqueeze(0)
        elif img.ndim != 4:
            raise ValueError(f"Expected 4D input for img, but got {img.shape}")

        # Convert mos to numpy array
        if isinstance(mos, torch.Tensor):
            mos = mos.cpu().numpy()
        elif isinstance(mos, float) or isinstance(mos, int):
            mos = np.array([mos])  # Ensure mos is a numpy array
        else:
            raise ValueError(f"Unsupported type for mos: {type(mos)}")

        # Ensure mos is at least 1D
        if mos.ndim == 0:
            mos = np.expand_dims(mos, axis=0)

        with torch.no_grad():
            feats, _ = model(img)
            features.append(feats.cpu().numpy())
            scores.append(mos)

    # Concatenate all features and scores
    features = np.vstack(features)
    scores = np.concatenate(scores)

    return features, scores



def alpha_grid_search(dataset: Dataset,
                      features: np.ndarray,
                      scores: np.ndarray,
                      num_splits: int) -> float:

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

# Train, Validation, Test 데이터셋에 대한 결과 출력
def evaluate_and_log(args, model, dataset, num_splits, phase, alpha, grid_search, crop_size, batch_size, num_workers, device, eval_type):
    # Compute metrics
    srocc_dataset, plcc_dataset, _, _, _ = compute_metrics(
        model=model,
        dataset=dataset,
        num_splits=num_splits,
        phase=phase,
        alpha=alpha,
        grid_search=grid_search,
        batch_size=batch_size,
        num_workers=num_workers,
        device=device,
        eval_type=eval_type
    )

    # Ensure "global" key exists
    if "global" not in srocc_dataset:
        raise KeyError(f"'global' key is missing in srocc_dataset. Available keys: {srocc_dataset.keys()}")

    # Compute median SRCC and PLCC
    srocc_median = np.median(srocc_dataset["global"])
    plcc_median = np.median(plcc_dataset["global"])
    print(f"Phase: {phase.upper()}, SRCC: {srocc_median:.4f}, PLCC: {plcc_median:.4f}")

    return srocc_median, plcc_median


# Train, Validation, Test 모두 평가 및 로그
def evaluate_all(args, model, train_dataset, val_dataset, test_dataset, num_splits, alpha, grid_search, crop_size, batch_size, num_workers, device, eval_type):
    print("Evaluating TRAIN dataset...")
    evaluate_and_log(args, model, train_dataset, num_splits, "train", alpha, grid_search, crop_size, batch_size, num_workers, device, eval_type)

    print("Evaluating VALIDATION dataset...")
    evaluate_and_log(args, model, val_dataset, num_splits, "val", alpha, grid_search, crop_size, batch_size, num_workers, device, eval_type)

    print("Evaluating TEST dataset...")
    evaluate_and_log(args, model, test_dataset, num_splits, "test", alpha, grid_search, crop_size, batch_size, num_workers, device, eval_type)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Configuration file')
    parser.add_argument("--eval_type", type=str, default="scratch", choices=["scratch", "arniqa"],
                        help="Whether to test a model trained from scratch or the one pretrained by the authors of the"
                             "paper. Must be in ['scratch', 'arniqa']")
    args, _ = parser.parse_known_args()
    eval_type = args.eval_type
    config = parse_config(args.config)
    args = parse_command_line_args(config)
    args = merge_configs(config, args)
    args.eval_type = eval_type
    args.data_base_path = Path(args.data_base_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.eval_type == "scratch":
        model = SimCLR(encoder_params=args.model.encoder, temperature=args.model.temperature)
        checkpoint_base_path = PROJECT_ROOT / "experiments"
        assert (checkpoint_base_path / args.experiment_name).exists(), \
            f"Experiment {(checkpoint_base_path / args.experiment_name)} does not exist"
        checkpoint_path = checkpoint_base_path / args.experiment_name / "pretrain"
        checkpoint_path = [ckpt_path for ckpt_path in checkpoint_path.glob("*.pth") if "best" in ckpt_path.name][0]
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        model.load_state_dict(checkpoint, strict=True)
    elif args.eval_type == "arniqa":
        model = torch.hub.load(repo_or_dir="miccunifi/ARNIQA", source="github", model="ARNIQA")
    else:
        raise ValueError(f"Eval type {args.eval_type} not supported")
    model.to(device)
    model.eval()

    if args.logging.use_wandb:
        logger = wandb.init(project=args.logging.wandb.project,
                            entity=args.logging.wandb.entity,
                            name=args.experiment_name,
                            config=args,
                            mode="online" if args.logging.wandb.online else "offline")
    else:
        logger = None

    test(args, model, logger, device)
