""" import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LRScheduler
import numpy as np
from pathlib import Path
from tqdm import tqdm
import os
import yaml
import wandb
from wandb.wandb_run import Run
from PIL import ImageFile
from dotmap import DotMap
from typing import Optional, Tuple

from data import KADID10KDataset
from test import get_results, synthetic_datasets, authentic_datasets
from utils.visualization import visualize_tsne_umap_mos

ImageFile.LOAD_TRUNCATED_IMAGES = True


def train(args: DotMap,
          model: torch.nn.Module,
          train_dataloader: DataLoader,
          optimizer: torch.optim.Optimizer,
          lr_scheduler: Optional[LRScheduler],
          scaler: torch.cuda.amp.GradScaler,
          logger: Optional[Run],
          device: torch.device) -> None:

    checkpoint_path = Path(args['checkpoint_base_path']) / args['experiment_name'] / "pretrain"
    checkpoint_path.mkdir(parents=True, exist_ok=True)
    print("Saving checkpoints in folder: ", checkpoint_path)

    with open(Path(args['checkpoint_base_path']) / args['experiment_name'] / "config.yaml", "w") as f:
        dumpable_args = args.copy()
        for key, value in dumpable_args.items():
            if isinstance(value, Path):
                dumpable_args[key] = str(value)
        yaml.dump(dumpable_args, f)

    if args.training.resume_training:
        start_epoch = args.training.start_epoch
        max_epochs = args.training.epochs
        best_srocc = args.best_srocc
    else:
        start_epoch = 0
        max_epochs = args.training.epochs
        best_srocc = 0

    last_srocc = 0
    last_plcc = 0
    last_model_filename = ""
    best_model_filename = ""

    for epoch in range(start_epoch, max_epochs):
        model.train()
        running_loss = 0.0
        progress_bar = tqdm(train_dataloader, desc=f"Epoch [{epoch + 1}/{max_epochs}]")

        for i, batch in enumerate(progress_bar):
            num_logging_steps = i * args.training.batch_size + len(train_dataloader) * args.training.batch_size * epoch

            inputs_A_orig = batch["img_A_orig"].to(device=device, non_blocking=True)
            inputs_A_ds = batch["img_A_ds"].to(device=device, non_blocking=True)
            if inputs_A_orig.shape[2:] != inputs_A_ds.shape[2:]:
                inputs_A_ds = torch.nn.functional.interpolate(inputs_A_ds, size=inputs_A_orig.shape[2:], mode="bilinear", align_corners=False)
            inputs_A = torch.cat((inputs_A_orig, inputs_A_ds), dim=0)

            inputs_B_orig = batch["img_B_orig"].to(device=device, non_blocking=True)
            inputs_B_ds = batch["img_B_ds"].to(device=device, non_blocking=True)
            if inputs_B_orig.shape[2:] != inputs_B_ds.shape[2:]:
                inputs_B_ds = torch.nn.functional.interpolate(inputs_B_ds, size=inputs_B_orig.shape[2:], mode="bilinear", align_corners=False)
            inputs_B = torch.cat((inputs_B_orig, inputs_B_ds), dim=0)

            optimizer.zero_grad()

            with torch.cuda.amp.autocast():
                loss = model(inputs_A, inputs_B)

            if torch.isnan(loss):
                raise ValueError("Loss is NaN")

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            if lr_scheduler and lr_scheduler.__class__.__name__ == "CosineAnnealingWarmRestarts":
                lr_scheduler.step(epoch + i / len(train_dataloader))

            running_loss += loss.item()
            progress_bar.set_postfix(loss=running_loss / (i + 1), SROCC=last_srocc, PLCC=last_plcc)

            if logger:
                logger.log({"loss": loss.item(), "lr": optimizer.param_groups[0]["lr"]}, step=num_logging_steps)

        if lr_scheduler and lr_scheduler.__class__.__name__ != "CosineAnnealingWarmRestarts":
            lr_scheduler.step()

        if epoch % args.validation.frequency == 0:
            print("Starting validation...")
            last_srocc, last_plcc = validate(args, model, logger, num_logging_steps, device)
            print(f"Validation Results - SROCC: {last_srocc:.4f}, PLCC: {last_plcc:.4f}")  # 추가된 출력 코드

            if args.validation.visualize and logger:
                kadid10k_val = KADID10KDataset(args.data_base_path / "KADID10K", phase="val")
                val_dataloader = DataLoader(kadid10k_val, batch_size=args.test.batch_size, shuffle=False,
                                            num_workers=args.test.num_workers)
                figures = visualize_tsne_umap_mos(model, val_dataloader,
                                                  tsne_args=args.validation.visualization.tsne,
                                                  umap_args=args.validation.visualization.umap,
                                                  device=device)
                logger.log(figures, step=num_logging_steps)

        print("Saving checkpoint")

        if last_srocc > best_srocc:
            best_srocc = last_srocc
            best_plcc = last_plcc
            args.best_srocc = best_srocc
            args.best_plcc = best_plcc
            if best_model_filename:
                os.remove(checkpoint_path / best_model_filename)
            best_model_filename = f"best_epoch_{epoch}_srocc_{best_srocc:.3f}_plcc_{best_plcc:.3f}.pth"
            torch.save(model.state_dict(), checkpoint_path / best_model_filename)

        if last_model_filename:
            os.remove(checkpoint_path / last_model_filename)
        last_model_filename = f"last_epoch_{epoch}_srocc_{last_srocc:.3f}_plcc_{last_plcc:.3f}.pth"
        torch.save({"model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scaler_state_dict": scaler.state_dict(),
                    "epoch": epoch,
                    "config": args,
                    }, checkpoint_path / last_model_filename)

    print('Finished training')


def validate(args: DotMap,
             model: torch.nn.Module,
             logger: Optional[Run],
             num_logging_steps: int,
             device: torch.device) -> Tuple[float, float]:
    model.eval()

    srocc_all, plcc_all, _, _, _ = get_results(model=model, data_base_path=args.data_base_path,
                                               datasets=args.validation.datasets, num_splits=args.validation.num_splits,
                                               phase="val", alpha=args.validation.alpha, grid_search=False,
                                               crop_size=args.test.crop_size, batch_size=args.test.batch_size,
                                               num_workers=args.test.num_workers, device=device)

    # Compute the median for each list in srocc_all and plcc_all
    srocc_all_median = {key: np.median(value["global"]) for key, value in srocc_all.items()}
    plcc_all_median = {key: np.median(value["global"]) for key, value in plcc_all.items()}

    # Compute the synthetic and authentic averages
    srocc_synthetic_avg = np.mean(
        [srocc_all_median[key] for key in srocc_all_median.keys() if key in synthetic_datasets])
    plcc_synthetic_avg = np.mean([plcc_all_median[key] for key in plcc_all_median.keys() if key in synthetic_datasets])
    srocc_authentic_avg = np.mean(
        [srocc_all_median[key] for key in srocc_all_median.keys() if key in authentic_datasets])
    plcc_authentic_avg = np.mean([plcc_all_median[key] for key in plcc_all_median.keys() if key in authentic_datasets])

    # Compute the global average
    srocc_avg = np.mean(list(srocc_all_median.values()))
    plcc_avg = np.mean(list(plcc_all_median.values()))

    if logger:
        logger.log({f"val_srocc_{key}": srocc_all_median[key] for key in srocc_all_median.keys()}, step=num_logging_steps)
        logger.log({f"val_plcc_{key}": plcc_all_median[key] for key in plcc_all_median.keys()}, step=num_logging_steps)
        logger.log({"val_srocc_synthetic_avg": srocc_synthetic_avg, "val_plcc_synthetic_avg": plcc_synthetic_avg,
                    "val_srocc_authentic_avg": srocc_authentic_avg, "val_plcc_authentic_avg": plcc_authentic_avg,
                    "val_srocc_avg": srocc_avg, "val_plcc_avg": plcc_avg}, step=num_logging_steps)
        

    last_srocc, last_plcc = validate(args, model, logger, num_logging_steps, device)
    print(f"Validation Results - SROCC: {last_srocc}, PLCC: {last_plcc}")

    return srocc_avg, plcc_avg
 """

import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LRScheduler
import numpy as np
from pathlib import Path
from tqdm import tqdm
import os
import yaml
import wandb
from wandb.wandb_run import Run
from PIL import ImageFile
from dotmap import DotMap
from typing import Optional, Tuple

from data import KADID10KDataset
from test import get_results, synthetic_datasets, authentic_datasets
from utils.visualization import visualize_tsne_umap_mos

ImageFile.LOAD_TRUNCATED_IMAGES = True


def train(args: DotMap,
          model: torch.nn.Module,
          train_dataloader: DataLoader,
          optimizer: torch.optim.Optimizer,
          lr_scheduler: Optional[LRScheduler],
          scaler: torch.cuda.amp.GradScaler,
          logger: Optional[Run],
          device: torch.device) -> None:

    checkpoint_path = Path(args['checkpoint_base_path']) / args['experiment_name'] / "pretrain"
    checkpoint_path.mkdir(parents=True, exist_ok=True)
    print("Saving checkpoints in folder: ", checkpoint_path)

    start_epoch = args.training.start_epoch if args.training.resume_training else 0
    max_epochs = args.training.epochs
    best_srocc = args.best_srocc if args.training.resume_training else 0

    for epoch in range(start_epoch, max_epochs):
        model.train()
        running_loss = 0.0
        progress_bar = tqdm(train_dataloader, desc=f"Epoch [{epoch + 1}/{max_epochs}]")

        for i, batch in enumerate(progress_bar):
            optimizer.zero_grad()

            inputs_A = batch["img_A_orig"].to(device).view(-1, *batch["img_A_orig"].shape[2:])
            inputs_B = batch["img_B_orig"].to(device).view(-1, *batch["img_B_orig"].shape[2:])

            with torch.amp.autocast(device_type="cuda"):
                loss = model(inputs_A, inputs_B)

            if torch.isnan(loss):
                raise ValueError("Loss is NaN")

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            if lr_scheduler and lr_scheduler.__class__.__name__ == "CosineAnnealingWarmRestarts":
                lr_scheduler.step(epoch + i / len(train_dataloader))

            running_loss += loss.item()
            progress_bar.set_postfix(loss=running_loss / (i + 1))

            if logger:
                logger.log({"loss": loss.item()})

        if lr_scheduler and lr_scheduler.__class__.__name__ != "CosineAnnealingWarmRestarts":
            lr_scheduler.step()

        if epoch % args.validation.frequency == 0:
            print("Validating...")
            last_srocc, last_plcc = validate(args, model, logger, device)
            print(f"Validation Results - SROCC: {last_srocc:.4f}, PLCC: {last_plcc:.4f}")

    print('Finished training')

    print("Testing...")
    test(args, model, logger, device)

def test(args: DotMap,
         model: torch.nn.Module,
         logger: Optional[Run],
         device: torch.device) -> None:
    model.eval()


    srocc_all, plcc_all, _, _, _ = get_results(
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
        device=device
    )

    srocc_all_median = {key: np.median(value["global"]) for key, value in srocc_all.items()}
    plcc_all_median = {key: np.median(value["global"]) for key, value in plcc_all.items()}

    print("\nTest Dataset Results:")
    print(f"{'Dataset':<15} {'SROCC':<15} {'PLCC':<15}")
    for dataset, srocc_value in srocc_all_median.items():
        plcc_value = plcc_all_median.get(dataset, 0)
        print(f"{dataset:<15} {srocc_value:<15.4f} {plcc_value:<15.4f}")

    synthetic_avg_srocc = np.mean([srocc_all_median.get(d, 0) for d in synthetic_datasets])
    synthetic_avg_plcc = np.mean([plcc_all_median.get(d, 0) for d in synthetic_datasets])
    authentic_avg_srocc = np.mean([srocc_all_median.get(d, 0) for d in authentic_datasets])
    authentic_avg_plcc = np.mean([plcc_all_median.get(d, 0) for d in authentic_datasets])

    print("\nSynthetic Averages:")
    print(f"SROCC: {synthetic_avg_srocc:.4f}, PLCC: {synthetic_avg_plcc:.4f}")
    print("\nAuthentic Averages:")
    print(f"SROCC: {authentic_avg_srocc:.4f}, PLCC: {authentic_avg_plcc:.4f}")


def validate(args: DotMap,
             model: torch.nn.Module,
             logger: Optional[Run],
             device: torch.device) -> Tuple[float, float]:
    model.eval()

    srocc_all, plcc_all, _, _, _ = get_results(
        model=model,
        data_base_path=args.data_base_path,
        datasets=args.validation.datasets,
        num_splits=args.validation.num_splits,
        phase="val",
        alpha=args.validation.alpha,
        grid_search=False,
        crop_size=args.test.crop_size,
        batch_size=args.test.batch_size,
        num_workers=args.test.num_workers,
        device=device
    )

    srocc_avg = np.mean([np.median(value["global"]) for value in srocc_all.values()])
    plcc_avg = np.mean([np.median(value["global"]) for value in plcc_all.values()])

    return srocc_avg, plcc_avg


def train_validate_test(args: DotMap,
                        model: torch.nn.Module,
                        device: torch.device) -> None:
    datasets = {
        "train": ["KADID10K_train"],
        "val": ["KADID10K_val"],
        "test": ["KADID10K_test"]
    }

    for phase, dataset_name in datasets.items():
        evaluate_and_log(args, model, dataset_name[0], None, 0, device)


def evaluate_and_log(args: DotMap,
                     model: torch.nn.Module,
                     phase: str,
                     logger: Optional[Run],
                     num_logging_steps: int,
                     device: torch.device) -> Tuple[float, float]:
    model.eval()

    srocc_all, plcc_all, _, _, _ = get_results(
        model=model,
        data_base_path=args.data_base_path,
        datasets=[phase],
        num_splits=args.validation.num_splits,
        phase=phase,
        alpha=args.validation.alpha,
        grid_search=False,
        crop_size=args.test.crop_size,
        batch_size=args.test.batch_size,
        num_workers=args.test.num_workers,
        device=device
    )

    srocc_avg = np.median(srocc_all["global"])
    plcc_avg = np.median(plcc_all["global"])

    print(f"{phase.upper()} Results - SROCC: {srocc_avg:.4f}, PLCC: {plcc_avg:.4f}")

    if logger:
        logger.log({f"{phase}_srocc_avg": srocc_avg, f"{phase}_plcc_avg": plcc_avg}, step=num_logging_steps)

    return srocc_avg, plcc_avg


if __name__ == "__main__":
    import argparse
    from utils.utils import parse_config, parse_command_line_args, merge_configs
    from models.simclr import SimCLR

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Path to the configuration file')
    cli_args = parser.parse_args()

    config = parse_config(cli_args.config)
    args = merge_configs(config, parse_command_line_args(config))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #train_dataset = KADID10KDataset(
    #    root=args['data_base_path'] / "KADID10K",
    #    crop_size=args['training']['data']['patch_size'],
    #    max_distortions=args['training']['data']['max_distortions'],
    #    num_levels=args['training']['data']['num_levels'],
    #    pristine_prob=args['training']['data']['pristine_prob']
    #)

    train_dataset = KADID10KDataset(
    root=args.data_base_path / "KADID10K",
    crop_size=args.training.data.patch_size,  # 논문과 동일한 크기 확인
    max_distortions=args.training.data.max_distortions,
    num_levels=args.training.data.num_levels,
    pristine_prob=args.training.data.pristine_prob  # 프리스틴 이미지 포함 비율
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.training.batch_size,
        num_workers=args.training.num_workers,
        shuffle=True,
        pin_memory=True,
        drop_last=True
    )

    model = SimCLR(encoder_params=args.model.encoder, temperature=args.model.temperature).to(device)

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=args.training.lr,
        momentum=args.training.optimizer.momentum,
        weight_decay=args.training.optimizer.weight_decay
    )

    lr_scheduler = None
    if args.training.lr_scheduler.name == "CosineAnnealingWarmRestarts":
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=args.training.lr_scheduler.T_0,
            T_mult=args.training.lr_scheduler.T_mult,
            eta_min=args.training.lr_scheduler.eta_min,
        )

    scaler = torch.cuda.amp.GradScaler()
    train(args, model, train_dataloader, optimizer, lr_scheduler, scaler, None, device)
