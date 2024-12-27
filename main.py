import argparse
import torch
from torch.utils.data import DataLoader
from pathlib import Path
import wandb
import random
import os
import numpy as np
from dotmap import DotMap
from train import train
from test import test
from models.simclr import SimCLR
from data import KADID10KDataset
from utils.utils import PROJECT_ROOT, parse_config, parse_command_line_args, merge_configs

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Path to the configuration file')
    cli_args = parser.parse_args()

    # Load and merge configurations
    config = parse_config(cli_args.config)
    args = merge_configs(config, parse_command_line_args(config))

    # Convert to DotMap for consistent access
    if not isinstance(args, DotMap):
        args = DotMap(args)

    print(args)  # Debugging purposes

    # Set the device
    if args.device != -1 and torch.cuda.is_available():
        device = torch.device(f"cuda:{args.device}")
    else:
        device = torch.device("cpu")

    print(f"Using device: {device}")

    # Set random seed
    SEED = args.seed
    torch.manual_seed(SEED)
    random.seed(SEED)
    np.random.seed(SEED)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    args.data_base_path = Path(args.data_base_path)
    args.checkpoint_base_path = Path(args.checkpoint_base_path)

    # Initialize the training dataset and dataloader
    train_dataset = KADID10KDataset(
        root=args['data_base_path'] / "KADID10K",
        crop_size=args['training']['data']['patch_size'],  # 수정된 부분
        max_distortions=args['training']['data']['max_distortions'],
        num_levels=args['training']['data']['num_levels'],
        pristine_prob=args['training']['data']['pristine_prob']
    )
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.training.batch_size,
        num_workers=args.training.num_workers,
        shuffle=True,
        pin_memory=True,
        drop_last=True
    )

    # Initialize the model
    model = SimCLR(encoder_params=args.model.encoder, temperature=args.model.temperature)
    model = model.to(device)

    # Initialize the optimizer
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=args.training.lr,
        momentum=args.training.optimizer.momentum,
        weight_decay=args.training.optimizer.weight_decay
    )

    # Initialize the learning rate scheduler
    lr_scheduler = None
    if args.training.lr_scheduler.name == "CosineAnnealingWarmRestarts":
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=args.training.lr_scheduler.T_0,
            T_mult=args.training.lr_scheduler.T_mult,
            eta_min=args.training.lr_scheduler.eta_min,
        )

    # Automatic mixed precision scaler
    scaler = torch.cuda.amp.GradScaler()

    # Resume training if necessary
    run_id = None
    if args.training.resume_training:
        try:
            checkpoint_path = args.checkpoint_base_path / args.experiment_name / "pretrain"
            checkpoint_path = [el for el in checkpoint_path.glob("*.pth") if "last" in el.name][0]
            checkpoint = torch.load(checkpoint_path, map_location="cpu")
            model.load_state_dict(checkpoint["model_state_dict"], strict=True)
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            scaler.load_state_dict(checkpoint["scaler_state_dict"])
            args.training.start_epoch = checkpoint["epoch"] + 1
            run_id = checkpoint["config"].logging.wandb.run_id
            args.best_srocc = checkpoint["config"].best_srocc
            print(f"--- Resuming training from epoch {checkpoint['epoch'] + 1} ---")
        except Exception as e:
            print(f"Failed to resume training: {e}. Starting from scratch.")

    # Initialize logger
    logger = None
    if args.logging.use_wandb:
        logger = wandb.init(
            project=args.logging.wandb.project,
            entity=args.logging.wandb.entity,
            name=args.experiment_name if not args.training.resume_training else None,
            config=args.toDict(),
            mode="online" if args.logging.wandb.online else "offline",
            resume=args.training.resume_training,
            id=run_id
        )
        args.logging.wandb.run_id = logger.id

    # Training
    train(args, model, train_dataloader, optimizer, lr_scheduler, scaler, logger, device)
    print("--- Training finished ---")

    # Testing
    checkpoint_path = args.checkpoint_base_path / args.experiment_name / "pretrain"
    checkpoint_path = [ckpt_path for ckpt_path in checkpoint_path.glob("*.pth") if "best" in ckpt_path.name][0]
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(checkpoint, strict=True)
    model.to(device)
    print("Starting testing with best checkpoint...")

    test(args, model, logger, device)
    print("--- Testing finished ---")

if __name__ == "__main__":
    main()
