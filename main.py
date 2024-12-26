import argparse
import torch
from torch.utils.data import DataLoader
from pathlib import Path
import random
import os
import numpy as np

from train import train  # Training and Validation
from run_test import test  # Testing
from models.simclr import SimCLR  # SimCLR Model
from data import KADID10KDataset  # Dataset Definition
from utils.utils import PROJECT_ROOT, parse_config, parse_command_line_args, merge_configs  # Utility Functions


def main():
    # Parse Command-Line Arguments and Configuration
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default="e:/ARNIQA/ARNIQA/config.yaml", help='Path to the configuration file')
    args, unknown = parser.parse_known_args()

    # Load and Merge Configurations
    config = parse_config(args.config)
    args = parse_command_line_args(config)
    args = merge_configs(config, args)
    print(args)

    # Device Configuration
    if args.device != -1 and torch.cuda.is_available():
        device = torch.device(f"cuda:{args.device}")
    else:
        device = torch.device("cpu")

    # Set Random Seeds
    SEED = args.seed
    torch.manual_seed(SEED)
    random.seed(SEED)
    torch.use_deterministic_algorithms(True)
    np.random.seed(SEED)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    # Dataset Path Configuration
    args.data_base_path = Path(args.data_base_path)
    args.checkpoint_base_path = PROJECT_ROOT / "experiments"

    # Load Dataset and Split into Train, Validation, and Test Sets
    dataset = KADID10KDataset(root=args.data_base_path / "KADID10K", phase="all")
    total_size = len(dataset)
    train_size = int(0.7 * total_size)
    val_size = int(0.2 * total_size)
    test_size = total_size - train_size - val_size

    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size, test_size]
    )

    # Create DataLoaders
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.training.batch_size,
        num_workers=args.training.num_workers,
        shuffle=True,
        pin_memory=True,
        drop_last=True,
    )

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=args.training.batch_size,
        num_workers=args.training.num_workers,
        shuffle=False,
        pin_memory=True,
    )

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=args.test.batch_size,
        num_workers=args.test.num_workers,
        shuffle=False,
        pin_memory=True,
    )

    # Initialize SimCLR Model
    model = SimCLR(encoder_params=args.model.encoder, temperature=args.model.temperature)
    model = model.to(device)

    # Initialize Optimizer
    if args.training.optimizer.name == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=args.training.lr,
                                     weight_decay=args.training.optimizer.weight_decay,
                                     betas=args.training.optimizer.betas, eps=args.training.optimizer.eps)
    elif args.training.optimizer.name == "SGD":
        optimizer = torch.optim.SGD(model.parameters(), lr=args.training.lr,
                                    momentum=args.training.optimizer.momentum,
                                    weight_decay=args.training.optimizer.weight_decay)
    else:
        raise NotImplementedError(f"Optimizer {args.training.optimizer.name} not implemented")

    # Initialize Scheduler
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.training.step_size, gamma=args.training.gamma)

    # Initialize Scaler for Mixed Precision Training
    scaler = torch.cuda.amp.GradScaler()

    # Training and Validation
    train(args, model, train_dataloader, val_dataloader, test_dataloader, optimizer, scheduler, scaler, device)

    # Testing
    print("Starting Testing...")
    test(args, model, test_dataloader, device)


if __name__ == '__main__':
    main()
