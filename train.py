# KADID
""" 
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import numpy as np
from dotmap import DotMap
from pathlib import Path
from scipy import stats
from tqdm import tqdm
from sklearn.linear_model import Ridge
from data import KADID10KDataset
from models.simclr import SimCLR
from utils.utils_distortions import apply_random_distortions, generate_hard_negatives
from torch.cuda.amp import custom_fwd
import matplotlib.pyplot as plt
from torch.amp import autocast
import yaml

def load_config(config_path):
    with open(config_path, "r", encoding="utf-8") as file:
        return DotMap(yaml.safe_load(file))

def save_checkpoint(model: nn.Module, checkpoint_path: Path, epoch: int, srocc: float) -> None:
    filename = f"epoch_{epoch}_srocc_{srocc:.3f}.pth"
    torch.save(model.state_dict(), checkpoint_path / filename)

def calculate_srcc_plcc(proj_A, proj_B):
    proj_A = proj_A.detach().cpu().numpy()
    proj_B = proj_B.detach().cpu().numpy()

    srocc, _ = stats.spearmanr(proj_A.flatten(), proj_B.flatten())
    plcc, _ = stats.pearsonr(proj_A.flatten(), proj_B.flatten())
    return srocc, plcc

def validate(args: DotMap, model: nn.Module, val_dataloader: DataLoader, device: torch.device):
    model.eval()
    srocc_list, plcc_list = [], []

    with torch.no_grad():
        for batch in val_dataloader:
            inputs_A = batch["img_A"].to(device)  # [batch_size, num_crops, 3, 224, 224]
            inputs_B = batch["img_B"].to(device)  # [batch_size, num_crops, 3, 224, 224]

            inputs_A = inputs_A.view(-1, *inputs_A.shape[2:])  # [batch_size * num_crops, 3, 224, 224]
            inputs_B = inputs_B.view(-1, *inputs_B.shape[2:])  # [batch_size * num_crops, 3, 224, 224]

            proj_A, proj_B = model(inputs_A, inputs_B)

            if isinstance(proj_A, torch.Tensor) and isinstance(proj_B, torch.Tensor):
                srocc, plcc = calculate_srcc_plcc(proj_A, proj_B)
                srocc_list.append(srocc)
                plcc_list.append(plcc)
            else:
                raise ValueError("Model did not return expected outputs during validation.")

    avg_srocc = np.mean(srocc_list) if srocc_list else 0
    avg_plcc = np.mean(plcc_list) if plcc_list else 0
    return avg_srocc, avg_plcc

def train(args: DotMap, model: nn.Module, train_dataloader: DataLoader, val_dataloader: DataLoader, optimizer: torch.optim.Optimizer, lr_scheduler: torch.optim.lr_scheduler.StepLR, scaler: torch.cuda.amp.GradScaler, device: torch.device) -> None:
    checkpoint_path = Path(args.checkpoint_base_path) / args.experiment_name
    checkpoint_path.mkdir(parents=True, exist_ok=True)

    best_srocc = 0
    for epoch in range(args.training.epochs):
        model.train()
        running_loss = 0.0
        progress_bar = tqdm(train_dataloader, desc=f"Epoch [{epoch + 1}/{args.training.epochs}]")

        for batch in progress_bar:
            inputs_A = batch["img_A"].to(device)
            inputs_B = batch["img_B"].to(device)

            inputs_A = inputs_A.view(-1, *inputs_A.shape[2:])
            inputs_B = inputs_B.view(-1, *inputs_B.shape[2:])

            optimizer.zero_grad()
            with autocast(device_type='cuda'):
                loss = model(inputs_A, inputs_B)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()
            progress_bar.set_postfix(loss=running_loss / len(progress_bar))

        lr_scheduler.step()
        avg_srocc_val, avg_plcc_val = validate(args, model, val_dataloader, device)
        print(f"Epoch [{epoch + 1}] Validation SRCC: {avg_srocc_val:.4f}, PLCC: {avg_plcc_val:.4f}")

        if avg_srocc_val > best_srocc:
            best_srocc = avg_srocc_val
            save_checkpoint(model, checkpoint_path, epoch, best_srocc)

def test(args: DotMap, model: nn.Module, test_dataloader: DataLoader, device: torch.device):
    model.eval()
    srocc_list, plcc_list = [], []

    with torch.no_grad():
        for batch in test_dataloader:
            inputs_A = batch["img_A"].to(device)
            inputs_B = batch["img_B"].to(device)

            inputs_A = inputs_A.view(-1, *inputs_A.shape[2:])
            inputs_B = inputs_B.view(-1, *inputs_B.shape[2:])

            proj_A, proj_B = model(inputs_A, inputs_B)

            srocc, plcc = calculate_srcc_plcc(proj_A, proj_B)
            srocc_list.append(srocc)
            plcc_list.append(plcc)

    avg_srocc = np.mean(srocc_list) if srocc_list else 0
    avg_plcc = np.mean(plcc_list) if plcc_list else 0
    return avg_srocc, avg_plcc

if __name__ == "__main__":
    config_path = "E:/ARNIQA/ARNIQA/config.yaml"
    args = load_config(config_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = KADID10KDataset(args.data_base_path + "/KADID10K/kadid10k.csv")
    train_size = int(0.7 * len(dataset))
    val_size = int(0.1 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.training.batch_size,
        shuffle=True,
        num_workers=args.training.num_workers,
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=args.training.batch_size,
        shuffle=False,
        num_workers=args.training.num_workers,
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=args.training.batch_size,
        shuffle=False,
        num_workers=args.training.num_workers,
    )

    model = SimCLR(
        encoder_params=DotMap({"embedding_dim": args.model.encoder.embedding_dim, "pretrained": args.model.encoder.pretrained}),
        temperature=args.model.temperature
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.training.learning_rate)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.training.step_size, gamma=0.5)
    scaler = torch.cuda.amp.GradScaler()

    train(args, model, train_dataloader, val_dataloader, optimizer, lr_scheduler, scaler, device)

    print("Testing the model on the test dataset...")
    avg_srocc_test, avg_plcc_test = test(args, model, test_dataloader, device)
    print(f"Test Results: SRCC = {avg_srocc_test:.4f}, PLCC = {avg_plcc_test:.4f}")

 """

# TID
""" import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import numpy as np
from dotmap import DotMap
from pathlib import Path
from scipy import stats
from tqdm import tqdm
from sklearn.linear_model import Ridge
from data import TID2013Dataset
from models.simclr import SimCLR
from utils.utils_distortions import apply_random_distortions, generate_hard_negatives
from torch.cuda.amp import custom_fwd
import matplotlib.pyplot as plt
from torch.amp import autocast
import yaml

def load_config(config_path):
    with open(config_path, "r", encoding="utf-8") as file:
        return DotMap(yaml.safe_load(file))

def save_checkpoint(model: nn.Module, checkpoint_path: Path, epoch: int, srocc: float) -> None:
    filename = f"epoch_{epoch}_srocc_{srocc:.3f}.pth"
    torch.save(model.state_dict(), checkpoint_path / filename)

def calculate_srcc_plcc(proj_A, proj_B):
    proj_A = proj_A.detach().cpu().numpy()
    proj_B = proj_B.detach().cpu().numpy()

    srocc, _ = stats.spearmanr(proj_A.flatten(), proj_B.flatten())
    plcc, _ = stats.pearsonr(proj_A.flatten(), proj_B.flatten())
    return srocc, plcc

def validate(args: DotMap, model: nn.Module, val_dataloader: DataLoader, device: torch.device):
    model.eval()
    srocc_list, plcc_list = [], []

    with torch.no_grad():
        for batch in val_dataloader:
            inputs_A = batch["img_A"].to(device)  # [batch_size, num_crops, 3, 224, 224]
            inputs_B = batch["img_B"].to(device)  # [batch_size, num_crops, 3, 224, 224]

            inputs_A = inputs_A.view(-1, *inputs_A.shape[2:])  # [batch_size * num_crops, 3, 224, 224]
            inputs_B = inputs_B.view(-1, *inputs_B.shape[2:])  # [batch_size * num_crops, 3, 224, 224]

            proj_A, proj_B = model(inputs_A, inputs_B)

            if isinstance(proj_A, torch.Tensor) and isinstance(proj_B, torch.Tensor):
                srocc, plcc = calculate_srcc_plcc(proj_A, proj_B)
                srocc_list.append(srocc)
                plcc_list.append(plcc)
            else:
                raise ValueError("Model did not return expected outputs during validation.")

    avg_srocc = np.mean(srocc_list) if srocc_list else 0
    avg_plcc = np.mean(plcc_list) if plcc_list else 0
    return avg_srocc, avg_plcc

def train(args: DotMap, model: nn.Module, train_dataloader: DataLoader, val_dataloader: DataLoader, optimizer: torch.optim.Optimizer, lr_scheduler: torch.optim.lr_scheduler.StepLR, scaler: torch.cuda.amp.GradScaler, device: torch.device) -> None:
    checkpoint_path = Path(args.checkpoint_base_path) / args.experiment_name
    checkpoint_path.mkdir(parents=True, exist_ok=True)

    best_srocc = 0
    for epoch in range(args.training.epochs):
        model.train()
        running_loss = 0.0
        progress_bar = tqdm(train_dataloader, desc=f"Epoch [{epoch + 1}/{args.training.epochs}]")

        for batch in progress_bar:
            inputs_A = batch["img_A"].to(device)
            inputs_B = batch["img_B"].to(device)

            inputs_A = inputs_A.view(-1, *inputs_A.shape[2:])
            inputs_B = inputs_B.view(-1, *inputs_B.shape[2:])

            optimizer.zero_grad()
            with autocast(device_type='cuda'):
                loss = model(inputs_A, inputs_B)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()
            progress_bar.set_postfix(loss=running_loss / len(progress_bar))

        lr_scheduler.step()
        avg_srocc_val, avg_plcc_val = validate(args, model, val_dataloader, device)
        print(f"Epoch [{epoch + 1}] Validation SRCC: {avg_srocc_val:.4f}, PLCC: {avg_plcc_val:.4f}")

        if avg_srocc_val > best_srocc:
            best_srocc = avg_srocc_val
            save_checkpoint(model, checkpoint_path, epoch, best_srocc)

def test(args: DotMap, model: nn.Module, test_dataloader: DataLoader, device: torch.device):
    model.eval()
    srocc_list, plcc_list = [], []

    with torch.no_grad():
        for batch in test_dataloader:
            inputs_A = batch["img_A"].to(device)
            inputs_B = batch["img_B"].to(device)

            inputs_A = inputs_A.view(-1, *inputs_A.shape[2:])
            inputs_B = inputs_B.view(-1, *inputs_B.shape[2:])

            proj_A, proj_B = model(inputs_A, inputs_B)

            srocc, plcc = calculate_srcc_plcc(proj_A, proj_B)
            srocc_list.append(srocc)
            plcc_list.append(plcc)

    avg_srocc = np.mean(srocc_list) if srocc_list else 0
    avg_plcc = np.mean(plcc_list) if plcc_list else 0
    return avg_srocc, avg_plcc

if __name__ == "__main__":
    config_path = "E:/ARNIQA/ARNIQA/config.yaml"
    args = load_config(config_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = TID2013Dataset(args.data_base_path + "/TID2013/mos.csv")
    train_size = int(0.7 * len(dataset))
    val_size = int(0.1 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.training.batch_size,
        shuffle=True,
        num_workers=args.training.num_workers,
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=args.training.batch_size,
        shuffle=False,
        num_workers=args.training.num_workers,
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=args.training.batch_size,
        shuffle=False,
        num_workers=args.training.num_workers,
    )

    model = SimCLR(
        encoder_params=DotMap({"embedding_dim": args.model.encoder.embedding_dim, "pretrained": args.model.encoder.pretrained}),
        temperature=args.model.temperature
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.training.learning_rate)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.training.step_size, gamma=0.5)
    scaler = torch.cuda.amp.GradScaler()

    train(args, model, train_dataloader, val_dataloader, optimizer, lr_scheduler, scaler, device)

    print("Testing the model on the test dataset...")
    avg_srocc_test, avg_plcc_test = test(args, model, test_dataloader, device)
    print(f"Test Results: SRCC = {avg_srocc_test:.4f}, PLCC = {avg_plcc_test:.4f}")


 """
# SPAQ ver1
""" 
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import numpy as np
from dotmap import DotMap
from pathlib import Path
from scipy import stats
from tqdm import tqdm
from sklearn.linear_model import Ridge
from data import SPAQDataset
from models.simclr import SimCLR
from torch.amp import autocast
import yaml

def load_config(config_path):
    with open(config_path, "r", encoding="utf-8") as file:
        return DotMap(yaml.safe_load(file))

def save_checkpoint(model: nn.Module, checkpoint_path: Path, epoch: int, srocc: float) -> None:
    filename = f"epoch_{epoch}_srocc_{srocc:.3f}.pth"
    torch.save(model.state_dict(), checkpoint_path / filename)

def calculate_srcc_plcc(proj_A, proj_B):
    proj_A = proj_A.detach().cpu().numpy()
    proj_B = proj_B.detach().cpu().numpy()

    srocc, _ = stats.spearmanr(proj_A.flatten(), proj_B.flatten())
    plcc, _ = stats.pearsonr(proj_A.flatten(), proj_B.flatten())
    return srocc, plcc

def validate(args: DotMap, model: nn.Module, val_dataloader: DataLoader, device: torch.device):
    model.eval()
    srocc_list, plcc_list = [], []

    with torch.no_grad():
        for batch in val_dataloader:
            inputs_A = batch["img_A"].to(device)
            inputs_B = batch["img_B"].to(device)

            print(f"inputs_A shape: {inputs_A.shape}, inputs_B shape: {inputs_B.shape}")  # Debugging

            proj_A, proj_B = model(inputs_A, inputs_B)

            if isinstance(proj_A, torch.Tensor) and isinstance(proj_B, torch.Tensor):
                srocc, plcc = calculate_srcc_plcc(proj_A, proj_B)
                srocc_list.append(srocc)
                plcc_list.append(plcc)
            else:
                raise ValueError("Model did not return expected outputs during validation.")

    avg_srocc = np.mean(srocc_list) if srocc_list else 0
    avg_plcc = np.mean(plcc_list) if plcc_list else 0
    return avg_srocc, avg_plcc


def train(args: DotMap, model: nn.Module, train_dataloader: DataLoader, val_dataloader: DataLoader, optimizer: torch.optim.Optimizer, lr_scheduler: torch.optim.lr_scheduler.StepLR, scaler: torch.cuda.amp.GradScaler, device: torch.device) -> None:
    checkpoint_path = Path(args.checkpoint_base_path) / args.experiment_name
    checkpoint_path.mkdir(parents=True, exist_ok=True)

    best_srocc = 0
    for epoch in range(args.training.epochs):
        model.train()
        running_loss = 0.0
        progress_bar = tqdm(train_dataloader, desc=f"Epoch [{epoch + 1}/{args.training.epochs}]")

        for batch in progress_bar:
            inputs_A = batch["img_A"].to(device)
            inputs_B = batch["img_B"].to(device)

            print(f"inputs_A shape: {inputs_A.shape}, inputs_B shape: {inputs_B.shape}")  # Debugging

            optimizer.zero_grad()
            with autocast(device_type='cuda'):
                loss = model(inputs_A, inputs_B)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()
            progress_bar.set_postfix(loss=running_loss / len(progress_bar))

        lr_scheduler.step()
        avg_srocc_val, avg_plcc_val = validate(args, model, val_dataloader, device)
        print(f"Epoch [{epoch + 1}] Validation SRCC: {avg_srocc_val:.4f}, PLCC: {avg_plcc_val:.4f}")

        if avg_srocc_val > best_srocc:
            best_srocc = avg_srocc_val
            save_checkpoint(model, checkpoint_path, epoch, best_srocc)


def test(args: DotMap, model: nn.Module, test_dataloader: DataLoader, device: torch.device):
    model.eval()
    srocc_list, plcc_list = [], []

    with torch.no_grad():
        for batch in test_dataloader:
            inputs_A = batch["img_A"].to(device)
            inputs_B = batch["img_B"].to(device)

            proj_A, proj_B = model(inputs_A, inputs_B)

            srocc, plcc = calculate_srcc_plcc(proj_A, proj_B)
            srocc_list.append(srocc)
            plcc_list.append(plcc)

    avg_srocc = np.mean(srocc_list) if srocc_list else 0
    avg_plcc = np.mean(plcc_list) if plcc_list else 0
    return avg_srocc, avg_plcc


if __name__ == "__main__":
    config_path = "E:/ARNIQA/ARNIQA/config.yaml"
    args = load_config(config_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = SPAQDataset(
        csv_path=args.data_base_path + "/SPAQ/Annotations/MOS and Image attribute scores.csv",
        image_dir=args.data_base_path + "/SPAQ/TestImage"
    )

    train_size = int(0.7 * len(dataset))
    val_size = int(0.1 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.training.batch_size,
        shuffle=True,
        num_workers=args.training.num_workers,
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=args.training.batch_size,
        shuffle=False,
        num_workers=args.training.num_workers,
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=args.training.batch_size,
        shuffle=False,
        num_workers=args.training.num_workers,
    )

    model = SimCLR(
        encoder_params=DotMap({"embedding_dim": args.model.encoder.embedding_dim, "pretrained": args.model.encoder.pretrained}),
        temperature=args.model.temperature
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.training.learning_rate)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.training.step_size, gamma=0.5)
    scaler = torch.cuda.amp.GradScaler()

    train(args, model, train_dataloader, val_dataloader, optimizer, lr_scheduler, scaler, device)

    print("Testing the model on the test dataset...")
    avg_srocc_test, avg_plcc_test = test(args, model, test_dataloader, device)
    print(f"Test Results: SRCC = {avg_srocc_test:.4f}, PLCC = {avg_plcc_test:.4f}")
 """

# SPAQ ver2

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import numpy as np
from dotmap import DotMap
from pathlib import Path
from scipy import stats
from tqdm import tqdm
from sklearn.linear_model import Ridge
from data import SPAQDataset
from models.simclr import SimCLR
from torch.amp import autocast
import yaml

def load_config(config_path):
    with open(config_path, "r", encoding="utf-8") as file:
        return DotMap(yaml.safe_load(file))

def save_checkpoint(model: nn.Module, checkpoint_path: Path, epoch: int, srocc: float) -> None:
    filename = f"epoch_{epoch}_srocc_{srocc:.3f}.pth"
    torch.save(model.state_dict(), checkpoint_path / filename)

def calculate_srcc_plcc(predictions, ground_truths):
    predictions = predictions.detach().cpu().numpy()
    ground_truths = ground_truths.detach().cpu().numpy()

    predictions = predictions.flatten()
    ground_truths = ground_truths.flatten()

    srocc, _ = stats.spearmanr(predictions, ground_truths)
    plcc, _ = stats.pearsonr(predictions, ground_truths)
    return srocc, plcc

def train_ridge_regression(embeddings, mos_scores):
    ridge = Ridge(alpha=1.0)
    ridge.fit(embeddings, mos_scores)
    return ridge

def validate(args: DotMap, model: nn.Module, val_dataloader: DataLoader, device: torch.device):
    if not isinstance(model, nn.Module):
        raise TypeError("Expected 'model' to be of type 'nn.Module', but got type '{}'".format(type(model)))
    model.eval()
    srocc_list, plcc_list = [], []

    with torch.no_grad():
        for batch in val_dataloader:
            inputs_A = batch["img_A"].to(device)
            inputs_B = batch["img_B"].to(device)

            proj_A, proj_B = model(inputs_A, inputs_B)

            srocc, plcc = calculate_srcc_plcc(proj_A, proj_B)
            srocc_list.append(srocc)
            plcc_list.append(plcc)

    avg_srocc = np.mean(srocc_list) if srocc_list else 0
    avg_plcc = np.mean(plcc_list) if plcc_list else 0
    return avg_srocc, avg_plcc

def train(args: DotMap, model: nn.Module, train_dataloader: DataLoader, val_dataloader: DataLoader, optimizer: torch.optim.Optimizer, lr_scheduler: torch.optim.lr_scheduler.StepLR, scaler: torch.cuda.amp.GradScaler, device: torch.device) -> None:
    checkpoint_path = Path(args.checkpoint_base_path) / args.experiment_name
    checkpoint_path.mkdir(parents=True, exist_ok=True)

    best_srocc = 0
    for epoch in range(args.training.epochs):
        model.train()
        running_loss = 0.0
        progress_bar = tqdm(train_dataloader, desc=f"Epoch [{epoch + 1}/{args.training.epochs}]")

        for batch in progress_bar:
            inputs_A = batch["img_A"].to(device)
            inputs_B = batch["img_B"].to(device)

            optimizer.zero_grad()
            with autocast(device_type='cuda'):
                loss = model(inputs_A, inputs_B)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()
            progress_bar.set_postfix(loss=running_loss / len(progress_bar))

        lr_scheduler.step()
        avg_srocc_val, avg_plcc_val = validate(args, model, val_dataloader, device)
        print(f"Epoch [{epoch + 1}] Validation SRCC: {avg_srocc_val:.4f}, PLCC: {avg_plcc_val:.4f}")

        if avg_srocc_val > best_srocc:
            best_srocc = avg_srocc_val
            save_checkpoint(model, checkpoint_path, epoch, best_srocc)

def test(args: DotMap, model: nn.Module, test_dataloader: DataLoader, device: torch.device):
    model.eval()
    srocc_list, plcc_list = [], []

    with torch.no_grad():
        for batch in test_dataloader:
            inputs_A = batch["img_A"].to(device)
            inputs_B = batch["img_B"].to(device)

            proj_A, proj_B = model(inputs_A, inputs_B)

            srocc, plcc = calculate_srcc_plcc(proj_A, proj_B)
            srocc_list.append(srocc)
            plcc_list.append(plcc)

    avg_srocc = np.mean(srocc_list) if srocc_list else 0
    avg_plcc = np.mean(plcc_list) if plcc_list else 0
    return avg_srocc, avg_plcc

if __name__ == "__main__":
    config_path = "E:/ARNIQA/ARNIQA/config.yaml"
    args = load_config(config_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = SPAQDataset(
        csv_path=args.data_base_path + "/SPAQ/Annotations/MOS and Image attribute scores.csv",
        image_dir=args.data_base_path + "/SPAQ/TestImage"
    )

    train_size = int(0.7 * len(dataset))
    val_size = int(0.1 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.training.batch_size,
        shuffle=True,
        num_workers=args.training.num_workers,
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=args.training.batch_size,
        shuffle=False,
        num_workers=args.training.num_workers,
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=args.training.batch_size,
        shuffle=False,
        num_workers=args.training.num_workers,
    )

    model = SimCLR(
        encoder_params=DotMap({"embedding_dim": args.model.encoder.embedding_dim, "pretrained": args.model.encoder.pretrained}),
        temperature=args.model.temperature
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.training.learning_rate)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.training.step_size, gamma=0.5)
    scaler = torch.cuda.amp.GradScaler()

    train(args, model, train_dataloader, val_dataloader, optimizer, lr_scheduler, scaler, device)

    print("Testing the model on the test dataset...")
    avg_srocc_test, avg_plcc_test = test(args, model, test_dataloader, device)
    print(f"테스트 결과: SRCC = {avg_srocc_test:.4f}, PLCC = {avg_plcc_test:.4f}")
    avg_srocc_train, avg_plcc_train = validate(args, model, train_dataloader, device)
    print(f"훈련 결과: SRCC = {avg_srocc_train:.4f}, PLCC = {avg_plcc_train:.4f}")
    avg_srocc_val, avg_plcc_val = validate(args, model, val_dataloader, device)
    print(f"검증 결과: SRCC = {avg_srocc_val:.4f}, PLCC = {avg_plcc_val:.4f}")



# CSIQ
""" 
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import numpy as np
from dotmap import DotMap
from pathlib import Path
from scipy import stats
from tqdm import tqdm
from sklearn.linear_model import Ridge
from data import CSIQDataset
from models.simclr import SimCLR
from utils.utils_distortions import apply_random_distortions, generate_hard_negatives
from torch.cuda.amp import custom_fwd
import matplotlib.pyplot as plt
from torch.amp import autocast
import yaml

def load_config(config_path):
    with open(config_path, "r", encoding="utf-8") as file:
        return DotMap(yaml.safe_load(file))

def save_checkpoint(model: nn.Module, checkpoint_path: Path, epoch: int, srocc: float) -> None:
    filename = f"epoch_{epoch}_srocc_{srocc:.3f}.pth"
    torch.save(model.state_dict(), checkpoint_path / filename)

def calculate_srcc_plcc(proj_A, proj_B):
    proj_A = proj_A.detach().cpu().numpy()
    proj_B = proj_B.detach().cpu().numpy()

    srocc, _ = stats.spearmanr(proj_A.flatten(), proj_B.flatten())
    plcc, _ = stats.pearsonr(proj_A.flatten(), proj_B.flatten())
    return srocc, plcc

def validate(args: DotMap, model: nn.Module, val_dataloader: DataLoader, device: torch.device):
    model.eval()
    srocc_list, plcc_list = [], []

    with torch.no_grad():
        for batch in val_dataloader:
            inputs_A = batch["img_A"].to(device)  # [batch_size, num_crops, 3, 224, 224]
            inputs_B = batch["img_B"].to(device)  # [batch_size, num_crops, 3, 224, 224]

            inputs_A = inputs_A.view(-1, *inputs_A.shape[2:])  # [batch_size * num_crops, 3, 224, 224]
            inputs_B = inputs_B.view(-1, *inputs_B.shape[2:])  # [batch_size * num_crops, 3, 224, 224]

            proj_A, proj_B = model(inputs_A, inputs_B)

            if isinstance(proj_A, torch.Tensor) and isinstance(proj_B, torch.Tensor):
                srocc, plcc = calculate_srcc_plcc(proj_A, proj_B)
                srocc_list.append(srocc)
                plcc_list.append(plcc)
            else:
                raise ValueError("Model did not return expected outputs during validation.")

    avg_srocc = np.mean(srocc_list) if srocc_list else 0
    avg_plcc = np.mean(plcc_list) if plcc_list else 0
    return avg_srocc, avg_plcc

def train(args: DotMap, model: nn.Module, train_dataloader: DataLoader, val_dataloader: DataLoader, optimizer: torch.optim.Optimizer, lr_scheduler: torch.optim.lr_scheduler.StepLR, scaler: torch.cuda.amp.GradScaler, device: torch.device) -> None:
    checkpoint_path = Path(args.checkpoint_base_path) / args.experiment_name
    checkpoint_path.mkdir(parents=True, exist_ok=True)

    best_srocc = 0
    for epoch in range(args.training.epochs):
        model.train()
        running_loss = 0.0
        progress_bar = tqdm(train_dataloader, desc=f"Epoch [{epoch + 1}/{args.training.epochs}]")

        for batch in progress_bar:
            inputs_A = batch["img_A"].to(device)  # [batch_size, 3, 224, 224]
            inputs_B = batch["img_B"].to(device)  # [batch_size, 3, 224, 224]

            # 차원 변환: 모델이 [batch_size * num_crops, 3, 224, 224]를 기대하는 경우
            inputs_A = inputs_A.view(-1, *inputs_A.shape[1:])  # [batch_size * num_crops, 3, 224, 224]
            inputs_B = inputs_B.view(-1, *inputs_B.shape[1:])  # [batch_size * num_crops, 3, 224, 224]

            optimizer.zero_grad()
            with autocast(device_type='cuda'):
                loss = model(inputs_A, inputs_B)  # 모델에 전달

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()


            running_loss += loss.item()
            progress_bar.set_postfix(loss=running_loss / len(progress_bar))

        lr_scheduler.step()
        avg_srocc_val, avg_plcc_val = validate(args, model, val_dataloader, device)
        print(f"Epoch [{epoch + 1}] Validation SRCC: {avg_srocc_val:.4f}, PLCC: {avg_plcc_val:.4f}")

        if avg_srocc_val > best_srocc:
            best_srocc = avg_srocc_val
            save_checkpoint(model, checkpoint_path, epoch, best_srocc)

def test(args: DotMap, model: nn.Module, test_dataloader: DataLoader, device: torch.device):
    model.eval()
    srocc_list, plcc_list = [], []

    with torch.no_grad():
        for batch in test_dataloader:
            inputs_A = batch["img_A"].to(device)
            inputs_B = batch["img_B"].to(device)

            inputs_A = inputs_A.view(-1, *inputs_A.shape[2:])
            inputs_B = inputs_B.view(-1, *inputs_B.shape[2:])

            proj_A, proj_B = model(inputs_A, inputs_B)

            srocc, plcc = calculate_srcc_plcc(proj_A, proj_B)
            srocc_list.append(srocc)
            plcc_list.append(plcc)

    avg_srocc = np.mean(srocc_list) if srocc_list else 0
    avg_plcc = np.mean(plcc_list) if plcc_list else 0
    return avg_srocc, avg_plcc

if __name__ == "__main__":
    config_path = "E:/ARNIQA/ARNIQA/config.yaml"
    args = load_config(config_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = CSIQDataset(args.data_base_path + "/CSIQ/CSIQ.txt")
    train_size = int(0.7 * len(dataset))
    val_size = int(0.1 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.training.batch_size,
        shuffle=True,
        num_workers=args.training.num_workers,
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=args.training.batch_size,
        shuffle=False,
        num_workers=args.training.num_workers,
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=args.training.batch_size,
        shuffle=False,
        num_workers=args.training.num_workers,
    )

    model = SimCLR(
        encoder_params=DotMap({"embedding_dim": args.model.encoder.embedding_dim, "pretrained": args.model.encoder.pretrained}),
        temperature=args.model.temperature
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.training.learning_rate)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.training.step_size, gamma=0.5)
    scaler = torch.cuda.amp.GradScaler()

    train(args, model, train_dataloader, val_dataloader, optimizer, lr_scheduler, scaler, device)

    print("Testing the model on the test dataset...")
    avg_srocc_test, avg_plcc_test = test(args, model, test_dataloader, device)
    print(f"Test Results: SRCC = {avg_srocc_test:.4f}, PLCC = {avg_plcc_test:.4f}")
 """