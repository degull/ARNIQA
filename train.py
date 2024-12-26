
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
import argparse
from tqdm import tqdm
from data import LIVEDataset, CSIQDataset, TID2013Dataset, KADID10KDataset, FLIVEDataset, SPAQDataset
from utils.utils import PROJECT_ROOT, parse_command_line_args, merge_configs, parse_config
from models.simclr import SimCLR
# 필요한 함수들을 가져옵니다.
from utils.utils import (
    save_checkpoint,
    calculate_srcc_plcc,
    configure_degradation_model,
    parse_args
)
from models.simclr import SimCLR
from data import KADID10KDataset

import torch
from torch.utils.data import DataLoader
from utils.utils import calculate_srcc_plcc
from models.simclr import SimCLR




def train(args, model, train_dataloader, val_dataloader, test_dataloader, optimizer, scheduler, scaler, device):
    for epoch in range(args.training.epochs):
        model.train()
        print(f"Epoch {epoch + 1}/{args.training.epochs} 시작")
        total_loss = 0

        for batch_idx, batch in enumerate(train_dataloader):
            img_A, img_B = batch["img_A_orig"].to(device), batch["img_B_orig"].to(device)
            mos = batch["mos"].to(device)

            optimizer.zero_grad()

            with torch.cuda.amp.autocast():
                proj_A, proj_B = model(img_A, img_B)
                loss = model.compute_loss(proj_A, proj_B, mos)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()

        print(f"Train Results - Epoch {epoch + 1}: Loss = {total_loss / len(train_dataloader):.4f}")

        # Validation
        validate(args, model, val_dataloader, device)

    # Test
    test(args, model, test_dataloader, device)



def validate(args, model, val_dataloader, device):
    model.eval()
    srcc_total, plcc_total = 0, 0
    with torch.no_grad():
        for batch_idx, batch in enumerate(val_dataloader):
            img_A, img_B = batch["img_A_orig"].to(device), batch["img_B_orig"].to(device)
            mos = batch["mos"].to(device)

            if img_A.shape[1] != 3:
                img_A = img_A[:, :3, :, :]
            if img_B.shape[1] != 3:
                img_B = img_B[:, :3, :, :]

            proj_A, proj_B = model(img_A, img_B)
            srcc, plcc = calculate_srcc_plcc(proj_A, proj_B)
            srcc_total += srcc
            plcc_total += plcc

    avg_srcc = srcc_total / len(val_dataloader)
    avg_plcc = plcc_total / len(val_dataloader)
    print(f"Validation Results - Average SRCC: {avg_srcc:.4f}, Average PLCC: {avg_plcc:.4f}")


def test(args, model, test_dataloader, device):
    model.eval()
    srcc_total, plcc_total = 0, 0
    with torch.no_grad():
        for batch in test_dataloader:
            img_A, img_B = batch["img_A_orig"].to(device), batch["img_B_orig"].to(device)
            mos = batch["mos"].to(device)
            proj_A, proj_B = model(img_A, img_B)
            srcc, plcc = calculate_srcc_plcc(proj_A, proj_B)
            srcc_total += srcc
            plcc_total += plcc

    print(f"Test Results: SRCC = {srcc_total / len(test_dataloader):.4f}, PLCC = {plcc_total / len(test_dataloader):.4f}")


""" if __name__ == "__main__":
    from utils.utils import parse_args
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # KADID10KDataset에 전달하는 경로를 문자열로 변환
    train_dataset = KADID10KDataset(str(args.data_base_path) + "/KADID10K", phase="train")
    train_dataloader = DataLoader(train_dataset, batch_size=args.training.batch_size, shuffle=True)

    model = SimCLR(args.model.encoder, args.model.temperature).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.training.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.training.lr_step, gamma=args.training.lr_gamma)

    train(args, model, train_dataloader, optimizer, scheduler, device) """

if __name__ == "__main__":
    from utils.utils import parse_args
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Dataset 설정
    train_dataset = KADID10KDataset(str(args.data_base_path) + "/KADID10K", phase="train")
    val_dataset = KADID10KDataset(str(args.data_base_path) + "/KADID10K", phase="val")
    test_dataset = KADID10KDataset(str(args.data_base_path) + "/KADID10K", phase="test")

    train_dataloader = DataLoader(
        train_dataset, batch_size=args.training.batch_size, shuffle=True, num_workers=args.training.num_workers
    )
    val_dataloader = DataLoader(
        val_dataset, batch_size=args.validation.batch_size, shuffle=False, num_workers=args.validation.num_workers
    )
    test_dataloader = DataLoader(
        test_dataset, batch_size=args.test.batch_size, shuffle=False, num_workers=args.test.num_workers
    )

    # 모델, 옵티마이저, 스케줄러 초기화
    model = SimCLR(args.model.encoder, args.model.temperature).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.training.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.training.lr_step, gamma=args.training.lr_gamma)
    scaler = torch.amp.GradScaler()

    # train 함수 호출
    train(
        args=args,
        model=model,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        test_dataloader=test_dataloader,  # test_dataloader 추가
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        device=device  # device 추가
    )
