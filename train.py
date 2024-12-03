
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

def train(args, model, train_dataloader, optimizer, scheduler, device):
    for epoch in range(args.training.epochs):
        model.train()
        for batch_idx, batch in enumerate(train_dataloader):
            # 데이터를 GPU로 이동
            img_A, img_B = batch["img_A_orig"].to(device), batch["img_B_orig"].to(device)

            # 크롭 수 확인
            batch_size, num_crops, _, _, _ = img_A.shape
            expected_crops = 4  # 1 original + 3 augmentations
            assert num_crops == expected_crops, (
                f"Batch {batch_idx}: Expected {expected_crops} crops per sample, but got {num_crops}."
            )

            # 배치 크기 조정
            img_A = img_A.view(-1, *img_A.shape[2:])  # [batch_size * num_crops, C, H, W]
            img_B = img_B.view(-1, *img_B.shape[2:])  # [batch_size * num_crops, C, H, W]

            optimizer.zero_grad()

            # 모델 Forward 및 손실 계산
            proj_A, proj_B = model(img_A, img_B)
            assert proj_A.requires_grad, "proj_A does not require grad. Check model's forward function."
            assert proj_B.requires_grad, "proj_B does not require grad. Check model's forward function."
            
            loss = model.compute_loss(proj_A, proj_B)
            loss.backward()
            optimizer.step()

        scheduler.step()  # 이 부분에서 TypeError가 발생할 수 있습니다
        print(f"Epoch {epoch + 1}: Loss = {loss.item()}")

    # 선형 회귀 학습
    model.linear_regressor.train()
    all_features = []
    all_labels = []
    for batch in train_dataloader:
        img_A, img_B = batch["img_A_orig"].to(device), batch["img_B_orig"].to(device)
        img_A = img_A.view(-1, *img_A.shape[2:])
        _, proj_A = model.encoder(img_A)
        all_features.append(proj_A.detach().cpu())
        all_labels.append(batch["mos"])
    all_features = torch.cat(all_features, dim=0)
    all_labels = torch.tensor(all_labels).float()

    # 선형 회귀기 업데이트
    regressor = Ridge(alpha=1.0)
    regressor.fit(all_features.numpy(), all_labels.numpy())
    print("Linear regressor training completed.")




if __name__ == "__main__":
    from utils.utils import parse_args
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # KADID10KDataset에 전달하는 경로를 문자열로 변환
    train_dataset = KADID10KDataset(str(args.data_base_path) + "/KADID10K", phase="train")
    train_dataloader = DataLoader(train_dataset, batch_size=args.training.batch_size, shuffle=True)

    model = SimCLR(args.model.encoder, args.model.temperature).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.training.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.training.lr_step, gamma=args.training.lr_gamma)

    train(args, model, train_dataloader, optimizer, scheduler, device)
