import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.linear_model import Ridge
from data import KADID10KDataset
from models.simclr import SimCLR
from utils.utils import (
    calculate_srcc_plcc,
    parse_args
)

def apply_gaussian_noise(img, mean=0, std=0.1):
    """
    이미지에 가우시안 노이즈를 추가하는 함수
    """
    noise = torch.normal(mean=mean, std=std, size=img.size()).to(img.device)
    noisy_img = img + noise
    return torch.clamp(noisy_img, 0, 1)

def downsample_image(img, scale_factor=0.5):
    """
    이미지를 50%로 다운샘플링하는 함수
    """
    batch_size, channels, height, width = img.size()
    new_height = int(height * scale_factor)
    new_width = int(width * scale_factor)
    downsampled_img = nn.functional.interpolate(img, size=(new_height, new_width), mode='bilinear', align_corners=False)
    return downsampled_img

def train(args, model, train_dataloader, optimizer, scheduler, device):
    for epoch in range(args.training.epochs):
        model.train()
        print(f"Epoch {epoch + 1}/{args.training.epochs} 시작")
        for batch_idx, batch in enumerate(train_dataloader):
            img_A, img_B = batch["img_A_orig"].to(device), batch["img_B_orig"].to(device)
            mos = batch["mos"].to(device)

            # 배치 크기 조정
            img_A = img_A.view(-1, *img_A.shape[2:])  # [batch_size * num_crops, C, H, W]
            img_B = img_B.view(-1, *img_B.shape[2:])  # [batch_size * num_crops, C, H, W]
            mos = mos.repeat(img_A.size(0) // mos.size(0))  # MOS 확장

            # 다운샘플링 처리
            img_A_ds = downsample_image(img_A)
            img_B_ds = downsample_image(img_B)

            optimizer.zero_grad()

            # 모델 Forward 및 손실 계산
            proj_A, proj_B = model(img_A, img_B)
            proj_A_ds, proj_B_ds = model(img_A_ds, img_B_ds)
            
            # 양성 및 음성 쌍 손실 계산
            loss = model.compute_loss(proj_A, proj_B, mos)
            loss_ds = model.compute_loss(proj_A_ds, proj_B_ds, mos)  # 경음성 쌍 손실
            total_loss = loss + loss_ds

            total_loss.backward()
            optimizer.step()

            # 로그 출력
            if batch_idx % 10 == 0:
                print(f"Epoch {epoch + 1}, Batch {batch_idx}/{len(train_dataloader)}: Positive Loss = {loss.item()}, Negative Loss = {loss_ds.item()}, Total Loss = {total_loss.item()}")

        scheduler.step()
        print(f"Epoch {epoch + 1}: Final Loss = {total_loss.item()}")

        # SRCC, PLCC 계산
        srcc, plcc = calculate_srcc_plcc(proj_A, proj_B, mos)
        print(f"Epoch {epoch + 1} - SRCC: {srcc:.4f}, PLCC: {plcc:.4f}")

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

def test_batch_generation(batch, model, device, mos):
    """
    배치의 경음성 쌍, 손실 함수 및 SRCC/PLCC 계산을 점검하는 함수입니다.
    """
    img_A, img_B = batch["img_A_orig"].to(device), batch["img_B_orig"].to(device)
    img_A_ds, img_B_ds = downsample_image(img_A), downsample_image(img_B)

    print(f"Original Image Size: {img_A.shape}")
    print(f"Downsampled Image Size: {img_A_ds.shape}")

    # 손실 함수 확인
    proj_A, proj_B = model(img_A, img_B)
    proj_A_ds, proj_B_ds = model(img_A_ds, img_B_ds)
    loss = model.compute_loss(proj_A, proj_B, mos)
    loss_ds = model.compute_loss(proj_A_ds, proj_B_ds, mos)

    print(f"Positive Pair Loss: {loss.item()}, Negative Pair Loss: {loss_ds.item()}")

    srcc, plcc = calculate_srcc_plcc(proj_A, proj_B, mos)
    print(f"SRCC: {srcc:.4f}, PLCC: {plcc:.4f}")

if __name__ == "__main__":
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 데이터셋 및 DataLoader 초기화
    train_dataset = KADID10KDataset(str(args.data_base_path) + "/KADID10K", phase="train")
    train_dataloader = DataLoader(train_dataset, batch_size=args.training.batch_size, shuffle=True)

    # 모델, 옵티마이저 및 스케줄러 초기화
    model = SimCLR(args.model.encoder_params, args.model.temperature).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.training.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=args.training.step_size,
        gamma=args.training.gamma
    )

    # 학습 시작
    train(args, model, train_dataloader, optimizer, scheduler, device)
