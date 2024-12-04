import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.linear_model import Ridge
from data import KADID10KDataset
from models.simclr import SimCLR
import matplotlib.pyplot as plt
from torchvision.transforms import functional as F  # F 정의 추가
from torchvision import transforms  # transforms 정의 추가
import random
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
    Downsample the image by 50%.
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

            # 다운샘플링 처리
            img_A_ds = downsample_image(img_A)
            img_B_ds = downsample_image(img_B)

            optimizer.zero_grad()

            # 모델 Forward 및 손실 계산
            proj_A, proj_B = model(img_A, img_B)
            proj_A_ds, proj_B_ds = model(img_A_ds, img_B_ds)
            loss = model.compute_loss(proj_A, proj_B, mos)
            loss_ds = model.compute_loss(proj_A_ds, proj_B_ds, mos)
            total_loss = loss + loss_ds

            total_loss.backward()
            optimizer.step()

            # SRCC 및 PLCC 계산
            srcc, plcc = calculate_srcc_plcc(proj_A, proj_B, mos)

            # 로그 출력
            print(
                f"Epoch {epoch + 1}, Batch {batch_idx}/{len(train_dataloader)}: "
                f"Positive Loss = {loss.item()}, Negative Loss = {loss_ds.item()}, Total Loss = {total_loss.item()}, "
                f"SRCC = {srcc:.4f}, PLCC = {plcc:.4f}"
            )

        scheduler.step()
        print(f"Epoch {epoch + 1}: Final Loss = {total_loss.item()}")

    print("Training completed.")


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
    random_index = random.randint(0, len(train_dataset) - 1)
    sample = train_dataset[random_index]

    # 원본 이미지와 왜곡 이미지
    img_A_orig = sample["img_A_orig"][0]
    distorted_image, distortions = train_dataset.apply_distortion(F.to_pil_image(img_A_orig))

    print(f"적용된 왜곡: {distortions}")

    # 시각화
    plt.subplot(1, 2, 1)
    plt.title("Original Image A")
    plt.imshow(img_A_orig.permute(1, 2, 0))  # C, H, W -> H, W, C

    plt.subplot(1, 2, 2)
    plt.title(f"Distorted Image A\nApplied: {distortions}")
    plt.imshow(transforms.ToTensor()(distorted_image).permute(1, 2, 0))  # PIL -> Tensor

    plt.show()