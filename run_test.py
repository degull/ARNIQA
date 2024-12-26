import torch
from pathlib import Path
from utils.utils import calculate_srcc_plcc, configure_degradation_model  # configure_degradation_model 임포트
from models.simclr import SimCLR

def test(args, model, test_dataloader, device):
    model.eval()
    srcc_total, plcc_total = 0, 0
    with torch.no_grad():
        for batch in test_dataloader:
            img_A, img_B = batch["img_A_orig"].to(device), batch["img_B_orig"].to(device)
            mos = batch["mos"].to(device)

            # 입력 데이터 형태 확인
            print(f"[DEBUG] img_A shape before ResNet: {img_A.shape}")
            print(f"[DEBUG] img_B shape before ResNet: {img_B.shape}")

            # 채널 강제 조정 (필요한 경우)
            if img_A.shape[1] != 3:
                img_A = img_A[:, :3, :, :]
            if img_B.shape[1] != 3:
                img_B = img_B[:, :3, :, :]

            proj_A, proj_B = model(img_A, img_B)
            srcc, plcc = calculate_srcc_plcc(proj_A, proj_B)
            srcc_total += srcc
            plcc_total += plcc

    avg_srcc = srcc_total / len(test_dataloader)
    avg_plcc = plcc_total / len(test_dataloader)
    print(f"Test Results - SRCC: {avg_srcc:.4f}, PLCC: {avg_plcc:.4f}")

