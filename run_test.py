import torch
from pathlib import Path
from utils.utils import calculate_srcc_plcc, configure_degradation_model  # configure_degradation_model 임포트
from models.simclr import SimCLR

def test(args, model, dataloader, device):
    model.eval()
    srocc_avg, plcc_avg = 0, 0
    for batch in dataloader:
        # configure_degradation_model 호출
        img_A_orig, img_A_ds, img_B_orig, img_B_ds = configure_degradation_model(batch, device)
        proj_A, proj_B = model(img_A_orig, img_B_orig)
        srocc, plcc = calculate_srcc_plcc(proj_A, proj_B)
        srocc_avg += srocc
        plcc_avg += plcc
    srocc_avg /= len(dataloader)
    plcc_avg /= len(dataloader)
    print(f"Test Results - SRCC: {srocc_avg:.4f}, PLCC: {plcc_avg:.4f}")
