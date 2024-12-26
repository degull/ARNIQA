import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
from torch.utils.data import DataLoader
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from data.dataset_spaq import SPAQDataset
import numpy as np
from torchvision import transforms

# 입력 크기 조정 함수
def prepare_input_for_model(images, target_shape):
    """
    Adjust input images to match the target shape expected by the model.
    """
    batch_size, channels, height, width = images.shape
    print(f"Original input shape: {images.shape}")
    new_height, new_width = target_shape
    if height != new_height or width != new_width:
        print(f"Resizing input to {new_height}x{new_width}...")
        resize_transform = transforms.Compose([
            transforms.Resize((new_height, new_width)),
            transforms.ToTensor()
        ])
        resized_images = torch.stack([resize_transform(img.permute(1, 2, 0).cpu().numpy()) for img in images])
        return resized_images.to(images.device)
    return images

# TorchScript 모델에서 특징 추출
def extract_features_torchscript(model, dataloader, device, target_shape):
    model.eval()
    features = []
    mos_labels = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            print(f"Processing batch {batch_idx + 1}/{len(dataloader)}...")
            images = batch["img_A_orig"].to(device)
            mos = batch["mos"]

            # 입력 크기 조정
            images = prepare_input_for_model(images, target_shape)

            try:
                outputs = model(images)
            except RuntimeError as e:
                print(f"Error during model inference: {e}")
                print(f"Input shape: {images.shape}")
                raise

            features.append(outputs.cpu().numpy())
            mos_labels.extend(mos)

    features = np.vstack(features)
    return features, mos_labels

# t-SNE 시각화 함수
def visualize_tsne(features, labels, title="t-SNE Visualization"):
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    features_2d = tsne.fit_transform(features)

    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(features_2d[:, 0], features_2d[:, 1], c=labels, cmap='viridis', alpha=0.7)
    plt.colorbar(scatter, label='MOS')
    plt.title(title)
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")
    plt.grid(True)
    plt.show()

# 데이터 로더 준비
def prepare_dataloader(dataset_path, batch_size=16, crop_size=224):
    dataset = SPAQDataset(root=dataset_path, crop_size=crop_size)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)

# TorchScript 모델 로드
def load_and_debug_model(model_path, example_input, device):
    model = torch.jit.load(model_path, map_location=device)  # 모델을 디바이스로 이동
    print("Model loaded successfully.")
    print("Example input shape:", example_input.shape)

    try:
        # 모델에 예제 입력 전달
        with torch.no_grad():
            output = model(example_input.to(device))  # 입력 텐서도 디바이스로 이동
        print("Model output shape:", output.shape)
    except RuntimeError as e:
        print("Model failed during inference:", e)
        raise

    return model.to(device)  # 모델을 디바이스로 이동

# 메인 함수
def main():
    dataset_path = "E:/ARNIQA/ARNIQA/dataset/SPAQ"
    model_path = "E:/ARNIQA/ARNIQA/experiments/my_experiment/regressors/regressor_spaq.pth"

    batch_size = 16
    crop_size = 224
    target_shape = (64, 64)  # TorchScript 모델이 기대하는 입력 크기
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 사용 가능한 디바이스 선택

    # 데이터 로더 준비
    dataloader = prepare_dataloader(dataset_path, batch_size=batch_size, crop_size=crop_size)

    # 예제 입력 데이터 생성
    example_input = next(iter(dataloader))["img_A_orig"].to(device)

    # TorchScript 모델 로드 및 디버깅
    model = load_and_debug_model(model_path, example_input, device)

    # 특징 추출
    print("Extracting features using TorchScript model...")
    features, mos_labels = extract_features_torchscript(model, dataloader, device, target_shape)

    # t-SNE 시각화
    print("Visualizing t-SNE for distortion types...")
    visualize_tsne(features, mos_labels, title="t-SNE Visualization of SPAQ Dataset")

if __name__ == "__main__":
    main()
