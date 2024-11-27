""" import pandas as pd
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from pathlib import Path
import torch

class SPAQDataset(Dataset):
    def __init__(self, root: str, phase: str = "train", crop_size: int = 224):
        super().__init__()
        self.root = Path(root)
        self.phase = phase
        self.crop_size = crop_size

        # Load scores from CSV
        print("Loading scores from CSV...")
        scores_csv = pd.read_csv(self.root / "Annotations" / "MOS and Image attribute scores.csv")
        print("Scores CSV head:", scores_csv.head())  # Debugging output

        self.images = scores_csv["Image name"].values.tolist()
        self.images = np.array([self.root / "TestImage" / img for img in self.images])
        self.mos = np.array(scores_csv["MOS"].values.tolist())

        print(f"Loaded {len(self.images)} images with corresponding MOS scores.")

        if self.phase != "all":
            split_idxs = np.load(self.root / "splits" / f"{self.phase}.npy")
            self.images = self.images[split_idxs]
            self.mos = self.mos[split_idxs]

    def __getitem__(self, index: int) -> dict:
        img_A_orig = Image.open(self.images[index]).convert("RGB")
        img_A_orig = img_A_orig.resize((self.crop_size, self.crop_size), Image.BICUBIC)

        # Create crops for img_A_orig
        crops = center_corners_crop(img_A_orig, crop_size=self.crop_size)
        crops = [transforms.ToTensor()(crop) for crop in crops]

        # Stack crops and normalize
        img_A = torch.stack(crops, dim=0)
        img_A = img_A.unsqueeze(0)  # Shape: [1, num_crops, 3, crop_size, crop_size]

        # Create a copy for img_B
        img_B = img_A.clone()

        return {
            "img_A_orig": img_A,
            "img_B_orig": img_B,
            "img_A_ds": img_A,
            "img_B_ds": img_B,
            "mos": self.mos[index],
        }

    def __len__(self):
        return len(self.images)

def center_corners_crop(image: Image, crop_size: int) -> list:
    # Crop the image into center and corners as defined before
    width, height = image.size
    crops = []

    # Center crop
    left = (width - crop_size) // 2
    top = (height - crop_size) // 2
    crops.append(image.crop((left, top, left + crop_size, top + crop_size)))

    # Four corners crop
    crops.append(image.crop((0, 0, crop_size, crop_size)))  # Top-left
    crops.append(image.crop((width - crop_size, 0, width, crop_size)))  # Top-right
    crops.append(image.crop((0, height - crop_size, crop_size, height)))  # Bottom-left
    crops.append(image.crop((width - crop_size, height - crop_size, width, height)))  # Bottom-right

    return crops
 """

## 원본 코드
""" 
import pandas as pd
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from pathlib import Path
from utils.utils_data import resize_crop, center_corners_crop

class SPAQDataset(Dataset):

    def __init__(self,
                 root: str,
                 phase: str = "train",
                 split_idx: int = 0,
                 crop_size: int = 224):
        super().__init__()
        self.root = Path(root)
        self.phase = phase
        self.crop_size = crop_size

        # Load scores from CSV
        print("Loading scores from CSV...")
        scores_csv = pd.read_csv(self.root / "Annotations" / "MOS and Image attribute scores.csv")
        self.images = scores_csv["Image name"].values.tolist()
        self.images = np.array([self.root / "TestImage" / img for img in self.images])
        self.mos = np.array(scores_csv["MOS"].values.tolist())

        if self.phase != "all":
            split_idxs = np.load(self.root / "splits" / f"{self.phase}.npy")
            self.images = self.images[split_idxs]
            self.mos = self.mos[split_idxs]

        self.target_size = 512

    def __getitem__(self, index: int) -> dict:
        img = Image.open(self.images[index]).convert("RGB")

        # Resize image to target size while maintaining aspect ratio
        width, height = img.size
        aspect_ratio = width / height
        if width < height:
            new_width = self.target_size
            new_height = int(self.target_size / aspect_ratio)
        else:
            new_height = self.target_size
            new_width = int(self.target_size * aspect_ratio)

        img = img.resize((new_width, new_height), Image.BICUBIC)

        # Create crops
        img_ds = resize_crop(img, crop_size=None, downscale_factor=2)
        crops = center_corners_crop(img, crop_size=self.crop_size)
        crops = [transforms.ToTensor()(crop) for crop in crops]

        # Stack crops and normalize
        img_A = torch.stack(crops, dim=0)  # Shape: [num_crops, 3, crop_size, crop_size]
        crops_ds = center_corners_crop(img_ds, crop_size=self.crop_size)
        crops_ds = [transforms.ToTensor()(crop) for crop in crops_ds]
        img_B = torch.stack(crops_ds, dim=0)  # Distorted images

        img_A = self.normalize(img_A)
        img_B = self.normalize(img_B)
        mos = self.mos[index]

        return {
            "img_A_orig": img_A,
            "img_B_orig": img_B,
            "img_A_ds": img_A,  # 여기가 문제일 수 있습니다
            "img_B_ds": img_B,
            "mos": mos
        }

    def __len__(self):
        return len(self.images)

# Center corners crop function definition
def center_corners_crop(image: Image, crop_size: int) -> list:
    # Crop the image into center and corners
    width, height = image.size
    crops = []

    # Center crop
    left = (width - crop_size) // 2
    top = (height - crop_size) // 2
    crops.append(image.crop((left, top, left + crop_size, top + crop_size)))

    # Four corners crop
    crops.append(image.crop((0, 0, crop_size, crop_size)))  # Top-left
    crops.append(image.crop((width - crop_size, 0, width, crop_size)))  # Top-right
    crops.append(image.crop((0, height - crop_size, crop_size, height)))  # Bottom-left
    crops.append(image.crop((width - crop_size, height - crop_size, width, height)))  # Bottom-right

    return crops
 """

## 시각화하는 코드

import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from dotmap import DotMap
from models.simclr import SimCLR


# Custom SPAQ Dataset
class SPAQDataset(Dataset):
    """
    SPAQ IQA dataset with MOS in range [1, 100]. Resizes the images so that the smallest side is 512 pixels.
    """
    def __init__(self, root: str, crop_size: int = 224, phase: str = "train"):
        super().__init__()
        self.root = Path(root)
        self.crop_size = crop_size
        self.phase = phase

        # Load scores from CSV
        print("Loading scores from CSV...")
        scores_csv = pd.read_csv(self.root / "Annotations" / "MOS and Image attribute scores.csv")
        self.images = scores_csv["Image name"].values.tolist()
        self.images = np.array([self.root / "TestImage" / img for img in self.images])
        self.mos = np.array(scores_csv["MOS"].values.tolist())

        self.target_size = 512

    def resize_image(self, img: Image) -> Image:
        """
        Resize image to target size while maintaining aspect ratio.
        """
        width, height = img.size
        aspect_ratio = width / height
        if width < height:
            new_width = self.target_size
            new_height = int(self.target_size / aspect_ratio)
        else:
            new_height = self.target_size
            new_width = int(self.target_size * aspect_ratio)

        return img.resize((new_width, new_height), Image.BICUBIC)

    def __getitem__(self, index: int) -> dict:
        img_path = self.images[index]
        img = Image.open(img_path).convert("RGB")
        img = self.resize_image(img)

        # Create crops
        img_ds = resize_crop(img, crop_size=None, downscale_factor=2)
        crops = center_corners_crop(img, crop_size=self.crop_size)
        crops = [transforms.ToTensor()(crop) for crop in crops]

        # Stack crops and normalize
        img_A = torch.stack(crops, dim=0)  # Shape: [num_crops, 3, crop_size, crop_size]
        crops_ds = center_corners_crop(img_ds, crop_size=self.crop_size)
        crops_ds = [transforms.ToTensor()(crop) for crop in crops_ds]
        img_B = torch.stack(crops_ds, dim=0)  # Distorted images

        img_A = self.normalize(img_A)
        img_B = self.normalize(img_B)
        mos = self.mos[index]

        return {
            "img_A_orig": img_A,
            "img_B_orig": img_B,
            "img_A_ds": img_A,
            "img_B_ds": img_B,
            "mos": mos
        }

    def __len__(self):
        return len(self.images)

    def normalize(self, tensor):
        return transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(tensor)


# Center corners crop function
def center_corners_crop(image: Image, crop_size: int) -> list:
    width, height = image.size
    crops = []

    # Center crop
    left = (width - crop_size) // 2
    top = (height - crop_size) // 2
    crops.append(image.crop((left, top, left + crop_size, top + crop_size)))

    # Four corners crop
    crops.append(image.crop((0, 0, crop_size, crop_size)))  # Top-left
    crops.append(image.crop((width - crop_size, 0, width, crop_size)))  # Top-right
    crops.append(image.crop((0, height - crop_size, crop_size, height)))  # Bottom-left
    crops.append(image.crop((width - crop_size, height - crop_size, width, height)))  # Bottom-right

    return crops


# Resize crop function
def resize_crop(img: Image, crop_size: int = None, downscale_factor: float = 2):
    width, height = img.size
    img = img.resize((int(width / downscale_factor), int(height / downscale_factor)), Image.BICUBIC)
    if crop_size:
        left = (img.size[0] - crop_size) // 2
        top = (img.size[1] - crop_size) // 2
        img = img.crop((left, top, left + crop_size, top + crop_size))
    return img


# Extract Features
def extract_features(model, dataloader, device):
    model.eval()
    features = []
    labels = []

    with torch.no_grad():
        for batch in dataloader:
            images = batch["img_A_orig"].to(device)
            mos = batch["mos"]

            outputs, _ = model.encoder(images.view(-1, *images.shape[2:]))  # Flatten crops
            features.append(outputs.cpu().numpy())
            labels.extend(mos.numpy())

    features = np.vstack(features)
    return features, np.array(labels)


# Visualize Features with t-SNE
def visualize_tsne(features, labels):
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    features_2d = tsne.fit_transform(features)

    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(features_2d[:, 0], features_2d[:, 1], c=labels, cmap='viridis', alpha=0.7)
    plt.colorbar(scatter, label='MOS')
    plt.title("t-SNE Visualization of SPAQ Dataset")
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")
    plt.grid(True)
    plt.show()


# Load Modified State Dict
def load_modified_state_dict(model, checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    state_dict = checkpoint
    model_state_dict = model.state_dict()

    new_state_dict = {}
    for key in model_state_dict.keys():
        if key in state_dict and model_state_dict[key].shape == state_dict[key].shape:
            new_state_dict[key] = state_dict[key]
        else:
            print(f"Skipping key: {key} (Shape mismatch or not found in checkpoint)")

    model.load_state_dict(new_state_dict, strict=False)
    return model


# Main Function
def main():
    dataset_path = "E:/SPAQ"  # SPAQ 데이터셋 루트 경로
    model_path = "E:/ARNIQA/ARNIQA/checkpoints/arniqa_checkpoint.pth"  # SimCLR 체크포인트 경로
    batch_size = 16
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load SimCLR Model
    encoder_params = DotMap({
        'embedding_dim': 128,
        'pretrained': False,
        'use_norm': True
    })
    model = SimCLR(encoder_params).to(device)
    model = load_modified_state_dict(model, model_path)

    # Prepare DataLoader
    dataloader = DataLoader(SPAQDataset(root=dataset_path), batch_size=batch_size, shuffle=False)

    # Extract Features and Visualize
    features, labels = extract_features(model, dataloader, device)
    visualize_tsne(features, labels)


if __name__ == "__main__":
    main()
