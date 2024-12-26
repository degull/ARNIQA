""" import pandas as pd
import re
import numpy as np
from PIL import Image, ImageFilter
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import random
from pathlib import Path

# Distortion types mapping
distortion_types_mapping = {
    1: "gaussian_blur",
    2: "lens_blur",
    3: "motion_blur",
    4: "color_diffusion",
    5: "color_shift",
    6: "color_quantization",
    7: "color_saturation_1",
    8: "color_saturation_2",
    9: "jpeg2000",
    10: "jpeg",
    11: "white_noise",
    12: "white_noise_color_component",
    13: "impulse_noise",
    14: "multiplicative_noise",
    15: "denoise",
    16: "brighten",
    17: "darken",
    18: "mean_shift",
    19: "jitter",
    20: "non_eccentricity_patch",
    21: "pixelate",
    22: "quantization",
    23: "color_block",
    24: "high_sharpen",
    25: "contrast_change"
}


class KADID10KDataset(Dataset):
    def __init__(self, root: str, phase: str = "train", crop_size: int = 224):
        self.root = Path(root)
        self.phase = phase
        self.crop_size = crop_size

        # Load scores from CSV
        scores_csv = pd.read_csv(self.root / "kadid10k.csv")
        scores_csv = scores_csv[["dist_img", "ref_img", "dmos"]]

        # Assuming images are in the 'images' folder
        self.images = np.array([self.root / "images" / img for img in scores_csv["dist_img"].values])
        self.ref_images = np.array([self.root / "images" / img for img in scores_csv["ref_img"].values])
        self.mos = np.array(scores_csv["dmos"].values.tolist())

        for img in self.images:
            if not img.exists():
                print(f"Image not found: {img}")

    def transform(self, image):
        return transforms.Compose([
            transforms.Resize((self.crop_size, self.crop_size)),
            transforms.ToTensor(),
        ])(image)

    def apply_distortion(self, image):
        # Convert to PIL image for applying distortion
        pil_image = transforms.ToPILImage()(image)
        if random.random() > 0.5:
            return transforms.ToTensor()(pil_image.filter(ImageFilter.GaussianBlur(radius=2)))
        return image

    def __getitem__(self, index: int) -> dict:
        img_A_orig = Image.open(self.images[index]).convert("RGB")
        img_B_orig = Image.open(self.ref_images[index]).convert("RGB")

        img_A_orig = self.transform(img_A_orig)
        img_B_orig = self.transform(img_B_orig)

        # Create crops and stack images
        crops_A = [img_A_orig]
        crops_B = [img_B_orig]

        # Apply additional crops
        crops_A += [self.apply_distortion(img_A_orig) for _ in range(3)]
        crops_B += [self.apply_distortion(img_B_orig) for _ in range(3)]

        # Stack crops
        img_A = torch.stack(crops_A)  # Shape: [num_crops, 3, crop_size, crop_size]
        img_B = torch.stack(crops_B)  # Shape: [num_crops, 3, crop_size, crop_size]

        # Reshape to [1, num_crops, 3, crop_size, crop_size]
        img_A = img_A.unsqueeze(0)  
        img_B = img_B.unsqueeze(0)

        return {
            "img_A_orig": img_A,
            "img_B_orig": img_B,
            "img_A_ds": img_A,
            "img_B_ds": img_B,
            "mos": self.mos[index],
        }

    def __len__(self):
        return len(self.images)

    def get_split_indices(self, split: int, phase: str) -> np.ndarray:

        split_file_path = self.root / "splits" / f"{phase}.npy"
        split_indices = np.load(split_file_path)[split]
        return split_indices """

""" import pandas as pd
import re
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from pathlib import Path
from PIL import ImageFilter
import random

# 왜곡 유형 매핑
distortion_types_mapping = {
    1: "gaussian_blur",
    2: "lens_blur",
    3: "motion_blur",
    4: "color_diffusion",
    5: "color_shift",
    6: "color_quantization",
    7: "color_saturation_1",
    8: "color_saturation_2",
    9: "jpeg2000",
    10: "jpeg",
    11: "white_noise",
    12: "white_noise_color_component",
    13: "impulse_noise",
    14: "multiplicative_noise",
    15: "denoise",
    16: "brighten",
    17: "darken",
    18: "mean_shift",
    19: "jitter",
    20: "non_eccentricity_patch",
    21: "pixelate",
    22: "quantization",
    23: "color_block",
    24: "high_sharpen",
    25: "contrast_change"
}

class KADID10KDataset(Dataset):
    def __init__(self, root: str, phase: str = "train", split_idx: int = 0, crop_size: int = 224):
        super().__init__()
        self.root = Path(root)
        self.phase = phase
        self.crop_size = crop_size

        # CSV 파일에서 점수 로드
        scores_csv = pd.read_csv(self.root / "kadid10k.csv")
        scores_csv = scores_csv[["dist_img", "ref_img", "dmos"]]

        # 이미지 경로 설정
        self.images = np.array([self.root / "images" / img for img in scores_csv["dist_img"].values])
        self.ref_images = np.array([self.root / "images" / img for img in scores_csv["ref_img"].values])
        self.mos = np.array(scores_csv["dmos"].values.tolist())

        self.distortion_types = []
        self.distortion_levels = []

        for img in self.images:
            # 이미지 이름에서 왜곡 유형과 레벨 추출
            match = re.search(r'I\d+_(\d+)_(\d+)\.png$', str(img))
            if match:
                dist_type = distortion_types_mapping[int(match.group(1))]
                self.distortion_types.append(dist_type)
                self.distortion_levels.append(int(match.group(2)))

        self.distortion_types = np.array(self.distortion_types)
        self.distortion_levels = np.array(self.distortion_levels)

        if self.phase != "all":
            split_idxs = np.load(self.root / "splits" / f"{self.phase}.npy")[split_idx]
            split_idxs = np.array(list(filter(lambda x: x != -1, split_idxs)))  # 패딩 제거
            self.images = self.images[split_idxs]
            self.ref_images = self.ref_images[split_idxs]
            self.mos = self.mos[split_idxs]
            self.distortion_types = self.distortion_types[split_idxs]
            self.distortion_levels = self.distortion_levels[split_idxs]

    def transform(self, image: Image) -> torch.Tensor:
        # Transform image to desired size and convert to tensor
        return transforms.Compose([
            transforms.Resize((self.crop_size, self.crop_size)),
            transforms.ToTensor(),
        ])(image)
    
    def apply_distortion(self, image: torch.Tensor) -> torch.Tensor:
        # Convert tensor to PIL image
        pil_image = transforms.ToPILImage()(image)
        
        # Apply a random distortion (e.g., Gaussian Blur)
        if random.random() > 0.5:
            return transforms.ToTensor()(pil_image.filter(ImageFilter.GaussianBlur(radius=2)))
        
        return image  # Return original image if no distortion applied


    def __getitem__(self, index: int) -> dict:
        img_A_orig = Image.open(self.images[index]).convert("RGB")
        img_B_orig = Image.open(self.ref_images[index]).convert("RGB")

        img_A_orig = self.transform(img_A_orig)
        img_B_orig = self.transform(img_B_orig)

        # Create crops and stack images
        crops_A = [img_A_orig]
        crops_B = [img_B_orig]

        # Apply additional crops
        crops_A += [self.apply_distortion(img_A_orig) for _ in range(3)]
        crops_B += [self.apply_distortion(img_B_orig) for _ in range(3)]

        # Stack crops
        img_A = torch.stack(crops_A)  # Shape: [num_crops, 3, crop_size, crop_size]
        img_B = torch.stack(crops_B)  # Shape: [num_crops, 3, crop_size, crop_size]

        # Reshape to [1, num_crops, 3, crop_size, crop_size]
        img_A = img_A.unsqueeze(0)
        img_B = img_B.unsqueeze(0)

        return {
            "img_A_orig": img_A,
            "img_B_orig": img_B,
            "img_A_ds": img_A,  # img_A_ds를 추가합니다
            "img_B_ds": img_B,  # img_B_ds를 추가합니다
            "mos": self.mos[index],
        }

    def __len__(self):
        return len(self.images)

    def get_split_indices(self, split: int, phase: str) -> np.ndarray:
        split_file_path = self.root / "splits" / f"{phase}.npy"
        split_indices = np.load(split_file_path)[split]
        return split_indices
 """

import sys
import os

# 현재 스크립트 디렉토리를 PYTHONPATH에 추가
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


import pandas as pd
import re
import numpy as np
from pathlib import Path
from PIL import Image, ImageFilter
import random
import torch
from torchvision import transforms
from data.dataset_synthetic_base_iqa import SyntheticIQADataset
from torch.utils.data import random_split, DataLoader
from torchvision.transforms import functional as F
import sys
import os
from PIL import Image
from torchvision import transforms

# 현재 파일의 상위 디렉토리를 PYTHONPATH에 추가
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# 왜곡 타입 매핑 (예시로 포함)
distortion_types_mapping = {
    1: "gaussian_blur",
    2: "lens_blur",
    3: "motion_blur",
    4: "color_diffusion",
    5: "color_shift",
    6: "color_quantization",
    7: "color_saturation_1",
    8: "color_saturation_2",
    9: "jpeg2000",
    10: "jpeg",
    11: "white_noise",
    12: "white_noise_color_component",
    13: "impulse_noise",
    14: "multiplicative_noise",
    15: "denoise",
    16: "brighten",
    17: "darken",
    18: "mean_shift",
    19: "jitter",
    20: "non_eccentricity_patch",
    21: "pixelate",
    22: "quantization",
    23: "color_block",
    24: "high_sharpen",
    25: "contrast_change"
}



    # Split dataset
def split_dataset(dataset, train_ratio=0.7, val_ratio=0.2, test_ratio=0.1, seed=42):
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1."
    dataset_length = len(dataset)
    train_len = int(dataset_length * train_ratio)
    val_len = int(dataset_length * val_ratio)
    test_len = dataset_length - train_len - val_len
    generator = torch.Generator().manual_seed(seed)
    return random_split(dataset, [train_len, val_len, test_len], generator=generator)


# 데이터셋 클래스 정의
class KADID10KDataset(SyntheticIQADataset):

    def __init__(self, root: str, phase: str = "all", crop_size: int = 224, train_ratio=0.7, val_ratio=0.2, test_ratio=0.1, seed=42):
        self.crop_size = crop_size

        # Load CSV file
        scores_csv = pd.read_csv(Path(root) / "kadid10k.csv")
        scores_csv = scores_csv[["dist_img", "ref_img", "dmos"]]

        # Load image paths and MOS values
        self.images = np.array([Path(root) / "images" / img for img in scores_csv["dist_img"].values])
        self.ref_images = np.array([Path(root) / "images" / img for img in scores_csv["ref_img"].values])
        self.mos = np.array(scores_csv["dmos"].values.tolist())

        # Split dataset
        dataset_length = len(self.images)
        train_len = int(dataset_length * train_ratio)
        val_len = int(dataset_length * val_ratio)
        test_len = dataset_length - train_len - val_len

        generator = torch.Generator().manual_seed(seed)
        indices = torch.randperm(dataset_length, generator=generator).tolist()

        self.train_indices = indices[:train_len]
        self.val_indices = indices[train_len:train_len + val_len]
        self.test_indices = indices[train_len + val_len:]

        # Select indices based on phase
        if phase == "train":
            selected_indices = self.train_indices
        elif phase == "val":
            selected_indices = self.val_indices
        elif phase == "test":
            selected_indices = self.test_indices
        elif phase == "all":
            selected_indices = indices
        else:
            raise ValueError(f"Invalid phase: {phase}")

        # Set data for selected indices
        self.images = self.images[selected_indices]
        self.ref_images = self.ref_images[selected_indices]
        self.mos = self.mos[selected_indices]

    
    def transform(self, image: Image) -> torch.Tensor:
        return transforms.Compose([
            transforms.Resize((self.crop_size, self.crop_size)),
            transforms.ToTensor(),
        ])(image)

    def __getitem__(self, index):
        img_A_orig = Image.open(self.images[index]).convert("RGB")
        img_B_orig = Image.open(self.ref_images[index]).convert("RGB")

        img_A_orig = self.transform(img_A_orig)  # [C, H, W]
        img_B_orig = self.transform(img_B_orig)  # [C, H, W]

        # 디버그: 입력 데이터 형태 확인
        #print(f"[DEBUG] img_A_orig shape: {img_A_orig.shape}, img_B_orig shape: {img_B_orig.shape}")

        # 채널 강제 변환
        if img_A_orig.shape[0] != 3:
            img_A_orig = img_A_orig[:3, :, :]
        if img_B_orig.shape[0] != 3:
            img_B_orig = img_B_orig[:3, :, :]

        return {
            "img_A_orig": img_A_orig,
            "img_B_orig": img_B_orig,
            "mos": self.mos[index],
        }


    def __len__(self):
        return len(self.images)



if __name__ == "__main__":
    dataset_path = "E:/ARNIQA/ARNIQA/dataset/KADID10K"

    # Load full dataset
    full_dataset = KADID10KDataset(root=dataset_path)

    # Split dataset
    train_dataset, val_dataset, test_dataset = split_dataset(full_dataset, train_ratio=0.7, val_ratio=0.2, test_ratio=0.1)

    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(val_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")

    # Create DataLoaders
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)

    # Verify DataLoader
    for batch in train_dataloader:
        print(f"Train batch size: {batch['img_A_orig'].shape}, {batch['img_B_orig'].shape}")
        break