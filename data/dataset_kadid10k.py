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
from torch.utils.data import Dataset
import torch.nn.functional as F
import os
from torchvision.transforms import functional as F
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

# 데이터셋 클래스 정의
class KADID10KDataset(Dataset):
    def __init__(self, root: str, phase: str = "all", crop_size: int = 224, train_ratio=0.7, val_ratio=0.2, test_ratio=0.1, seed=42):
        self.crop_size = crop_size
        mos_type = "dmos"
        mos_range = (1, 5)
        is_synthetic = True
        super().__init__()

        # CSV 파일 로드
        scores_csv = pd.read_csv(Path(root) / "kadid10k.csv")
        scores_csv = scores_csv[["dist_img", "ref_img", "dmos"]]

        # 이미지 경로 및 MOS 값 로드
        self.images = np.array([Path(root) / "images" / img for img in scores_csv["dist_img"].values])
        self.ref_images = np.array([Path(root) / "images" / img for img in scores_csv["ref_img"].values])
        self.mos = np.array(scores_csv["dmos"].values.tolist())

        # 왜곡 정보 파싱
        self.distortion_types = []
        self.distortion_levels = []
        for img in self.images:
            match = re.search(r'I\d+_(\d+)_(\d+)\.png$', str(img))
            if match:
                dist_type = distortion_types_mapping[int(match.group(1))]
                self.distortion_types.append(dist_type)
                self.distortion_levels.append(int(match.group(2)))

        self.distortion_types = np.array(self.distortion_types)
        self.distortion_levels = np.array(self.distortion_levels)

        # 데이터셋 분리
        dataset_length = len(self.images)
        train_len = int(dataset_length * train_ratio)
        val_len = int(dataset_length * val_ratio)
        test_len = dataset_length - train_len - val_len

        generator = torch.Generator().manual_seed(seed)
        indices = torch.randperm(dataset_length, generator=generator).tolist()

        self.train_indices = indices[:train_len]
        self.val_indices = indices[train_len:train_len + val_len]
        self.test_indices = indices[train_len + val_len:]

        # Phase 처리
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

        # 선택된 인덱스에 따라 데이터 설정
        self.images = self.images[selected_indices]
        self.ref_images = self.ref_images[selected_indices]
        self.mos = self.mos[selected_indices]
        self.distortion_types = self.distortion_types[selected_indices]
        self.distortion_levels = self.distortion_levels[selected_indices]

    def transform(self, image: Image) -> torch.Tensor:
        # 이미지 변환
        return transforms.Compose([transforms.Resize((self.crop_size, self.crop_size)),
                                   transforms.ToTensor()])(image)

    def apply_distortion(self, image: Image) -> Image:
        # 왜곡 강도를 가우시안 분포에서 샘플링
        gaussian_strength = np.random.normal(loc=0.5, scale=0.15)  # 평균 0.5, 표준편차 0.15
        if random.random() > 0.5:
            image = image.filter(ImageFilter.GaussianBlur(radius=gaussian_strength))
        return image

    def __getitem__(self, index: int) -> dict:
        img_A_orig = Image.open(self.images[index]).convert("RGB")
        img_B_orig = Image.open(self.ref_images[index]).convert("RGB")

        # 고해상도 변환
        img_A_orig = self.transform(img_A_orig)
        img_B_orig = self.transform(img_B_orig)

        # 다운스케일링
        img_A_ds = F.resize(img_A_orig, (self.crop_size // 2, self.crop_size // 2))
        img_B_ds = F.resize(img_B_orig, (self.crop_size // 2, self.crop_size // 2))

        # 크롭 생성 (원본 + 3개의 왜곡 크롭)
        crops_A = [img_A_orig] + [self.transform(self.apply_distortion(F.to_pil_image(img_A_orig))) for _ in range(3)]
        crops_B = [img_B_orig] + [self.transform(self.apply_distortion(F.to_pil_image(img_B_orig))) for _ in range(3)]

        # 경음성 쌍 포함 (50% 다운샘플링 비율)
        hard_neg_A = torch.stack(crops_A)
        hard_neg_B = torch.stack(crops_B)

        return {
            "img_A_orig": hard_neg_A,  # [num_crops, C, H, W]
            "img_B_orig": hard_neg_B,  # [num_crops, C, H, W]
            "img_A_ds": img_A_ds,  # 다운샘플링된 img_A
            "img_B_ds": img_B_ds,  # 다운샘플링된 img_B
            "mos": torch.tensor(self.mos[index], dtype=torch.float32)  # MOS 값
        }


    def __len__(self):
        return len(self.images)

# 데이터셋 분리 함수
def split_dataset(dataset, train_ratio=0.7, val_ratio=0.2, test_ratio=0.1, seed=42):
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "비율의 합이 1이 되어야 합니다."
    dataset_length = len(dataset)
    train_len = int(dataset_length * train_ratio)
    val_len = int(dataset_length * val_ratio)
    test_len = dataset_length - train_len - val_len

    generator = torch.Generator().manual_seed(seed)
    return random_split(dataset, [train_len, val_len, test_len], generator=generator)

# 메인 코드 실행
if __name__ == "__main__":
    dataset_path = "E:/ARNIQA/ARNIQA/dataset/KADID10K"
    
    # 데이터셋 로드
    full_dataset = KADID10KDataset(root=dataset_path)

    # 데이터셋 분리
    train_dataset, val_dataset, test_dataset = split_dataset(full_dataset, train_ratio=0.7, val_ratio=0.2, test_ratio=0.1)

    print(f"훈련 데이터셋 크기: {len(train_dataset)}")
    print(f"검증 데이터셋 크기: {len(val_dataset)}")
    print(f"테스트 데이터셋 크기: {len(test_dataset)}")

    # DataLoader 생성
    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)

    # DataLoader 확인
    for batch in train_dataloader:
        print(f"훈련 배치 크기: {batch['img_A_orig'].shape}, {batch['img_B_orig'].shape}")  # [batch_size, num_crops, 3, crop_size, crop_size]
        break




## 시각화할 때 사용하는 코드
""" 

import re
import pandas as pd
import numpy as np
from pathlib import Path
from PIL import Image, ImageFilter
from torchvision import transforms
import torch
import random
import torchvision.transforms.functional as F

class KADID10KDataset(torch.utils.data.Dataset):
    def __init__(self, root: str, phase: str = "all", crop_size: int = 224, train_ratio=0.7, val_ratio=0.2, test_ratio=0.1, seed=42):
        self.crop_size = crop_size

        # CSV 파일 로드
        scores_csv = pd.read_csv(Path(root) / "kadid10k.csv")
        scores_csv = scores_csv[["dist_img", "ref_img", "dmos"]]

        # 이미지 파일 경로 생성
        self.images = np.array([Path(root) / "images" / img for img in scores_csv["dist_img"].values])
        self.ref_images = np.array([Path(root) / "images" / img for img in scores_csv["ref_img"].values])
        self.mos = np.array(scores_csv["dmos"].values.tolist())

        # 왜곡 정보 파싱
        self.distortion_types = []
        self.distortion_levels = []
        for img_path in self.images:
            match = re.search(r'I\d+_(\d+)_(\d+)\.png$', str(img_path))
            if match:
                self.distortion_types.append(int(match.group(1)))  # 왜곡 종류
                self.distortion_levels.append(int(match.group(2)))  # 왜곡 레벨
            else:
                self.distortion_types.append(-1)  # 오류 발생 시 기본값
                self.distortion_levels.append(-1)

        self.distortion_types = np.array(self.distortion_types)
        self.distortion_levels = np.array(self.distortion_levels)

        # 이미지 파일만 선택
        valid_extensions = [".png", ".jpg", ".jpeg", ".bmp", ".tiff"]
        valid_indices = [i for i, img_path in enumerate(self.images) if img_path.suffix.lower() in valid_extensions]

        self.images = self.images[valid_indices]
        self.ref_images = self.ref_images[valid_indices]
        self.mos = self.mos[valid_indices]
        self.distortion_types = self.distortion_types[valid_indices]
        self.distortion_levels = self.distortion_levels[valid_indices]

        # 데이터셋 분리
        dataset_length = len(self.images)
        train_len = int(dataset_length * train_ratio)
        val_len = int(dataset_length * val_ratio)
        test_len = dataset_length - train_len - val_len

        generator = torch.Generator().manual_seed(seed)
        indices = torch.randperm(dataset_length, generator=generator).tolist()

        self.train_indices = indices[:train_len]
        self.val_indices = indices[train_len:train_len + val_len]
        self.test_indices = indices[train_len + val_len:]

        # Phase 처리
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

        self.images = self.images[selected_indices]
        self.ref_images = self.ref_images[selected_indices]
        self.mos = self.mos[selected_indices]
        self.distortion_types = self.distortion_types[selected_indices]
        self.distortion_levels = self.distortion_levels[selected_indices]

    def transform(self, image: Image) -> torch.Tensor:

        return transforms.Compose([
            transforms.Resize((self.crop_size, self.crop_size)),
            transforms.ToTensor(),
        ])(image)

    def apply_distortion(self, image: Image) -> Image:

        if random.random() > 0.5:
            image = image.filter(ImageFilter.GaussianBlur(radius=2))
        return image  # Always return a PIL.Image object

    def __getitem__(self, index: int) -> dict:
        img_A_path = self.images[index]
        img_B_path = self.ref_images[index]

        # 이미지 로드 시 예외 처리 추가
        try:
            img_A_orig = Image.open(img_A_path).convert("RGB")
        except Exception as e:
            print(f"Error loading image {img_A_path}: {e}")
            raise

        try:
            img_B_orig = Image.open(img_B_path).convert("RGB")
        except Exception as e:
            print(f"Error loading image {img_B_path}: {e}")
            raise

        img_A_orig = self.transform(img_A_orig)
        img_B_orig = self.transform(img_B_orig)

        # Distortion 적용 및 크롭 생성
        crops_A = [img_A_orig] + [self.transform(self.apply_distortion(F.to_pil_image(img_A_orig))) for _ in range(3)]
        crops_B = [img_B_orig] + [self.transform(self.apply_distortion(F.to_pil_image(img_B_orig))) for _ in range(3)]

        # Stack crops
        img_A = torch.stack(crops_A)  # Shape: [num_crops, 3, crop_size, crop_size]
        img_B = torch.stack(crops_B)  # Shape: [num_crops, 3, crop_size, crop_size]

        # Reshape to [1, num_crops, 3, crop_size, crop_size]
        img_A = img_A.unsqueeze(0)
        img_B = img_B.unsqueeze(0)

        return {
            "img_A_orig": img_A,
            "img_B_orig": img_B,
            "mos": self.mos[index],
            "dist_group": self.distortion_types[index],  # 왜곡 종류 추가
            "dist_level": self.distortion_levels[index],  # 왜곡 레벨 추가
        }

    def __len__(self):
        return len(self.images)
 """