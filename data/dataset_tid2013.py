import pandas as pd
import numpy as np
import re
from PIL import Image, ImageFilter
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from pathlib import Path
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

# 왜곡 그룹 정의
distortion_groups = {
    'blur': [1, 2, 3],
    'color': [4, 5, 6, 7, 8],
    'compression': [9, 10],
    'noise': [11, 12, 13, 14],
    'enhancement': [15, 16, 17],
    'shift': [18, 19],
    'others': [20, 21, 22, 23, 24, 25]
}

# 각 왜곡에 대한 강도 레벨 정의
distortion_levels = {
    'gaussian_blur': [0.5, 1.0, 1.5, 2.0, 2.5],
    'lens_blur': [1, 2, 3, 4, 5],
    'motion_blur': [1, 2, 3, 4, 5],
    'color_diffusion': [0.05, 0.1, 0.2, 0.3, 0.4],
    'color_shift': [-30, -20, -10, 10, 20],
    'color_quantization': [8, 16, 32, 64, 128],
    'color_saturation_1': [0.5, 0.6, 0.7, 0.8, 1.0],
    'color_saturation_2': [0.5, 0.6, 0.7, 0.8, 1.0],
    'jpeg2000': [0.1, 0.2, 0.3, 0.4, 0.5],
    'jpeg': [0.1, 0.2, 0.3, 0.4, 0.5],
    'white_noise': [5, 10, 15, 20, 25],
    'impulse_noise': [0.05, 0.1, 0.2, 0.3, 0.4],
    'multiplicative_noise': [0.1, 0.2, 0.3, 0.4, 0.5],
    'denoise': [0.5, 0.6, 0.7, 0.8, 1.0],
    'brighten': [10, 20, 30, 40, 50],
    'darken': [10, 20, 30, 40, 50],
    'mean_shift': [5, 10, 15, 20, 25],
    'jitter': [5, 10, 15, 20, 25],
    'non_eccentricity_patch': [0.5, 1.0, 1.5, 2.0, 2.5],
    'pixelate': [5, 10, 15, 20, 25],
    'quantization': [2, 4, 8, 16, 32],
    'color_block': [10, 20, 30, 40, 50],
    'high_sharpen': [1, 2, 3, 4, 5],
    'contrast_change': [0.5, 0.6, 0.7, 0.8, 1.0]
}

class TID2013Dataset(Dataset):
    def __init__(self, root: str, phase: str = "train", split_idx: int = 0, crop_size: int = 224):
        super().__init__()
        self.root = Path(root).parent
        self.phase = phase
        self.crop_size = crop_size

        # MOS 값 읽기
        csv_path = Path(root)
        if not csv_path.exists():
            raise FileNotFoundError(f"CSV 파일이 존재하지 않습니다: {csv_path}")

        scores_csv = pd.read_csv(csv_path)
        self.images = scores_csv["image_id"].values
        self.mos = scores_csv["mean"].values

        # 이미지 경로 설정
        self.image_paths = [self.root / "distorted_images" / img for img in self.images]
        self.reference_paths = [self.root / "reference_images" / f"{img.split('_')[0]}.BMP" for img in self.images]

        # Split 설정
        split_file_path = self.root / "splits" / f"{phase}.npy"
        if not split_file_path.exists():
            raise FileNotFoundError(f"Split 파일이 존재하지 않습니다: {split_file_path}")
        split_idxs = np.load(split_file_path)[split_idx]
        split_idxs = np.array([idx for idx in split_idxs if idx != -1])  # 유효한 인덱스만 필터링
        self.image_paths = np.array(self.image_paths)[split_idxs]
        self.reference_paths = np.array(self.reference_paths)[split_idxs]
        self.mos = np.array(self.mos)[split_idxs]

    def transform(self, image: Image) -> torch.Tensor:
        return transforms.Compose([
            transforms.Resize((self.crop_size, self.crop_size)),
            transforms.ToTensor(),
        ])(image)

    def apply_random_distortions(self, image, num_distortions=4):
        """이미지에 랜덤하게 여러 왜곡 적용."""
        selected_distortions = random.sample(list(distortion_levels.keys()), num_distortions)
        for distortion in selected_distortions:
            level = random.choice(distortion_levels[distortion])
            image = self.apply_distortion(image, distortion, level)
        return image



    def apply_distortion(self, image, distortion, level):
        """
        Apply a specific distortion to the given image.

        Args:
            image (PIL.Image): The input image to distort.
            distortion (str): The type of distortion to apply.
            level (float/int): The intensity level of the distortion.

        Returns:
            PIL.Image: The distorted image.
        """
        if distortion == "gaussian_blur":
            image = image.filter(ImageFilter.GaussianBlur(radius=level))
        elif distortion == "lens_blur":
            image = image.filter(ImageFilter.GaussianBlur(radius=level))
        elif distortion == "motion_blur":
            image = image.filter(ImageFilter.BoxBlur(level))
        elif distortion == "color_diffusion":
            # Add random uniform noise for color diffusion
            diffused = np.array(image, dtype=np.float64)  # Ensure the array is float64 for uniform addition
            diffused += np.random.uniform(-level, level, diffused.shape)
            diffused = np.clip(diffused, 0, 255).astype(np.uint8)  # Clip and convert back to uint8
            image = Image.fromarray(diffused)

        elif distortion == "color_saturation_2":
            hsv_image = np.array(image.convert("HSV"))
            hsv_image[..., 1] = np.clip(hsv_image[..., 1] * level, 0, 255)
            image = Image.fromarray(hsv_image.astype(np.uint8)).convert("RGB")
        elif distortion == "white_noise":
            noise = np.random.normal(0, level, (image.height, image.width, 3))
            noisy_image = np.array(image) + noise
            image = Image.fromarray(np.clip(noisy_image, 0, 255).astype(np.uint8))
        elif distortion == "impulse_noise":
            image = np.array(image)
            prob = level * 0.01
            for i in range(image.shape[0]):
                for j in range(image.shape[1]):
                    if random.random() < prob:
                        image[i][j] = np.random.choice([0, 255], size=3)
            image = Image.fromarray(image)
        elif distortion == "jpeg":
            image = image.resize((image.width // 2, image.height // 2))
        elif distortion == "jpeg2000":
            image = image.resize((image.width // 4, image.height // 4))
        elif distortion == "color_quantization":
            quantized = np.array(image) // level * level
            image = Image.fromarray(quantized.astype(np.uint8))
        elif distortion == "denoise":
            image = image.filter(ImageFilter.GaussianBlur(radius=level * 0.5))
        elif distortion == "brighten":
            level = max(level, 0)  # Ensure the level is non-negative
            enhancer = transforms.ColorJitter(brightness=level)
            image = enhancer(image)

        elif distortion == "darken":
            # Decrease brightness
            level = min(level, 1)  # Ensure the level does not exceed 1
            enhancer = transforms.ColorJitter(brightness=1 - level)
            image = enhancer(image)

        elif distortion == "mean_shift":
            shifted = np.array(image) + level
            image = Image.fromarray(np.clip(shifted, 0, 255).astype(np.uint8))
        elif distortion == "color_shift":
            # Shift the color channels randomly
            shifted = np.array(image, dtype=np.int32)
            if level > 0:  # Ensure level is positive
                shifted += np.random.randint(-level, level + 1, shifted.shape)  # Avoid low >= high
            image = Image.fromarray(np.clip(shifted, 0, 255).astype(np.uint8))

        elif distortion == "jitter":
            # Apply small random pixel jitter
            jittered = np.array(image, dtype=np.int32)
            if level > 0:  # Ensure level is positive
                jittered += np.random.randint(-level, level + 1, jittered.shape)  # Avoid low >= high
            image = Image.fromarray(np.clip(jittered, 0, 255).astype(np.uint8))

        elif distortion == "non_eccentricity_patch":
            image = image.filter(ImageFilter.GaussianBlur(radius=level))
        elif distortion == "color_block":
            # Apply color blocking by reducing resolution
            block_size = min(level, image.width, image.height)  # 제한 추가
            if block_size > 0:  # block_size가 유효한지 확인
                small = image.resize((image.width // block_size, image.height // block_size), Image.NEAREST)
                image = small.resize((image.width, image.height), Image.NEAREST)

        elif distortion == "pixelate":
            # Simulate pixelation by downsampling and upsampling
            block_size = min(level, image.width, image.height)  # 제한 추가
            if block_size > 0:  # block_size가 유효한지 확인
                small = image.resize((image.width // block_size, image.height // block_size), Image.NEAREST)
                image = small.resize((image.width, image.height), Image.NEAREST)

        elif distortion == "quantization":
            quantized = np.array(image) // level * level
            image = Image.fromarray(quantized.astype(np.uint8))
        
        elif distortion == "high_sharpen":
            image = image.filter(ImageFilter.UnsharpMask(radius=level, percent=150, threshold=3))
        elif distortion == "contrast_change":
            enhancer = transforms.ColorJitter(contrast=level)
            image = enhancer(image)
        return image




    def __getitem__(self, index: int):
        img_A_orig = Image.open(self.image_paths[index]).convert("RGB")
        img_B_ref = Image.open(self.reference_paths[index]).convert("RGB")

        # 랜덤 왜곡을 두 이미지에 적용
        img_A_distorted = self.apply_random_distortions(img_A_orig)
        img_B_distorted = self.apply_random_distortions(img_B_ref)

        # 이미지를 텐서로 변환
        img_A_orig = self.transform(img_A_orig)
        img_B_ref = self.transform(img_B_ref)
        img_A_distorted = self.transform(img_A_distorted)
        img_B_distorted = self.transform(img_B_distorted)

        return {
            "img_A": torch.stack([img_A_orig, img_A_distorted]),
            "img_B": torch.stack([img_B_ref, img_B_distorted]),
            "mos": self.mos[index]
        }

    def __len__(self):
        return len(self.image_paths)
