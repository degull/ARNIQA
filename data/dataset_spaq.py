

import pandas as pd
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

# 왜곡 그룹 정의 (7개 그룹)
distortion_groups = {
    'blur': [1, 2, 3],  # gaussian_blur, lens_blur, motion_blur
    'color': [4, 5, 6, 7, 8],  # color_diffusion, color_shift, color_quantization, color_saturation_1, color_saturation_2
    'compression': [9, 10],  # jpeg2000, jpeg
    'noise': [11, 12, 13, 14],  # white_noise, white_noise_color_component, impulse_noise, multiplicative_noise
    'enhancement': [15, 16, 17],  # denoise, brighten, darken
    'shift': [18, 19],  # mean_shift, jitter
    'others': [20, 21, 22, 23, 24, 25]  # non_eccentricity_patch, pixelate, quantization, color_block, high_sharpen, contrast_change
}

# 각 왜곡에 대한 강도 레벨 정의 (5개 강도)
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

import pandas as pd
import numpy as np
from PIL import Image, ImageFilter
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import random
from pathlib import Path


class SPAQDataset(Dataset):
    def __init__(self, csv_path: str, image_dir: str, crop_size: int = 224):
        super().__init__()
        self.csv_path = Path(csv_path)
        self.image_dir = Path(image_dir)
        self.crop_size = crop_size

        # CSV 데이터 읽기
        data = pd.read_csv(self.csv_path)
        self.image_paths = [self.image_dir / img_name for img_name in data["Image name"]]
        self.mos = data["MOS"].values

        # 이미지 경로 검증
        self.image_paths = [img for img in self.image_paths if img.exists()]
        if len(self.image_paths) == 0:
            raise FileNotFoundError(f"'{self.image_dir}'에서 유효한 이미지 파일을 찾을 수 없습니다.")

        # 이미지 전처리
        self.transform = transforms.Compose([
            transforms.Resize((self.crop_size, self.crop_size)),
            transforms.ToTensor(),
        ])

    def apply_distortion(self, image, distortion, level):

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

    def apply_random_distortions(self, image, num_distortions=4):
        distortions = random.sample(list(distortion_levels.keys()), num_distortions)
        for distortion in distortions:
            level = random.choice(distortion_levels[distortion])
            image = self.apply_distortion(image, distortion, level)
        return image

    def __getitem__(self, index: int):
        # 원본 이미지 로드 및 변환
        img_orig = Image.open(self.image_paths[index]).convert("RGB")
        img_orig = self.transform(img_orig)

        # 왜곡된 이미지 생성
        img_distorted = self.apply_random_distortions(img_orig)
        img_distorted = self.transform(img_distorted)

        # 디버깅: 이미지 텐서의 모양 출력
        print(f"img_orig shape: {img_orig.shape}, img_distorted shape: {img_distorted.shape}")

        return {
            "img_A": img_orig,
            "img_B": img_distorted,
            "mos": self.mos[index],
        }



    def __len__(self):
        return len(self.image_paths)
