""" import pandas as pd
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from pathlib import Path
import re

class KADID10KDataset(Dataset):

    def __init__(self, root: str, phase: str = "all", crop_size: int = 224,
                 max_distortions: int = 4, num_levels: int = 5, pristine_prob: float = 0.05,
                 train_ratio=0.7, val_ratio=0.2, test_ratio=0.1, seed=42):
        self.patch_size = crop_size  # crop_size를 patch_size로 사용
        self.max_distortions = max_distortions
        self.num_levels = num_levels
        self.pristine_prob = pristine_prob
        self.is_synthetic = True 

        # Load CSV file
        scores_csv = pd.read_csv(Path(root) / "kadid10k.csv")
        scores_csv = scores_csv[["dist_img", "ref_img", "dmos"]]

        # 추론된 왜곡 유형 및 그룹 추가
        scores_csv['distortion_type'] = scores_csv['dist_img'].apply(self._infer_distortion_type)
        scores_csv['distortion_group'] = scores_csv['distortion_type'].apply(
            lambda dist_type: available_distortions.get(dist_type, "unknown")
        )

        # Load image paths and MOS values
        self.images = np.array([Path(root) / "images" / img for img in scores_csv["dist_img"].values])
        self.ref_images = np.array([Path(root) / "images" / img for img in scores_csv["ref_img"].values])
        self.mos = np.array(scores_csv["dmos"].values.tolist())
        self.distortion_types = np.array(scores_csv["distortion_type"].values.tolist())
        self.distortion_groups = np.array(scores_csv["distortion_group"].values.tolist())

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

        if phase == "train":
            selected_indices = self.train_indices
        elif phase == "val":
            selected_indices = self.val_indices
        elif phase == "test":
            selected_indices = self.test_indices
        else:
            selected_indices = indices

        self.images = self.images[selected_indices]
        self.ref_images = self.ref_images[selected_indices]
        self.mos = self.mos[selected_indices]
        self.distortion_types = self.distortion_types[selected_indices]
        self.distortion_groups = self.distortion_groups[selected_indices]

        # Transforms
        self.normalize_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.to_pil_transform = transforms.ToPILImage()
        self.resize_transform = transforms.Resize((crop_size // 2, crop_size // 2))  # crop_size를 사용

    def _infer_distortion_type(self, dist_img_name: str) -> str:
        match = re.search(r'_(\d{2})_', dist_img_name)
        if match:
            distortion_id = int(match.group(1))
            return distortion_types_mapping.get(distortion_id, "unknown")
        return "unknown"

    def normalize(self, img: Image):
        return self.normalize_transform(img)

    def __getitem__(self, index):
        # Load images
        img_A_orig = Image.open(self.images[index]).convert("RGB")
        img_B_orig = Image.open(self.ref_images[index]).convert("RGB")

        # Normalize original images
        img_A_orig = self.normalize(img_A_orig)  # Shape: [C, H, W]
        img_B_orig = self.normalize(img_B_orig)  # Shape: [C, H, W]

        # Downsample images
        img_A_ds = self.resize_transform(self.to_pil_transform(img_A_orig))
        img_B_ds = self.resize_transform(self.to_pil_transform(img_B_orig))

        # Normalize downsampled images
        img_A_ds = self.normalize(img_A_ds)  # Shape: [c, h, w]
        img_B_ds = self.normalize(img_B_ds)  # Shape: [c, h, w]

        # Ensure the batch dimension is added
        img_A_orig = img_A_orig.unsqueeze(0)  # Shape: [1, C, H, W]
        img_B_orig = img_B_orig.unsqueeze(0)  # Shape: [1, C, H, W]

        return {
            "img": img_A_orig,  # 'img' 키 추가
            "img_ds": img_A_ds,  # 다운샘플링된 이미지
            "img_A_orig": img_A_orig,
            "img_B_orig": img_B_orig,
            "img_A_ds": img_A_ds,
            "img_B_ds": img_B_ds,
            "mos": self.mos[index],
            "distortion_type": self.distortion_types[index],
            "distortion_group": self.distortion_groups[index]
        }

    def __len__(self):
        return len(self.images)
    
    
def get_split_indices(self, split: int, phase: str) -> np.ndarray:

    if phase == "train":
        return np.array(self.train_indices)
    elif phase == "val":
        return np.array(self.val_indices)
    elif phase == "test":
        return np.array(self.test_indices)
    else:
        raise ValueError(f"Invalid phase: {phase}. Must be 'train', 'val', or 'test'.")



# Distortion type mappings
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

# Available distortions by type
available_distortions = {
    "gaussian_blur": "blur",
    "lens_blur": "blur",
    "motion_blur": "blur",
    "color_diffusion": "color_distortion",
    "color_shift": "color_distortion",
    "color_quantization": "color_distortion",
    "color_saturation_1": "color_distortion",
    "color_saturation_2": "color_distortion",
    "jpeg2000": "jpeg",
    "jpeg": "jpeg",
    "white_noise": "noise",
    "white_noise_color_component": "noise",
    "impulse_noise": "noise",
    "multiplicative_noise": "noise",
    "denoise": "noise",
    "brighten": "brightness_change",
    "darken": "brightness_change",
    "mean_shift": "brightness_change",
    "jitter": "spatial_distortion",
    "non_eccentricity_patch": "spatial_distortion",
    "pixelate": "spatial_distortion",
    "quantization": "spatial_distortion",
    "color_block": "spatial_distortion",
    "high_sharpen": "sharpness_contrast",
    "contrast_change": "sharpness_contrast"
}

distortion_groups = {
    "blur": ["gaussian_blur", "lens_blur", "motion_blur"],
    "color_distortion": ["color_diffusion", "color_shift", "color_quantization", "color_saturation_1", "color_saturation_2"],
    "jpeg": ["jpeg2000", "jpeg"],
    "noise": ["white_noise", "white_noise_color_component", "impulse_noise", "multiplicative_noise", "denoise"],
    "brightness_change": ["brighten", "darken", "mean_shift"],
    "spatial_distortion": ["jitter", "non_eccentricity_patch", "pixelate", "quantization", "color_block"],
    "sharpness_contrast": ["high_sharpen", "contrast_change"]
}
 """

import pandas as pd
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from pathlib import Path
import re

class KADID10KDataset(Dataset):
    def __init__(self, root: str, phase: str = "all", crop_size: int = 224,
                 max_distortions: int = 4, num_levels: int = 5, pristine_prob: float = 0.05,
                 train_ratio=0.7, val_ratio=0.2, test_ratio=0.1, seed=42):
        self.patch_size = crop_size  # crop_size를 patch_size로 사용
        self.max_distortions = max_distortions
        self.num_levels = num_levels
        self.pristine_prob = pristine_prob
        self.is_synthetic = True 

        # Load CSV file
        scores_csv = pd.read_csv(Path(root) / "kadid10k.csv")
        scores_csv = scores_csv[["dist_img", "ref_img", "dmos"]]

        # 추론된 왜곡 유형 및 그룹 추가
        scores_csv['distortion_type'] = scores_csv['dist_img'].apply(self._infer_distortion_type)
        scores_csv['distortion_group'] = scores_csv['distortion_type'].apply(
            lambda dist_type: available_distortions.get(dist_type, "unknown")
        )

        # Load image paths and MOS values
        self.images = np.array([Path(root) / "images" / img for img in scores_csv["dist_img"].values])
        self.ref_images = np.array([Path(root) / "images" / img for img in scores_csv["ref_img"].values])
        self.mos = np.array(scores_csv["dmos"].values.tolist())
        self.distortion_types = np.array(scores_csv["distortion_type"].values.tolist())
        self.distortion_groups = np.array(scores_csv["distortion_group"].values.tolist())

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

        if phase == "train":
            selected_indices = self.train_indices
        elif phase == "val":
            selected_indices = self.val_indices
        elif phase == "test":
            selected_indices = self.test_indices
        else:
            selected_indices = indices

        self.images = self.images[selected_indices]
        self.ref_images = self.ref_images[selected_indices]
        self.mos = self.mos[selected_indices]
        self.distortion_types = self.distortion_types[selected_indices]
        self.distortion_groups = self.distortion_groups[selected_indices]

        # Transforms
        self.normalize_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.to_pil_transform = transforms.ToPILImage()
        self.resize_transform = transforms.Resize((crop_size // 2, crop_size // 2))  # crop_size를 사용

    def _infer_distortion_type(self, dist_img_name: str) -> str:
        """Extract distortion type from dist_img file name."""
        match = re.search(r'_(\d{2})_', dist_img_name)
        if match:
            distortion_id = int(match.group(1))
            return distortion_types_mapping.get(distortion_id, "unknown")
        return "unknown"

    def normalize(self, img: Image):
        """Normalize the input image."""
        return self.normalize_transform(img)

    #def __getitem__(self, index):
    #    # Load images
    #    img_A_orig = Image.open(self.images[index]).convert("RGB")
    #    img_B_orig = Image.open(self.ref_images[index]).convert("RGB")
#
    #    # Normalize original images
    #    img_A_orig = self.normalize(img_A_orig)  # Shape: [C, H, W]
    #    img_B_orig = self.normalize(img_B_orig)  # Shape: [C, H, W]
#
    #    # Downsample images
    #    img_A_ds = self.resize_transform(self.to_pil_transform(img_A_orig))
    #    img_B_ds = self.resize_transform(self.to_pil_transform(img_B_orig))
#
    #    # Normalize downsampled images
    #    img_A_ds = self.normalize(img_A_ds)  # Shape: [C, H, W]
    #    img_B_ds = self.normalize(img_B_ds)  # Shape: [C, H, W]
#
    #    # Ensure batch dimension is added
    #    img_A_orig = img_A_orig.unsqueeze(0)  # Shape: [1, C, H, W]
    #    img_B_orig = img_B_orig.unsqueeze(0)  # Shape: [1, C, H, W]
#
    #    # Debugging shapes
    #    print(f"img_A_orig shape (after unsqueeze): {img_A_orig.shape}")  # Expected: [1, 3, H, W]
    #    print(f"img_A_ds shape: {img_A_ds.shape}")  # Expected: [3, h, w]
#
    #    return {
    #        "img": img_A_orig,  # 'img' key 추가
    #        "img_ds": img_A_ds,  # Downsampled image
    #        "img_A_orig": img_A_orig,
    #        "img_B_orig": img_B_orig,
    #        "img_A_ds": img_A_ds,
    #        "img_B_ds": img_B_ds,
    #        "mos": self.mos[index],
    #        "distortion_type": self.distortion_types[index],
    #        "distortion_group": self.distortion_groups[index]
    #    }
    

    # 224 리사이즈

    def __getitem__(self, index):
        # Load images
        img_A_orig = Image.open(self.images[index]).convert("RGB")
        img_B_orig = Image.open(self.ref_images[index]).convert("RGB")
        # Resize original images to 224x224
        resize_to_patch = transforms.Resize((self.patch_size, self.patch_size))
        img_A_orig = resize_to_patch(img_A_orig)
        img_B_orig = resize_to_patch(img_B_orig)
        # Normalize original images
        img_A_orig = self.normalize(img_A_orig)  # Shape: [C, 224, 224]
        img_B_orig = self.normalize(img_B_orig)  # Shape: [C, 224, 224]
        # Downsample images
        img_A_ds = self.resize_transform(self.to_pil_transform(img_A_orig))
        img_B_ds = self.resize_transform(self.to_pil_transform(img_B_orig))
        # Normalize downsampled images
        img_A_ds = self.normalize(img_A_ds)  # Shape: [C, 112, 112]
        img_B_ds = self.normalize(img_B_ds)  # Shape: [C, 112, 112]
        # Ensure the batch dimension is added
        img_A_orig = img_A_orig.unsqueeze(0)  # Shape: [1, C, 224, 224]
        img_B_orig = img_B_orig.unsqueeze(0)  # Shape: [1, C, 224, 224]
        return {
            "img": img_A_orig,  # 'img' 키 추가
            "img_ds": img_A_ds,  # 다운샘플링된 이미지
            "img_A_orig": img_A_orig,
            "img_B_orig": img_B_orig,
            "img_A_ds": img_A_ds,
            "img_B_ds": img_B_ds,
            "mos": self.mos[index],
            "distortion_type": self.distortion_types[index],
            "distortion_group": self.distortion_groups[index]
        }



    def __len__(self):
        return len(self.images)

# Distortion type mappings
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

# Available distortions by type
available_distortions = {
    "gaussian_blur": "blur",
    "lens_blur": "blur",
    "motion_blur": "blur",
    "color_diffusion": "color_distortion",
    "color_shift": "color_distortion",
    "color_quantization": "color_distortion",
    "color_saturation_1": "color_distortion",
    "color_saturation_2": "color_distortion",
    "jpeg2000": "jpeg",
    "jpeg": "jpeg",
    "white_noise": "noise",
    "white_noise_color_component": "noise",
    "impulse_noise": "noise",
    "multiplicative_noise": "noise",
    "denoise": "noise",
    "brighten": "brightness_change",
    "darken": "brightness_change",
    "mean_shift": "brightness_change",
    "jitter": "spatial_distortion",
    "non_eccentricity_patch": "spatial_distortion",
    "pixelate": "spatial_distortion",
    "quantization": "spatial_distortion",
    "color_block": "spatial_distortion",
    "high_sharpen": "sharpness_contrast",
    "contrast_change": "sharpness_contrast"
}
