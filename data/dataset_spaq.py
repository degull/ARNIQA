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

import pandas as pd
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from pathlib import Path
from utils.utils_data import resize_crop, center_corners_crop

class SPAQDataset(Dataset):
    """
    SPAQ IQA dataset with MOS in range [1, 100]. Resizes the images so that the smallest side is 512 pixels.

    Args:
        root (string): root directory of the dataset
        phase (string): indicates the phase of the dataset. Value must be in ['train', 'test', 'val', 'all']. Default is 'train'.
        split_idx (int): index of the split to use between [0, 9]. Used only if phase != 'all'. Default is 0.
        crop_size (int): size of each crop. Default is 224.

    Returns:
        dictionary with keys:
            img (Tensor): image
            mos (float): mean opinion score of the image (in range [1, 100])
    """
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
