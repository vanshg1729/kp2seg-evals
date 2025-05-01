import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
from torchvision.transforms import functional as TF
import pickle
from PIL import Image
from PIL.ImageOps import exif_transpose

from src.utils.mask_rle_utils import coco_rle_to_masks


# Normalize to [-1, 1] range
ImgNorm = T.Compose([T.ToTensor(), T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


def resize_longest_side(pil_img, target=512):
    w, h = pil_img.size
    scale = target / max(w, h)
    new_size = (int(w * scale), int(h * scale))
    return pil_img.resize(new_size, Image.BILINEAR)


def resize_masks(masks: torch.Tensor, size: tuple) -> torch.Tensor:
    """Resize all masks (M, H, W) to (M, H_new, W_new) using nearest neighbor"""
    return torch.stack(
        [
            TF.resize(
                mask.unsqueeze(0), size, interpolation=TF.InterpolationMode.NEAREST
            ).squeeze(0)
            for mask in masks
        ]
    )


class PairedImageMaskDataset(Dataset):
    def __init__(self, img_paths0, img_paths1, mask_paths0, mask_paths1, target=512):
        assert (
            len(img_paths0) == len(img_paths1) == len(mask_paths0) == len(mask_paths1)
        )
        self.img_paths0 = img_paths0
        self.img_paths1 = img_paths1
        self.mask_paths0 = mask_paths0
        self.mask_paths1 = mask_paths1
        self.target = target

    def __len__(self):
        return len(self.img_paths0)

    def _load_masks(self, path):
        with open(path, "rb") as f:
            rles = pickle.load(f)["mask_coco_rles_resized"]

        masks = coco_rle_to_masks(rles)  # (M, H, W)
        # resized = resize_masks(masks, size_hw)  # (M, H_resize, W_resize)
        return masks.to(torch.uint8)

    def __getitem__(self, idx):
        masks0 = self._load_masks(self.mask_paths0[idx])
        masks1 = self._load_masks(self.mask_paths1[idx])

        return {"masks0": masks0, "masks1": masks1, "img0_path": str(self.img_paths0[idx]), "img1_path": str(self.img_paths1[idx])}