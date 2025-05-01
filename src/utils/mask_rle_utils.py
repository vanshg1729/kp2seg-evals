from typing import Any, Dict, List, Tuple, Optional
import numpy as np
import torch
import torch.nn.functional as F
from pycocotools import mask as mask_utils  # type: ignore

# Heavily adapted from https://github.com/facebookresearch/sam2/blob/main/sam2/utils/amg.py
# Use if you want major space savings when working with binary masks ;)

def mask_to_rle_pytorch(tensor: torch.Tensor) -> List[Dict[str, Any]]:
    """Encodes masks to an uncompressed RLE, in the format expected by pycocotools."""
    # Put in fortran order and flatten h,w
    b, h, w = tensor.shape
    tensor = tensor.permute(0, 2, 1).flatten(1)

    # Compute change indices
    diff = tensor[:, 1:] ^ tensor[:, :-1]
    change_indices = diff.nonzero()

    # Encode run length
    out = []
    for i in range(b):
        cur_idxs = change_indices[change_indices[:, 0] == i, 1]
        cur_idxs = torch.cat(
            [
                torch.tensor([0], dtype=cur_idxs.dtype, device=cur_idxs.device),
                cur_idxs + 1,
                torch.tensor([h * w], dtype=cur_idxs.dtype, device=cur_idxs.device),
            ]
        )
        btw_idxs = cur_idxs[1:] - cur_idxs[:-1]
        counts = [] if tensor[i, 0] == 0 else [0]
        counts.extend(btw_idxs.detach().cpu().tolist())
        out.append({"size": [h, w], "counts": counts})
    return out


def rle_to_mask(rle: Dict[str, Any]) -> np.ndarray:
    """Compute a binary mask from an uncompressed RLE."""
    h, w = rle["size"]
    mask = np.empty(h * w, dtype=bool)
    idx = 0
    parity = False
    for count in rle["counts"]:
        mask[idx : idx + count] = parity
        idx += count
        parity ^= True
    mask = mask.reshape(w, h)
    return mask.transpose()  # Put in C order


def coco_encode_rle(uncompressed_rle: Dict[str, Any]) -> Dict[str, Any]:
    """Encodes an uncompressed RLE to the COCO format."""
    h, w = uncompressed_rle["size"]
    rle = mask_utils.frPyObjects(uncompressed_rle, h, w)
    rle["counts"] = rle["counts"].decode("utf-8")  # Necessary to serialize with json
    return rle


def area_from_rle(rle: Dict[str, Any]) -> int:
    return sum(rle["counts"][1::2])


# 👇👇 Main functions for converting masks to COCO RLE format and back
def masks_to_coco_rle(masks: torch.Tensor) -> List[Dict[str, Any]]:
    """
    Converts PyTorch masks to COCO RLE format
    Args:
    masks (torch.Tensor): Binary masks of shape BxHxW
    """

    uncompressed_rles = mask_to_rle_pytorch(masks)
    coco_rles = [coco_encode_rle(rle) for rle in uncompressed_rles]

    return coco_rles


def coco_rle_to_masks(
    coco_rles: List[Dict[str, Any]],
) -> torch.Tensor:
    """
    Load COCO RLE encoded masks from a pickle file and convert back to PyTorch tensor.
    Args:
    coco_rles (List[Dict[str, Any]]): COCO RLE encoded masks
    Returns:
    torch.Tensor: Binary masks of shape BxHxW
    """
    masks = []

    for rle in coco_rles:
        rle["counts"] = rle["counts"].encode("utf-8")  # Convert back to bytes
        mask = mask_utils.decode(rle)
        masks.append(torch.from_numpy(mask))

    return torch.stack(masks)