import matplotlib.pyplot as plt
import cv2
import numpy as np
import colorsys
import torch


def load_image(image_path):
    print(f"Loading image from {image_path}")
    image = cv2.imread(str(image_path))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


def plot_images(image0, image1):
    # permute to HWC
    if image0.ndim == 4:
        image0 = image0[0].permute(1, 2, 0).cpu().numpy()
    if image1.ndim == 4:
        image1 = image1[0].permute(1, 2, 0).cpu().numpy()
    # convert to uint8
    if image0.dtype != np.uint8:
        image0 = (image0 * 255).astype(np.uint8)
    if image1.dtype != np.uint8:
        image1 = (image1 * 255).astype(np.uint8)

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(image0)
    plt.title("Image 0")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(image1)
    plt.title("Image 1")
    plt.axis("off")

    plt.show()


def plot_masks(image0, image1, masks0, masks1):
    imgwmasks0 = superimpose_masks(image0, masks0)
    imgwmasks1 = superimpose_masks(image1, masks1)

    # show masks
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(imgwmasks0)
    plt.title("Image 0 with Masks")
    plt.axis("off")
    plt.subplot(1, 2, 2)
    plt.imshow(imgwmasks1)
    plt.title("Image 1 with Masks")
    plt.axis("off")
    plt.show()


def superimpose_masks(base_img, masks):
    """
    Plot colored masks with transparency on top of the base image.

    Args:
        base_img (np.ndarray): Base image in RGB format, shape (H, W, 3), dtype uint8.
        masks (torch.Tensor or dict): A dictionary or tensor where keys are object IDs (1-based)
                                      and values are binary masks (H, W).

    Returns:
        np.ndarray: The resulting image with masks overlaid.
    """
    if base_img.ndim != 3 or base_img.shape[2] != 3:
        raise ValueError("base_img must be an RGB image of shape (H, W, 3).")

    H, W, _ = base_img.shape
    vis_img = base_img.astype(np.float32)

    if isinstance(masks, torch.Tensor):
        masks = masks.cpu().numpy()  # Convert PyTorch tensor to NumPy array

    # Handle both dictionary and array-like input for masks
    if isinstance(masks, dict):
        mask_colors = _generate_distinct_colors(len(masks))
        for obj_id, mask in masks.items():
            if isinstance(mask, torch.Tensor):
                mask = mask.cpu().numpy()
            if mask.shape != (H, W):
                raise ValueError(
                    f"Mask for object {obj_id} must have shape ({H}, {W})."
                )
            mask = mask.astype(bool)
            color = np.array(mask_colors[obj_id - 1], dtype=np.float32) / 255.0
            vis_img = _apply_mask(vis_img, mask, color)
    else:
        N, mask_H, mask_W = masks.shape
        if (mask_H, mask_W) != (H, W):
            raise ValueError(
                "All masks must have the same spatial dimensions as the base image."
            )
        mask_colors = _generate_distinct_colors(N)
        for i in range(N):
            mask = masks[i].astype(bool)
            color = np.array(mask_colors[i], dtype=np.float32) / 255.0
            vis_img = _apply_mask(vis_img, mask, color)

    return np.clip(vis_img, 0, 255).astype(np.uint8)


def _generate_distinct_colors(num_colors):
    """
    Generate a list of visually distinct and vibrant colors using HSV color space.

    Parameters:
    - num_colors: Number of colors to generate

    Returns:
    - List of RGB color tuples, each in the range [0, 255]
    """
    colors = []
    for i in range(num_colors):
        hue = (
            i * 0.618033988749895
        ) % 1.0  # Golden ratio method for uniform color spread
        saturation = 0.8
        value = 0.9
        rgb = colorsys.hsv_to_rgb(hue, saturation, value)
        color = tuple(int(x * 255) for x in rgb)
        colors.append(color)
    return colors


def _apply_mask(vis_img, mask, color, alpha=0.5):
    """Apply a single mask with color and transparency to the image."""
    color_mask = np.zeros_like(vis_img, dtype=np.float32)
    color_mask[mask] = color
    vis_img[mask] = vis_img[mask] * (1 - alpha) + color_mask[mask] * 255 * alpha
    return vis_img