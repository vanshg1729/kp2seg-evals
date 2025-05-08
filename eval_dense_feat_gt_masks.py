import os
import torch

torch.hub.set_dir("/scratch/toponav/indoor-topo-loc/checkpoints/torchhub")
print(torch.hub.get_dir())

import numpy as np
import json
from collections import defaultdict
from tqdm import tqdm

from configs.default import cfg
import sys

# HACK: To get around broken editable pip install
sys.path.insert(0, "/home2/rjayanti/workdirs/rjayanti/image-matching-models")

eval_spp_config = "configs/config_eval_spp_resz.yaml"
cfg.merge_from_file(eval_spp_config)

from src.datasets.scannetpp_resz_hard_iou_eval_interface import (
    ScanNetPPResizedHardIoUDataset,
)
from torch.utils.data import DataLoader
from matching import get_matcher


def compute_vote_matrix(matched_kpts0, matched_kpts1, masks0, masks1):
    matched_kpts0 = (
        torch.from_numpy(matched_kpts0)
        if isinstance(matched_kpts0, np.ndarray)
        else matched_kpts0
    )
    matched_kpts1 = (
        torch.from_numpy(matched_kpts1)
        if isinstance(matched_kpts1, np.ndarray)
        else matched_kpts1
    )

    K = matched_kpts0.shape[0]
    M, H, W = masks0.shape
    N = masks1.shape[0]

    x0 = matched_kpts0[:, 0].long().clamp(0, W - 1)
    y0 = matched_kpts0[:, 1].long().clamp(0, H - 1)
    x1 = matched_kpts1[:, 0].long().clamp(0, W - 1)
    y1 = matched_kpts1[:, 1].long().clamp(0, H - 1)

    src_mask_ids = masks0[:, y0, x0].T  # (K, M)
    tgt_mask_ids = masks1[:, y1, x1].T  # (K, N)

    valid = src_mask_ids.any(dim=1) & tgt_mask_ids.any(dim=1)
    src_idx = src_mask_ids[valid].int().argmax(dim=1)
    tgt_idx = tgt_mask_ids[valid].int().argmax(dim=1)

    votes = torch.zeros((M, N), dtype=torch.int32)
    for i, j in zip(src_idx.tolist(), tgt_idx.tolist()):
        votes[i, j] += 1
    return votes


def get_pred_assignment(votes):
    pred = votes.argmax(dim=1)
    pred[votes.sum(dim=1) == 0] = -1
    return pred


def compute_iou(pred_assignment, gt_assignment):
    M, N = gt_assignment.shape
    gt_idx = gt_assignment.argmax(dim=1)
    has_gt = gt_assignment.sum(dim=1) > 0  # Only consider source segments with a match

    correct = 0
    total = 0

    for i in range(M):
        if not has_gt[i]:
            continue  # Ignore unmatched GT
        total += 1
        if pred_assignment[i] == gt_idx[i]:
            correct += 1

    return correct / total if total > 0 else 0.0


def evaluate_all(loader, matcher, output_json_path):
    results = defaultdict(dict)

    for batch in tqdm(loader, desc="Evaluating"):
        scene_id = batch["scene_id"][0]
        frame0 = batch["img0_name"][0]
        frame1 = batch["img1_name"][0]
        key = f"{frame0}_{frame1}"

        img0 = matcher.load_image(str(batch["img0_path"][0]))
        img1 = matcher.load_image(str(batch["img1_path"][0]))
        result = matcher(img0, img1)

        mkpts0 = result["matched_kpts0"]
        mkpts1 = result["matched_kpts1"]
        masks0 = batch["masks_gt_0"][0].bool()
        masks1 = batch["masks_gt_1"][0].bool()
        gt_assignment = torch.diag_embed(batch["seg_corr_list_common"][0])

        votes = compute_vote_matrix(mkpts0, mkpts1, masks0, masks1)
        pred_assignment = get_pred_assignment(votes)
        avg_iou = compute_iou(pred_assignment, gt_assignment)

        results[scene_id][key] = avg_iou

    with open(output_json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {output_json_path}")
    return results


if __name__ == "__main__":
    MATCHER = "mast3r"

    ANGLES = [0, 45, 90, 135, 180]
    for i in range(len(ANGLES) - 1):
        START_ANGLE = ANGLES[i]
        END_ANGLE = ANGLES[i + 1]
        print(f"Evaluating {START_ANGLE} to {END_ANGLE} degrees")

        subset_split_file = "results/pose_bins_subset_val36.json"
        with open(subset_split_file, "r") as f:
            subset_split = json.load(f)

        # build pairs by looking at X-Y split under each scene, ignore if not present
        ref_query_pairs = []

        for scene_id, scene_data in subset_split.items():
            if f"{START_ANGLE}-{END_ANGLE}" in scene_data:
                for pair_str in scene_data[f"{START_ANGLE}-{END_ANGLE}"]:
                    ref, query = pair_str.split("_")
                    ref_query_pairs.append((scene_id, ref, query))

        # # val_selected_scenes = ["394a542a19", "9f79564dbf", "e8e81396b6"]
        # val_dataset = ScanNetPPResizedHardIoUDataset(cfg, val_selected_scenes, None)

        val_dataset = ScanNetPPResizedHardIoUDataset(cfg, None, ref_query_pairs)

        # NOTE: Currently dataset only supports batch size of 1 - to allow for variable number of masks
        val_loader = DataLoader(
            val_dataset,
            batch_size=1,
            num_workers=cfg.TRAINING.NUM_WORKERS,
            drop_last=False,  # Necessary for evaluation
            prefetch_factor=cfg.TRAINING.PREFETCH_FACTOR,
        )
        print(f"Length of Val DataLoader: {len(val_loader) = }")

        device = "cuda"
        matcher = get_matcher(
            MATCHER, device=device, max_num_keypoints=1024
        )  # can change to 4096

        results = evaluate_all(
            val_loader,
            matcher,
            output_json_path=f"results/{MATCHER}_val_36_{START_ANGLE}_{END_ANGLE}.json",
        )

print("Done evaluating all pairs.")
