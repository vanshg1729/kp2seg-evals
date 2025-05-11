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

eval_replica_config = "configs/config_eval_replica_ada.yaml"
cfg.merge_from_file(eval_replica_config)

from src.datasets.replica_hard_iou_eval_interface import (
    ReplicaHardIoUDataset,
)
from torch.utils.data import DataLoader
from matching import get_matcher

from src.utils.vpr_metrics import (
    createPR,
    recallAt100precision,
    recallAtK,
    meanReciprocalRank,
)


def compute_sample_metrics(pred_mat, gt_mat):
    return {
        "auprc": createPR(pred_mat, gt_mat)[2],
        "r_at_1": recallAtK(pred_mat, gt_mat, K=1),
        "r_at_5": recallAtK(pred_mat, gt_mat, K=5),
        "r_at_10": recallAtK(pred_mat, gt_mat, K=10),
        "r_at_100p": recallAt100precision(pred_mat, gt_mat),
        "mrr": meanReciprocalRank(pred_mat, gt_mat),
    }


def compute_pred_matrix(matched_kpts0, matched_kpts1, masks0, masks1):
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

    pred_mat = torch.zeros((M, N), dtype=torch.int32)
    for i, j in zip(src_idx.tolist(), tgt_idx.tolist()):
        pred_mat[i, j] += 1
    return pred_mat


def evaluate_all(loader, matcher, output_json_path):
    results = defaultdict(dict)
    print(f"{output_json_path = }")

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

        pred_mat = compute_pred_matrix(mkpts0, mkpts1, masks0, masks1)
        pred_mat_cpu = pred_mat.detach().cpu().numpy()
        gt_assignment_cpu = gt_assignment.detach().cpu().numpy()

        sample_metrics_dict = compute_sample_metrics(pred_mat_cpu, gt_assignment_cpu)

        results[scene_id][key] = {k: v for k, v in sample_metrics_dict.items()}

    os.makedirs(os.path.dirname(output_json_path), exist_ok=True)
    with open(output_json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {output_json_path}")
    return results


if __name__ == "__main__":
    MATCHER = "mast3r"
    print(f"Evaluating with matcher: {MATCHER}")

    ANGLES = [0, 45, 90, 135, 180]
    # ANGLES = [0, 45, 90]
    # ANGLES = [90, 135, 180]
    for i in range(len(ANGLES) - 1):
        START_ANGLE = ANGLES[i]
        END_ANGLE = ANGLES[i + 1]
        print(f"Evaluating {START_ANGLE} to {END_ANGLE} degrees")

        subset_split_file = cfg['DATASET']['PAIRS_PATH']
        with open(subset_split_file, "r") as f:
            subset_split = json.load(f)

        # build pairs by looking at X-Y split under each scene, ignore if not present
        ref_query_pairs = []

        for scene_id, scene_data in subset_split.items():
            if f"{START_ANGLE}-{END_ANGLE}" in scene_data:
                for pair_str in scene_data[f"{START_ANGLE}-{END_ANGLE}"]:
                    ref, query = pair_str.split("-")
                    ref_query_pairs.append((scene_id, ref, query))
        
        # ref_query_pairs = ref_query_pairs[:10] # for testing that the whole pipeline works

        # # val_selected_scenes = ["394a542a19", "9f79564dbf", "e8e81396b6"]
        # val_dataset = ScanNetPPResizedHardIoUDataset(cfg, val_selected_scenes, None)

        val_dataset = ReplicaHardIoUDataset(cfg, None, ref_query_pairs)

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
            output_json_path=f"{cfg['SAVE_DIR']}/{MATCHER}_val_8_{START_ANGLE}_{END_ANGLE}.json",
        )

print("Done evaluating all pairs.")