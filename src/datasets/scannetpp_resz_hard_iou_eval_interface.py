import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
import os
import pickle
import numpy as np

from pathlib import Path
from src.utils.mask_rle_utils import coco_rle_to_masks


def is_dslr_frame(name: str) -> bool:
    """Return True if the frame name is DSLR, e.g. 'DSC0...', not 'frame_xxx'."""
    return not name.startswith("frame_")


class ScanNetPPResizedHardIoUDataset(Dataset):
    def __init__(self, cfg, selected_scenes=None, ref_query_pairs_override=None):
        self.rgb_path = cfg["DATASET"]["DATA_ROOT"]
        self.segdata_path = cfg["DATASET"]["SEGDATA_ROOT"]
        self.pairs_path = cfg["DATASET"]["PAIRS_ROOT"]

        self.selected_scenes = selected_scenes
        self.pairs_list_override = ref_query_pairs_override

        self.ref_query_pairs = []

        if self.pairs_list_override is not None:
            self.ref_query_pairs = self.pairs_list_override
            print(f"Using overridden ref-query pairs: {len(self.ref_query_pairs)}")
        elif self.selected_scenes is not None:
            self._make_ref_query_pairs()
        else:
            raise ValueError(
                "Either selected_scenes or pairs_list_override must be provided."
            )

    def _make_ref_query_pairs(self):
        """
        Reads the segdata folder and constructs ref-query pairs
        """

        available_scenes = sorted(os.listdir(self.segdata_path))
        scenes_to_use = (
            self.selected_scenes
            if self.selected_scenes is not None
            else available_scenes
        )
        print(f"Available scenes: len={len(available_scenes)}")
        print(f"Selected scenes: len={len(scenes_to_use)}")
        print(f"Avaliable scenes: {available_scenes}")

        # check if scenes_to_use is a subset of available_scenes
        if not set(scenes_to_use).issubset(set(available_scenes)):
            incorrect_scenes = set(scenes_to_use) - set(available_scenes)
            raise ValueError(
                f"Scene(s) {incorrect_scenes} not found in the cached_feats_path"
            )
        print(f"Selected scenes: len={len(scenes_to_use)}")

        # NOTE: We will focus on using segdata to construct pairs and pick corresponding rgb images
        for scene_folder in scenes_to_use:
            pairs_file = os.path.join(
                self.pairs_path, scene_folder, "selected_pairs.npz"
            )
            with np.load(pairs_file, allow_pickle=True) as pairs_data:
                selection = pairs_data["selection"]
                pairs = pairs_data["pairs"]  # shape (N,3) => [idxA, idxB, conf]

            for idxA, idxB, conf in pairs:
                idxA, idxB = int(idxA), int(idxB)
                fA, fB = selection[idxA], selection[idxB]
                # Keep DSLR-DSLR
                if is_dslr_frame(fA) and is_dslr_frame(fB):
                    self.ref_query_pairs.append((scene_folder, fA, fB))

        print(f"Total ref-query pairs: {len(self.ref_query_pairs)}")
        if len(self.ref_query_pairs) == 0:
            raise ValueError("No valid pairs found for the selected scenes")

    def _read_pickle_safe(self, filepath):
        try:
            with open(filepath, "rb") as f:
                data = pickle.load(f)
                # Create a new dict with only needed data
                return {
                    "mask_coco_rles_resized": data.get("mask_coco_rles_resized"),
                    "seg_corr_list": data.get("seg_corr_list"),
                }
        except Exception as e:
            raise IndexError(f"Error reading pickle file {filepath}: {str(e)}")

    def _read_segdata(self, scene_id, frameA, frameB):
        ref_pkl_subpath = os.path.join(self.segdata_path, scene_id, f"{frameA}.JPG.pkl")
        ref_segdata = self._read_pickle_safe(ref_pkl_subpath)

        query_pkl_subpath = os.path.join(
            self.segdata_path, scene_id, f"{frameB}.JPG.pkl"
        )
        query_segdata = self._read_pickle_safe(query_pkl_subpath)

        ref_masks_variable_size = coco_rle_to_masks(
            ref_segdata["mask_coco_rles_resized"],
        )
        ref_seg_corr_list_variable_size = ref_segdata["seg_corr_list"].bool()

        query_masks_variable_size = coco_rle_to_masks(
            query_segdata["mask_coco_rles_resized"],
        )
        query_seg_corr_list_variable_size = query_segdata["seg_corr_list"].bool()

        common_seg_corr_list_variable_size = (
            ref_seg_corr_list_variable_size.bool()
            & query_seg_corr_list_variable_size.bool()
        )

        frame_segdata = {}

        frame_segdata[f"masks_gt_0"] = ref_masks_variable_size.to(torch.uint8)
        frame_segdata[f"masks_gt_1"] = query_masks_variable_size.to(torch.uint8)

        frame_segdata[f"seg_corr_list_common"] = common_seg_corr_list_variable_size.to(
            torch.int8
        )
        return frame_segdata
        
    def __len__(self):
        return len(self.ref_query_pairs)

    def __getitem__(self, idx):
        scene_id, frameA, frameB = self.ref_query_pairs[idx]
        frameA_path = Path(self.rgb_path) / scene_id / f"{frameA}.JPG"
        frameB_path = Path(self.rgb_path) / scene_id / f"{frameB}.JPG"
        # Get segdata for the reference and query frames
        try:
            # adding segdata to the ref frame data dicts
            ref_query_segdata = self._read_segdata(scene_id, frameA, frameB)
        except IndexError as e:
            if idx >= len(self):
                raise IndexError("Dataset index out of range")
            return self.__getitem__(idx + 1)

        # add rgb images to the data dict
        ref_query_pair_data_dict = {
            "scene_id": scene_id,
            "img0_name": frameA,
            "img1_name": frameB,
            "img0_path": str(frameA_path),
            "img1_path": str(frameB_path),
            **ref_query_segdata,
        }

        return ref_query_pair_data_dict
