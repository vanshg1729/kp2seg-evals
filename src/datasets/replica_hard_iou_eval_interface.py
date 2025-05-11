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


class ReplicaHardIoUDataset(Dataset):
    def __init__(self, cfg, selected_scenes=None, ref_query_pairs_override=None):
        self.data_root = cfg["DATASET"]["DATA_ROOT"]
        self.rgb_path = cfg["DATASET"]["DATA_ROOT"]
        self.segdata_path = cfg["DATASET"]["SEGDATA_ROOT"]
        self.pairs_json_path = cfg["DATASET"]["PAIRS_PATH"]

        self.selected_scenes = selected_scenes
        self.pairs_list_override = ref_query_pairs_override

        self.ref_query_pairs = []

        if self.pairs_list_override is not None:
            self.ref_query_pairs = self.pairs_list_override
            print(f"Using overridden ref-query pairs: {len(self.ref_query_pairs)}")
        elif self.selected_scenes is not None:
            self._get_ref_query_pairs()
        else:
            raise ValueError(
                "Either selected_scenes or pairs_list_override must be provided."
            )

    def _get_ref_query_pairs(self):
        """
        Reads replica_pose_bins_subset.json file and gets the (ref, query) pairs
        """
        with open(self.pairs_json_path, 'r') as file:
            pairs = json.load(file)

        # pairs[scene_name][pose] = {dict of (pair_name, pair_data like rotation-delta and trans)}
        
        for scene_folder, scene_data in pairs.items():
            for pose, pose_data in scene_data.items():
                for pair_name in pose_data.keys():
                    fA, fB = pair_name.split('-')
                    self.ref_query_pairs.append((scene_folder, fA, fB))
        
        print(f"Total ref-query pairs: {len(self.ref_query_pairs)}")
        if len(self.ref_query_pairs) == 0:
            raise ValueError("No valid pairs found for the replica-dataset scenes")

    def _read_pickle_safe(self, filepath):
        try:
            with open(filepath, "rb") as f:
                data = pickle.load(f)
                # Create a new dict with only needed data
                return {
                    "mask_coco_rles": data.get("masks_coco_rles"),
                    "seg_corr_list": data.get("seg_corr_list"),
                }
        except Exception as e:
            raise IndexError(f"Error reading pickle file {filepath}: {str(e)}")

    def _read_segdata(self, scene_id, frameA, frameB):
        ref_num = int(frameA.split('_')[-1])
        query_num = int(frameB.split('_')[-1])
        
        # Reference
        ref_pkl_subpath = os.path.join(self.segdata_path, scene_id, f"semantic_coco/semantic_class_{ref_num}.pkl")
        assert os.path.exists(ref_pkl_subpath), f"{ref_pkl_subpath} doesn't exist"
        ref_segdata = self._read_pickle_safe(ref_pkl_subpath)

        # Query
        query_pkl_subpath = os.path.join(
            self.segdata_path, scene_id, f"semantic_coco/semantic_class_{query_num}.pkl"
        )
        assert os.path.exists(query_pkl_subpath), f"{query_pkl_subpath} doesn't exist"
        query_segdata = self._read_pickle_safe(query_pkl_subpath)

        ref_masks_variable_size = coco_rle_to_masks(
            ref_segdata["mask_coco_rles"],
        )
        ref_seg_corr_list_variable_size = ref_segdata["seg_corr_list"].bool()

        query_masks_variable_size = coco_rle_to_masks(
            query_segdata["mask_coco_rles"],
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
        frameA_path = Path(self.data_root) / scene_id / f"rgb/{frameA}.png"
        frameB_path = Path(self.data_root) / scene_id / f"rgb/{frameB}.png"
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
