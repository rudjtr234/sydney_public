# src/data/dataset.py
# TRIDENT 패치 임베딩(.h5) 기반 bag 데이터셋.
# h5 레이아웃: features (n_patches, dim) float32, coords (n_patches, 2) int64  (TRIDENT features_<enc>/<slide>.h5)

import json
import os
from typing import Dict, List, Optional

import h5py
import torch
from torch.utils.data import Dataset


def feature_path(feat_dir: str, slide_id: str) -> str:
    return os.path.join(feat_dir, f"{slide_id}.h5")


def read_feature_info(path: str) -> Dict:
    """Read (n_patches, dim, encoder) without loading the features."""
    with h5py.File(path, "r") as f:
        n, dim = f["features"].shape
        encoder = f["features"].attrs.get("encoder", "")
    return {"n_patches": int(n), "dim": int(dim), "encoder": str(encoder)}


class H5BagDataset(Dataset):
    """
    One item = one slide: (features [N, D], label, slide_id).

    bag_size: if set and the slide has more patches, a random subset of bag_size
              patches is drawn on every __getitem__ (training augmentation).
              None = use all patches (validation / test).
    """

    def __init__(self, slide_ids: List[str], labels: Dict[str, int], feat_dir: str,
                 bag_size: Optional[int] = None):
        self.slide_ids = list(slide_ids)
        self.labels = labels
        self.feat_dir = feat_dir
        self.bag_size = bag_size

        missing = [s for s in self.slide_ids if not os.path.exists(feature_path(feat_dir, s))]
        if missing:
            raise FileNotFoundError(
                f"{len(missing)} slides have no features in {feat_dir} (e.g. {missing[:5]}). "
                f"Re-run make_splits.py with --feat_dir to drop them, or finish feature extraction."
            )

    def __len__(self):
        return len(self.slide_ids)

    def __getitem__(self, idx):
        slide_id = self.slide_ids[idx]
        with h5py.File(feature_path(self.feat_dir, slide_id), "r") as f:
            features = torch.from_numpy(f["features"][:]).float()

        if self.bag_size is not None and features.shape[0] > self.bag_size:
            # torch RNG is seeded per DataLoader worker, so workers do not draw identical subsets
            indices = torch.randperm(features.shape[0])[: self.bag_size]
            features = features[indices]

        return features, torch.tensor(self.labels[slide_id], dtype=torch.long), slide_id


def load_split_file(path: str) -> Dict:
    """Split JSON written by src/data/make_splits.py."""
    with open(path, "r") as f:
        split = json.load(f)
    for key in ("labels", "label_map", "folds"):
        if key not in split:
            raise KeyError(f"Split file {path} is missing '{key}'. Regenerate it with src/data/make_splits.py")
    return split
