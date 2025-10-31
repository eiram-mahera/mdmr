from __future__ import annotations
import os, glob
import logging
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple
import numpy as np, skimage.io
from .base import BaseDataset, register_dataset

BBox = Tuple[int, int, int, int]  # (y, x, h, w)
logger = logging.getLogger(__name__)

@register_dataset("ctc")
class CTCDataset(BaseDataset):
    def __init__(self, data_dir: str, dataset_name: str, crop_boxes: Optional[Dict[str,BBox]] = None):
        self.data_dir, self.dataset_name = data_dir, dataset_name
        self.train_video, self.test_video = "01", "02"
        root = Path(data_dir)/dataset_name
        self.image_dirs = {"01": str(root/"01"), "02": str(root/"02")}
        self.gt_dirs    = {"01": str(root/"01_GT"/"SEG"), "02": str(root/"02_GT"/"SEG")}
        for v in ("01","02"):
            if not Path(self.image_dirs[v]).is_dir():
                logger.error(f"File not found {self.image_dirs[v]}")
                raise FileNotFoundError(self.image_dirs[v])
            if not Path(self.gt_dirs[v]).is_dir():
                logger.error(f"File not found {self.gt_dirs[v]}")
                raise FileNotFoundError(self.gt_dirs[v])
        self.crop_boxes: Dict[str, Optional[BBox]] = {"01": None, "02": None}
        if crop_boxes:
            for k,v in crop_boxes.items():
                if k not in ("01","02"):
                    logger.error("crop_boxes keys must be '01'/'02'")
                    raise ValueError("crop_boxes keys must be '01'/'02'")
                if not (isinstance(v,(tuple,list)) and len(v)==4 and all(isinstance(x,int) for x in v)):
                    logger.error("each crop box must be (y,x,h,w) ints")
                    raise ValueError("each crop box must be (y,x,h,w) ints")
                self.crop_boxes[k] = (int(v[0]), int(v[1]), int(v[2]), int(v[3]))
        self.train_image_files = self._sorted_tifs(self.image_dirs["01"])
        self.test_image_files  = self._sorted_tifs(self.image_dirs["02"])
        self.train_mask_files  = self._sorted_tifs(self.gt_dirs["01"])
        self.test_mask_files   = self._sorted_tifs(self.gt_dirs["02"])
        assert len(self.train_image_files)==len(self.train_mask_files)
        assert len(self.test_image_files)==len(self.test_mask_files)
        self.labeled_indices: List[int] = []
        self.unlabeled_indices: List[int] = list(range(len(self.train_image_files)))
        self.test_indices: List[int] = list(range(len(self.test_image_files)))

    def _sorted_tifs(self, folder: str) -> List[str]:
        if not os.path.isdir(folder): return []
        return sorted(glob.glob(os.path.join(folder, "*.tif")))

    def load_images(self, indices: Iterable[int], video: str) -> List[np.ndarray]:
        files = self.train_image_files if video=="01" else self.test_image_files
        crop  = self.crop_boxes.get(video)
        return [self._read_and_crop(files[i], crop) for i in indices]

    def load_masks(self, indices: Iterable[int], video: str) -> List[np.ndarray]:
        files = self.train_mask_files if video=="01" else self.test_mask_files
        crop  = self.crop_boxes.get(video)
        return [self._read_and_crop(files[i], crop) for i in indices]

    def annotate(self, indices: Iterable[int]) -> None:
        move = set(int(i) for i in indices)
        self.labeled_indices = sorted(set(self.labeled_indices) | move)
        self.unlabeled_indices = sorted(set(self.unlabeled_indices) - move)

    def unannotate(self, indices: Iterable[int]) -> None:
        move = set(int(i) for i in indices)
        self.unlabeled_indices = sorted(set(self.unlabeled_indices) | move)
        self.labeled_indices   = sorted(set(self.labeled_indices) - move)

    @staticmethod
    def _normalize_box(box: BBox, shape: Tuple[int,...]) -> BBox:
        H,W = int(shape[0]), int(shape[1])
        y,x,h,w = int(box[0]), int(box[1]), int(box[2]), int(box[3])
        h=max(0,h); w=max(0,w); y=max(0,y); x=max(0,x)
        y2=min(H,y+h); x2=min(W,x+w)
        h=max(0,y2-y); w=max(0,x2-x)
        return y,x,h,w

    def _read_and_crop(self, path: str, crop: Optional[BBox]) -> np.ndarray:
        arr = skimage.io.imread(path)
        if crop is None: return arr
        y,x,h,w = self._normalize_box(crop, arr.shape)
        if h==0 or w==0: return arr
        if arr.ndim==2: return arr[y:y+h, x:x+w]
        sl = (slice(y,y+h), slice(x,x+w)) + (slice(None),)*(arr.ndim-2)
        return arr[sl]

    def uncrop_like_original(self, cropped: np.ndarray, video: str, idx: int) -> np.ndarray:
        src = self.train_image_files if video=="01" else self.test_image_files
        orig = skimage.io.imread(src[idx]); H,W = orig.shape[:2]
        canvas = np.zeros((H,W)+cropped.shape[2:], dtype=cropped.dtype)
        crop = self.crop_boxes.get(video)
        if crop is None: return cropped
        y,x,h,w = self._normalize_box(crop, (H,W))
        if h==0 or w==0: return cropped
        if cropped.ndim==2: canvas[y:y+h, x:x+w]=cropped
        else:
            sl=(slice(y,y+h),slice(x,x+w))+(slice(None),)*(cropped.ndim-2)
            canvas[sl]=cropped
        return canvas
