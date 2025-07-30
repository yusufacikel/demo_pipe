# utils/data.py
import re
from pathlib import Path
from collections import namedtuple
import cv2
import numpy as np
from typing import Union, List, TypedDict


INPUT_FILENAME_REGEX = re.compile(r"(?:Color|Depth)_?(\d+\.?\d*)\.png")

ImagePair = namedtuple('ImagePair', ['depth_path', 'color_path', 'raw_timestamp'])

class FrameData(TypedDict):
    timestamp: float
    color: np.ndarray
    depth_raw: np.ndarray

class RGBDDataset:
    def __init__(self, run_dir_path: Union[str, Path]):
        self.run_dir_path = Path(run_dir_path)
        self.paired_files = self._load_and_pair_files()

    def _extract_timestamp(self, filename: str) -> float:
        match = INPUT_FILENAME_REGEX.search(filename)
        return float(match.group(1)) if match else 0.0

    def _load_and_pair_files(self) -> List[ImagePair]:
        depth_dir = self.run_dir_path / "depth"
        color_dir = self.run_dir_path / "color"

        depth_paths = sorted(depth_dir.glob("*.png"), key=lambda p: self._extract_timestamp(p.name))
        color_paths = sorted(color_dir.glob("*.png"), key=lambda p: self._extract_timestamp(p.name))

        assert len(depth_paths) == len(color_paths), f"Depth and color counts differ: {len(depth_paths)} vs {len(color_paths)}"

        pairs = []
        for d_path, c_path in zip(depth_paths, color_paths):
            ts = self._extract_timestamp(d_path.name)
            pairs.append(ImagePair(
                depth_path=d_path,
                color_path=c_path,
                raw_timestamp=ts,
            ))
        return pairs

    def __len__(self) -> int:
        return len(self.paired_files)

    def __getitem__(self, idx: Union[int, slice]) -> Union[FrameData, List[FrameData]]:
        if isinstance(idx, slice):
            start, stop, step = idx.indices(len(self))
            return [self._get_single_item(i) for i in range(start, stop, step)]
        
        if isinstance(idx, int):
            if idx < 0:
                idx = len(self) + idx
            return self._get_single_item(idx)
        
        raise TypeError(f"Index must be int or slice, got {type(idx)}")

    def _get_single_item(self, idx: int) -> FrameData:
        if idx < 0 or idx >= len(self):
            raise IndexError("Index out of range")

        pair = self.paired_files[idx]

        color = cv2.imread(str(pair.color_path), cv2.IMREAD_COLOR)
        if color is None:
            raise IOError(f"Failed to load color image: {pair.color_path}")

        depth_raw = cv2.imread(str(pair.depth_path), cv2.IMREAD_ANYDEPTH)
        if depth_raw is None:
            raise IOError(f"Failed to load depth image: {pair.depth_path}")
        depth_raw = depth_raw.astype(np.uint16)

        return FrameData(
            timestamp=pair.raw_timestamp,
            color=color,
            depth_raw=depth_raw,
        )

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]