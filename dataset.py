# dataset.py
# Read training data for multi-head FIQA:
# Each line in labels.txt:
#   "<abs_or_any_path>", fr_score, gaze_score, kpt_score
# Example:
#   "/data/images/xxx_face000.jpg", 0.84,0.77,0.67
#
# Assumption:
#   labels.txt and images/ are under the same root directory:
#     <root>/
#       labels.txt
#       images/
#
# This dataset will:
#   1) parse labels.txt
#   2) resolve each image path:
#        - use the path directly if it exists
#        - else try <root>/images/<basename>
#   3) return:
#        image tensor + labels (3 floats) + path

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch.utils.data import Dataset
from PIL import Image

try:
    import torchvision.transforms as T
except Exception as e:
    raise ImportError("torchvision is required. Please install torchvision.") from e


@dataclass
class Sample:
    img_path: Path
    fr: float
    gaze: float
    kpt: float


def build_default_transforms(
    image_size: int = 224,
    train: bool = True,
) -> T.Compose:
    """
    Default transforms for MobileNetV3 style training (ImageNet normalization).
    Keep it simple & safe; you can replace with your own aug pipeline.
    """
    if train:
        return T.Compose(
            [
                T.Resize((image_size, image_size)),
                T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
                T.RandomHorizontalFlip(p=0.5),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
    return T.Compose(
        [
            T.Resize((image_size, image_size)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


class MultiScoreDataset(Dataset):
    """
    Dataset for:
      - face recognition quality score (fr)
      - gaze estimation quality score (gaze)
      - keypoint detection quality score (kpt)
    """

    def __init__(
        self,
        root_dir: str | os.PathLike,
        labels_name: str = "labels.txt",
        images_dirname: str = "images",
        transforms: Optional[Any] = None,
        image_mode: str = "RGB",
        strict: bool = True,
    ):
        """
        Args:
            root_dir: directory containing labels.txt and images/ folder
            labels_name: label file name (default: labels.txt)
            images_dirname: images folder name (default: images)
            transforms: torchvision transform callable; if None, no transform applied
            image_mode: "RGB" (default) or "L" etc.
            strict: if True -> raise error on missing file/parse issues
                    if False -> skip bad lines and missing images
        """
        self.root_dir = Path(root_dir)
        self.labels_path = self.root_dir / labels_name
        self.images_dir = self.root_dir / images_dirname
        self.transforms = transforms
        self.image_mode = image_mode
        self.strict = strict

        if not self.labels_path.exists():
            raise FileNotFoundError(f"labels file not found: {self.labels_path}")
        if not self.images_dir.exists():
            # images folder might not be needed if label paths are valid absolute paths
            # keep as soft check
            if strict:
                raise FileNotFoundError(f"images dir not found: {self.images_dir}")

        self.samples: List[Sample] = self._load_samples(self.labels_path)

        if len(self.samples) == 0:
            raise RuntimeError(f"No valid samples found in {self.labels_path}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        s = self.samples[idx]
        try:
            img = Image.open(s.img_path).convert(self.image_mode)
        except Exception as e:
            if self.strict:
                raise RuntimeError(f"Failed to read image: {s.img_path}") from e
            # fallback: return a black image
            img = Image.new(self.image_mode, (224, 224))

        if self.transforms is not None:
            img = self.transforms(img)

        # labels: float32
        y = torch.tensor([s.fr, s.gaze, s.kpt], dtype=torch.float32)

        return {
            "image": img,
            "labels": y,  # [fr, gaze, kpt]
            "fr": y[0],
            "gaze": y[1],
            "kpt": y[2],
            "path": str(s.img_path),
        }

    # --------------------------- internals ---------------------------

    _LINE_RE = re.compile(
        r"""
        ^\s*
        (?P<path>
            "(?:[^"\\]|\\.)*" |
            '(?:[^'\\]|\\.)*' |
            [^,]+
        )
        \s*,\s*
        (?P<fr>[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)
        \s*,\s*
        (?P<gaze>[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)
        \s*,\s*
        (?P<kpt>[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)
        \s*$
        """,
        re.VERBOSE,
    )

    def _strip_quotes(self, p: str) -> str:
        p = p.strip()
        if (p.startswith('"') and p.endswith('"')) or (p.startswith("'") and p.endswith("'")):
            return p[1:-1]
        return p

    def _resolve_img_path(self, raw_path: str) -> Path:
        """
        Resolve a path that may be:
          - absolute path existing on this machine
          - absolute path from another machine (not existing here)
          - relative path
        Strategy:
          1) try as-is (expanded)
          2) try relative to root_dir
          3) fallback to <root>/images/<basename>
        """
        raw_path = self._strip_quotes(raw_path)
        raw_path = os.path.expanduser(raw_path)
        raw_path = os.path.expandvars(raw_path)

        p = Path(raw_path)

        # 1) as-is
        if p.exists():
            return p

        # 2) relative to root
        p2 = (self.root_dir / p).resolve()
        if p2.exists():
            return p2

        # 3) fallback to images/<basename>
        p3 = (self.images_dir / p.name).resolve()
        if p3.exists():
            return p3

        # give up
        if self.strict:
            raise FileNotFoundError(
                f"Image not found. Tried: {p} | {p2} | {p3}"
            )
        return p3  # may not exist; will be handled in __getitem__

    def _load_samples(self, labels_path: Path) -> List[Sample]:
        samples: List[Sample] = []
        bad_lines = 0
        with labels_path.open("r", encoding="utf-8") as f:
            for ln, line in enumerate(f, start=1):
                line = line.strip()
                if not line or line.startswith("#"):
                    continue

                m = self._LINE_RE.match(line)
                if not m:
                    bad_lines += 1
                    if self.strict:
                        raise ValueError(f"Bad label line format at {labels_path}:{ln}\n{line}")
                    continue

                try:
                    raw_path = m.group("path")
                    fr = float(m.group("fr"))
                    gaze = float(m.group("gaze"))
                    kpt = float(m.group("kpt"))
                    img_path = self._resolve_img_path(raw_path)
                    samples.append(Sample(img_path=img_path, fr=fr, gaze=gaze, kpt=kpt))
                except Exception:
                    bad_lines += 1
                    if self.strict:
                        raise
                    continue

        if (not self.strict) and bad_lines > 0:
            print(f"[MultiScoreDataset] skipped bad/missing lines: {bad_lines}")

        return samples


if __name__ == "__main__":
    # quick test (adjust root_dir)
    root = "./"  # directory containing labels.txt and images/
    ds = MultiScoreDataset(
        root_dir=root,
        transforms=build_default_transforms(image_size=224, train=False),
        strict=False,
    )
    x = ds[0]
    print(x["path"], x["image"].shape, x["labels"])

