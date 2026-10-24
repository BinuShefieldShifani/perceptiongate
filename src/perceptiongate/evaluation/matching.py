"""Match detections to ground truth, once, and store the result as flat arrays.

Matching follows COCO: within each image and class, detections are taken in order of
score, and each one claims the unmatched ground-truth box it overlaps most, if the IoU
reaches the threshold. This is done for the ten IoU thresholds 0.50, 0.55, ..., 0.95.

Matching only depends on what is inside one image. So after this step, metrics for any
subset of images (a slice, or a bootstrap resample) are just sums and sorts over these
arrays. That is what makes per-slice confidence intervals cheap.

Boxes are normalised xyxy (0..1). IoU doesn't change when x and y are scaled
independently, so normalised coordinates give the same IoU as pixels.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

IOU_THRESHOLDS = np.round(np.arange(0.50, 0.96, 0.05), 2)

FloatArray = NDArray[np.float64]


@dataclass
class ImageData:
    """Ground truth and predictions for one image. Boxes are normalised xyxy."""

    name: str
    gt_boxes: FloatArray  # (M, 4)
    gt_cls: NDArray[np.int64]  # (M,)
    det_boxes: FloatArray  # (N, 4)
    det_cls: NDArray[np.int64]  # (N,)
    det_score: FloatArray  # (N,)


@dataclass
class MatchedArrays:
    names: list[str]
    det_img: NDArray[np.int64]
    det_cls: NDArray[np.int64]
    det_score: FloatArray
    det_tp: NDArray[np.bool_]  # (N, len(IOU_THRESHOLDS))
    gt_img: NDArray[np.int64]
    gt_cls: NDArray[np.int64]
    gt_area: FloatArray  # normalised area (0..1)
    gt_hit: NDArray[np.bool_]  # matched at IoU 0.5 by a detection with score >= op_conf
    op_conf: float

    def save(self, path: str) -> None:
        np.savez_compressed(
            path,
            names=np.array(self.names),
            det_img=self.det_img,
            det_cls=self.det_cls,
            det_score=self.det_score,
            det_tp=self.det_tp,
            gt_img=self.gt_img,
            gt_cls=self.gt_cls,
            gt_area=self.gt_area,
            gt_hit=self.gt_hit,
            op_conf=np.array(self.op_conf),
        )

    @classmethod
    def load(cls, path: str) -> MatchedArrays:
        z = np.load(path)
        return cls(
            names=[str(n) for n in z["names"]],
            det_img=z["det_img"],
            det_cls=z["det_cls"],
            det_score=z["det_score"],
            det_tp=z["det_tp"],
            gt_img=z["gt_img"],
            gt_cls=z["gt_cls"],
            gt_area=z["gt_area"],
            gt_hit=z["gt_hit"],
            op_conf=float(z["op_conf"]),
        )


def box_iou(a: FloatArray, b: FloatArray) -> FloatArray:
    """Pairwise IoU between (N,4) and (M,4) xyxy boxes."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    wh = np.clip(rb - lt, 0, None)
    inter = wh[..., 0] * wh[..., 1]
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    union = area_a[:, None] + area_b[None, :] - inter
    return np.where(union > 0, inter / np.maximum(union, 1e-12), 0.0)


def _match_class(
    det_boxes: FloatArray, det_score: FloatArray, gt_boxes: FloatArray, op_conf: float
) -> tuple[NDArray[np.bool_], NDArray[np.bool_]]:
    n_det, n_gt = len(det_boxes), len(gt_boxes)
    tp = np.zeros((n_det, len(IOU_THRESHOLDS)), dtype=bool)
    hit = np.zeros(n_gt, dtype=bool)
    if n_det == 0 or n_gt == 0:
        return tp, hit
    order = np.argsort(-det_score, kind="stable")
    iou = box_iou(det_boxes[order], gt_boxes)
    # Most detections at low confidence overlap nothing; skip them quickly.
    candidates = np.nonzero(iou.max(axis=1) >= IOU_THRESHOLDS[0])[0]
    for t, thr in enumerate(IOU_THRESHOLDS):
        taken = np.zeros(n_gt, dtype=bool)
        for d in candidates:
            row = np.where(taken, -1.0, iou[d])
            g = int(np.argmax(row))
            if row[g] >= thr:
                taken[g] = True
                tp[order[d], t] = True
                if t == 0 and det_score[order[d]] >= op_conf:
                    hit[g] = True
    return tp, hit


def match_images(images: list[ImageData], op_conf: float = 0.25) -> MatchedArrays:
    det_img, det_cls, det_score, det_tp = [], [], [], []
    gt_img, gt_cls, gt_area, gt_hit = [], [], [], []
    for i, im in enumerate(images):
        classes = np.union1d(im.gt_cls, im.det_cls)
        for c in classes:
            dm, gm = im.det_cls == c, im.gt_cls == c
            tp, hit = _match_class(im.det_boxes[dm], im.det_score[dm], im.gt_boxes[gm], op_conf)
            n_det, n_gt = int(dm.sum()), int(gm.sum())
            det_img.append(np.full(n_det, i))
            det_cls.append(np.full(n_det, c))
            det_score.append(im.det_score[dm])
            det_tp.append(tp)
            g = im.gt_boxes[gm]
            gt_img.append(np.full(n_gt, i))
            gt_cls.append(np.full(n_gt, c))
            gt_area.append((g[:, 2] - g[:, 0]) * (g[:, 3] - g[:, 1]))
            gt_hit.append(hit)

    def cat(parts: Sequence[Any], dtype: Any, shape: tuple[int, ...] = (0,)) -> Any:
        return np.concatenate(parts).astype(dtype) if parts else np.zeros(shape, dtype=dtype)

    det_order = np.argsort(cat(det_img, np.int64), kind="stable")
    return MatchedArrays(
        names=[im.name for im in images],
        det_img=cat(det_img, np.int64)[det_order],
        det_cls=cat(det_cls, np.int64)[det_order],
        det_score=cat(det_score, np.float64)[det_order],
        det_tp=cat(det_tp, bool, (0, len(IOU_THRESHOLDS)))[det_order],
        gt_img=cat(gt_img, np.int64),
        gt_cls=cat(gt_cls, np.int64),
        gt_area=cat(gt_area, np.float64),
        gt_hit=cat(gt_hit, bool),
        op_conf=op_conf,
    )
