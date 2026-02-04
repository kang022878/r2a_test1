from dataclasses import dataclass
from typing import Optional

from .config import IOU_THRESHOLD, TRACK_TTL_SEC


@dataclass
class Track:
    track_id: int
    class_id: int
    bbox: tuple
    last_seen: float
    stylized_crop: object
    last_mask_crop: Optional[object]
    seed: int


TRACKS = []
NEXT_TRACK_ID = 1
FRAME_COUNT = 0
LAST_DETS = []
LAST_IMAGE_WH = (0, 0)


def bbox_iou(a, b) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    iw = max(0, inter_x2 - inter_x1)
    ih = max(0, inter_y2 - inter_y1)
    inter = iw * ih
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - inter
    if union <= 0:
        return 0.0
    return inter / union


def match_track(class_id: int, bbox: tuple):
    best = None
    best_iou = 0.0
    for t in TRACKS:
        if t.class_id != class_id:
            continue
        iou = bbox_iou(t.bbox, bbox)
        if iou > best_iou:
            best_iou = iou
            best = t
    if best_iou >= IOU_THRESHOLD:
        return best
    return None


def prune_tracks(now_ts: float, keep_id: Optional[int] = None):
    global TRACKS
    kept = []
    for t in TRACKS:
        if keep_id is not None and t.track_id == keep_id:
            kept.append(t)
        elif (now_ts - t.last_seen) <= TRACK_TTL_SEC:
            kept.append(t)
    TRACKS = kept
