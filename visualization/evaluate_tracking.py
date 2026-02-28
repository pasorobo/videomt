"""Evaluate tracking predictions against ground truth using MOT metrics.

Computes IDSW, MOTA, IDF1, etc. using bbox IoU matching between
VidEoMT predictions (predictions.json) and warehouse GT (ground_truth.json).

Usage:
    python evaluate_tracking.py \
        --predictions ../demo_output_allframes/predictions.json \
        --gt /mnt/d/Program/3DSG/data/warehouse/ground_truth.json \
        --camera Camera_0030 \
        --chunk 1 \
        --iou-threshold 0.3
"""

import argparse
import json
import sys
from collections import defaultdict

import numpy as np


def compute_iou(box_a, box_b):
    """Compute IoU between two [x1, y1, x2, y2] boxes."""
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def greedy_match(cost_matrix, threshold):
    """Greedy matching: pick highest IoU pairs above threshold.

    Returns list of (row_idx, col_idx) matches.
    """
    matches = []
    if cost_matrix.size == 0:
        return matches
    rows, cols = cost_matrix.shape
    used_rows = set()
    used_cols = set()
    # Flatten and sort by descending IoU
    indices = np.argsort(-cost_matrix, axis=None)
    for flat_idx in indices:
        r, c = divmod(int(flat_idx), cols)
        if r in used_rows or c in used_cols:
            continue
        if cost_matrix[r, c] < threshold:
            break
        matches.append((r, c))
        used_rows.add(r)
        used_cols.add(c)
    return matches


def evaluate(predictions, gt_data, camera, chunk, iou_threshold=0.3,
             score_threshold=0.3):
    """Run MOT-style evaluation.

    Args:
        predictions: dict[str, list[dict]] from predictions.json
            key = frame index (within chunk), value = list of {id, label, score, bbox}
        gt_data: dict from ground_truth.json
            key = global frame index, value = list of objects
        camera: camera key in GT bbox dict (e.g. "Camera_0030")
        chunk: chunk index (0-4), determines frame offset
        iou_threshold: minimum IoU for matching
        score_threshold: minimum confidence for predictions

    Returns:
        dict with MOT metrics
    """
    frames_per_chunk = 1800
    gt_offset = chunk * frames_per_chunk

    # Build per-frame GT and prediction lists
    total_gt = 0
    total_pred = 0
    total_tp = 0
    total_fp = 0
    total_fn = 0
    total_idsw = 0

    # Track ID mapping: gt_id -> last matched pred_id
    id_map = {}
    frame_keys = sorted(predictions.keys(), key=lambda x: int(x))

    for frame_key in frame_keys:
        frame_idx = int(frame_key)
        gt_frame_idx = gt_offset + frame_idx
        gt_frame_key = str(gt_frame_idx)

        if gt_frame_key not in gt_data:
            continue

        # Get GT boxes for this camera
        gt_objects = gt_data[gt_frame_key]
        gt_boxes = []
        gt_ids = []
        for obj in gt_objects:
            bbox = obj.get('2d bounding box visible', {}).get(camera)
            if bbox is None:
                continue
            # Filter invalid/empty boxes
            if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
                continue
            # Filter very small boxes (< 10px)
            if (bbox[2] - bbox[0]) < 10 or (bbox[3] - bbox[1]) < 10:
                continue
            gt_boxes.append(bbox)
            gt_ids.append(obj['object id'])

        # Get prediction boxes
        pred_list = predictions[frame_key]
        pred_boxes = []
        pred_ids = []
        for p in pred_list:
            if p['score'] < score_threshold:
                continue
            pred_boxes.append(p['bbox'])
            pred_ids.append(p['id'])

        n_gt = len(gt_boxes)
        n_pred = len(pred_boxes)
        total_gt += n_gt
        total_pred += n_pred

        if n_gt == 0 and n_pred == 0:
            continue

        if n_gt == 0:
            total_fp += n_pred
            continue

        if n_pred == 0:
            total_fn += n_gt
            continue

        # Compute IoU matrix (n_gt x n_pred)
        iou_matrix = np.zeros((n_gt, n_pred))
        for gi in range(n_gt):
            for pi in range(n_pred):
                iou_matrix[gi, pi] = compute_iou(gt_boxes[gi], pred_boxes[pi])

        matches = greedy_match(iou_matrix, iou_threshold)
        matched_gt = {m[0] for m in matches}
        matched_pred = {m[1] for m in matches}

        tp = len(matches)
        fp = n_pred - tp
        fn = n_gt - tp
        total_tp += tp
        total_fp += fp
        total_fn += fn

        # Count ID switches
        for gi, pi in matches:
            gt_id = gt_ids[gi]
            pred_id = pred_ids[pi]
            if gt_id in id_map:
                if id_map[gt_id] != pred_id:
                    total_idsw += 1
            id_map[gt_id] = pred_id

    # Compute metrics
    mota = 1.0 - (total_fp + total_fn + total_idsw) / total_gt if total_gt > 0 else 0.0
    precision = total_tp / total_pred if total_pred > 0 else 0.0
    recall = total_tp / total_gt if total_gt > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    # IDF1: ratio of correctly identified detections
    # Using simplified version based on matched ID consistency
    id_correct = 0
    id_total = 0
    id_map2 = {}
    for frame_key in frame_keys:
        frame_idx = int(frame_key)
        gt_frame_key = str(gt_offset + frame_idx)
        if gt_frame_key not in gt_data:
            continue

        gt_objects = gt_data[gt_frame_key]
        gt_boxes = []
        gt_ids = []
        for obj in gt_objects:
            bbox = obj.get('2d bounding box visible', {}).get(camera)
            if bbox is None or bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
                continue
            if (bbox[2] - bbox[0]) < 10 or (bbox[3] - bbox[1]) < 10:
                continue
            gt_boxes.append(bbox)
            gt_ids.append(obj['object id'])

        pred_list = predictions[frame_key]
        pred_boxes = []
        pred_ids = []
        for p in pred_list:
            if p['score'] < score_threshold:
                continue
            pred_boxes.append(p['bbox'])
            pred_ids.append(p['id'])

        n_gt = len(gt_boxes)
        n_pred = len(pred_boxes)
        if n_gt == 0 or n_pred == 0:
            continue

        iou_matrix = np.zeros((n_gt, n_pred))
        for gi in range(n_gt):
            for pi in range(n_pred):
                iou_matrix[gi, pi] = compute_iou(gt_boxes[gi], pred_boxes[pi])

        matches = greedy_match(iou_matrix, iou_threshold)
        for gi, pi in matches:
            gt_id = gt_ids[gi]
            pred_id = pred_ids[pi]
            id_total += 1
            # Majority vote: assign gt_id to most common pred_id
            if gt_id not in id_map2:
                id_map2[gt_id] = pred_id
            if id_map2[gt_id] == pred_id:
                id_correct += 1

    idf1 = 2 * id_correct / (total_gt + total_pred) if (total_gt + total_pred) > 0 else 0.0

    # Unique GT IDs and Pred IDs tracked
    unique_gt_ids = set()
    unique_pred_ids = set()
    for frame_key in frame_keys:
        gt_frame_key = str(gt_offset + int(frame_key))
        if gt_frame_key in gt_data:
            for obj in gt_data[gt_frame_key]:
                bbox = obj.get('2d bounding box visible', {}).get(camera)
                if bbox and bbox[2] > bbox[0] and bbox[3] > bbox[1]:
                    unique_gt_ids.add(obj['object id'])
        for p in predictions[frame_key]:
            if p['score'] >= score_threshold:
                unique_pred_ids.add(p['id'])

    return {
        "frames_evaluated": len(frame_keys),
        "total_gt_detections": total_gt,
        "total_pred_detections": total_pred,
        "unique_gt_ids": len(unique_gt_ids),
        "unique_pred_ids": len(unique_pred_ids),
        "TP": total_tp,
        "FP": total_fp,
        "FN": total_fn,
        "IDSW": total_idsw,
        "MOTA": round(mota, 4),
        "Precision": round(precision, 4),
        "Recall": round(recall, 4),
        "F1": round(f1, 4),
        "IDF1": round(idf1, 4),
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate MOT tracking metrics")
    parser.add_argument("--predictions", required=True, help="predictions.json path")
    parser.add_argument("--gt", required=True, help="ground_truth.json path")
    parser.add_argument("--camera", default="Camera_0030", help="camera key in GT")
    parser.add_argument("--chunk", type=int, default=1,
                        help="video chunk index (0-4), determines GT frame offset")
    parser.add_argument("--iou-threshold", type=float, default=0.3,
                        help="IoU threshold for matching")
    parser.add_argument("--score-threshold", type=float, default=0.3,
                        help="confidence threshold for predictions")
    args = parser.parse_args()

    print(f"Loading predictions from {args.predictions}...")
    with open(args.predictions) as f:
        predictions = json.load(f)
    print(f"  {len(predictions)} frames")

    print(f"Loading GT from {args.gt}...")
    with open(args.gt) as f:
        gt_data = json.load(f)
    print(f"  {len(gt_data)} frames")

    print(f"\nEvaluating: camera={args.camera}, chunk={args.chunk}, "
          f"IoU>={args.iou_threshold}, score>={args.score_threshold}")
    print("=" * 60)

    results = evaluate(
        predictions, gt_data, args.camera, args.chunk,
        iou_threshold=args.iou_threshold,
        score_threshold=args.score_threshold,
    )

    for k, v in results.items():
        print(f"  {k:25s}: {v}")

    print("=" * 60)
    print(f"\n  MOTA = {results['MOTA']:.4f}  |  IDSW = {results['IDSW']}  |  "
          f"IDF1 = {results['IDF1']:.4f}")
    print(f"  Precision = {results['Precision']:.4f}  |  "
          f"Recall = {results['Recall']:.4f}  |  F1 = {results['F1']:.4f}")

    return results


if __name__ == "__main__":
    main()
