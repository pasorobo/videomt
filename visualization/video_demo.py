# Copyright (c) 2021-2022, NVIDIA Corporation & Affiliates. All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, visit
# https://github.com/NVlabs/MinVIS/blob/main/LICENSE

# Copyright (c) Facebook, Inc. and its affiliates.
# Modified by Bowen Cheng from: https://github.com/facebookresearch/detectron2/blob/master/demo/demo.py

import torch
import argparse
import cv2
import glob
import json
import multiprocessing as mp
import numpy as np
import os
import shutil
import subprocess

# fmt: off
import sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))
# fmt: on

import time
import tqdm

from detectron2.config import get_cfg
from detectron2.data.detection_utils import read_image
from detectron2.projects.deeplab import add_deeplab_config
from detectron2.utils.logger import setup_logger

from videomt import add_videomt_config
from predictor import VisualizationDemo, VisualizationDemo_windows

VIDEO_EXTENSIONS = {'.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm'}


def setup_cfg(args):
    # load config from file and command-line arguments
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_videomt_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    return cfg


def get_parser():
    parser = argparse.ArgumentParser(description="VidEoMT video demo")
    parser.add_argument(
        "--config-file",
        default="configs/youtubevis_2019/video_maskformer2_R50_bs32_8ep_frame.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument(
        "--input",
        help="video file (.mp4/.avi/.mov etc.) or directory of input frames",
        required=True,
    )
    parser.add_argument(
        "--output",
        help="directory to save output",
        default="./demo_output",
    )
    parser.add_argument(
        "-i", "--sample-interval",
        type=float,
        default=None,
        help="frame sampling interval in seconds (video input only)",
    )
    parser.add_argument(
        "-n", "--max-frames",
        type=int,
        default=None,
        help="maximum number of frames to process",
    )
    parser.add_argument(
        "--save-video",
        action="store_true",
        help="save output as MP4 video",
    )
    parser.add_argument(
        "--no-save-frames",
        action="store_true",
        help="disable saving individual frame images",
    )
    parser.add_argument(
        "--video-fps",
        type=int,
        default=10,
        help="output video FPS (default: 10)",
    )
    parser.add_argument(
        "--video-codec",
        choices=["h264", "mp4v"],
        default="h264",
        help="output video codec: h264 (high compression) or mp4v (fast)",
    )
    parser.add_argument(
        "--save-predictions",
        action="store_true",
        help="save per-frame predictions (bbox, id, score, label) as JSON",
    )
    parser.add_argument(
        "--confidence_threshold",
        type=float,
        default=0.5,
        help="Minimum score for instance predictions to be shown",
    )
    parser.add_argument(
        "--windows_size",
        type=int,
        default=20,
        help="Windows size for semi-offline mode",
    )
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=[],
        nargs=argparse.REMAINDER,
    )
    return parser


def read_video_frames(video_path, sample_interval=None, max_frames=None):
    """Read frames from a video file using cv2.VideoCapture.

    Args:
        video_path: path to video file.
        sample_interval: sampling interval in seconds. None = every frame.
        max_frames: maximum number of frames to return.

    Returns:
        frames: list of BGR np.ndarray images.
        source_fps: original video FPS.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    source_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if sample_interval is not None and sample_interval > 0:
        frame_interval = max(1, int(source_fps * sample_interval))
    else:
        frame_interval = 1

    frames = []
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % frame_interval == 0:
            frames.append(frame)  # BGR
            if max_frames is not None and len(frames) >= max_frames:
                break
        frame_idx += 1

    cap.release()
    return frames, source_fps


def read_directory_frames(dir_path, max_frames=None):
    """Read frames from a directory of images.

    Args:
        dir_path: path to directory containing image files.
        max_frames: maximum number of frames to return.

    Returns:
        frames: list of BGR np.ndarray images.
        frame_names: list of filenames (for output naming).
    """
    frames_path = glob.glob(os.path.expanduser(os.path.join(dir_path, '*.*')))
    # Filter to common image extensions
    image_exts = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}
    frames_path = [p for p in frames_path if os.path.splitext(p)[1].lower() in image_exts]
    frames_path.sort()

    if max_frames is not None:
        frames_path = frames_path[:max_frames]

    frames = []
    frame_names = []
    for path in frames_path:
        img = read_image(path, format="BGR")
        frames.append(img)
        frame_names.append(os.path.basename(path))

    return frames, frame_names


class VideoWriter:
    """Write frames to an MP4 video file.

    Attempts H.264 encoding via ffmpeg for better compression.
    Falls back to mp4v (OpenCV built-in) if ffmpeg is unavailable.
    """

    def __init__(self, output_path, fps=10, codec="h264"):
        self.output_path = output_path
        self.fps = fps
        self.codec = codec
        self._writer = None
        self._tmp_path = None
        self._frame_size = None

    def add_frame(self, frame_rgb):
        """Add a frame (RGB np.ndarray, H x W x 3) to the video."""
        frame_bgr = frame_rgb[:, :, ::-1]  # RGB -> BGR
        h, w = frame_bgr.shape[:2]

        if self._writer is None:
            self._frame_size = (w, h)
            if self.codec == "h264":
                # Write with mp4v first, then re-encode to H.264 via ffmpeg
                self._tmp_path = self.output_path + ".tmp.mp4"
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                self._writer = cv2.VideoWriter(
                    self._tmp_path, fourcc, self.fps, self._frame_size
                )
            else:
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                self._writer = cv2.VideoWriter(
                    self.output_path, fourcc, self.fps, self._frame_size
                )

        self._writer.write(frame_bgr)

    def close(self):
        """Finalize the video file. Re-encode to H.264 if requested."""
        if self._writer is None:
            return

        self._writer.release()
        self._writer = None

        if self._tmp_path is not None and os.path.exists(self._tmp_path):
            if self._try_ffmpeg_h264():
                os.remove(self._tmp_path)
            else:
                # Fallback: just rename tmp as final output
                shutil.move(self._tmp_path, self.output_path)

    def _try_ffmpeg_h264(self):
        """Try to re-encode tmp file to H.264 using ffmpeg."""
        if shutil.which("ffmpeg") is None:
            return False
        try:
            cmd = [
                "ffmpeg", "-y",
                "-i", self._tmp_path,
                "-c:v", "libx264",
                "-preset", "fast",
                "-crf", "23",
                "-pix_fmt", "yuv420p",
                "-movflags", "+faststart",
                self.output_path,
            ]
            subprocess.run(
                cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                check=True, timeout=600,
            )
            return True
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
            return False


def _is_video_file(path):
    """Check if path is a video file based on extension."""
    return os.path.isfile(path) and os.path.splitext(path)[1].lower() in VIDEO_EXTENSIONS


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    args = get_parser().parse_args()
    setup_logger(name="fvcore")
    logger = setup_logger()
    logger.info("Arguments: " + str(args))

    cfg = setup_cfg(args)

    demo = VisualizationDemo_windows(cfg)

    assert args.input and args.output

    output_root = args.output
    score_threshold = args.confidence_threshold
    windows_size = args.windows_size
    save_frames = not args.no_save_frames
    save_predictions = args.save_predictions

    os.makedirs(output_root, exist_ok=True)

    # --- Determine input type and load frames ---
    is_video = _is_video_file(args.input)

    if is_video:
        logger.info(f"Reading video file: {args.input}")
        all_frames, source_fps = read_video_frames(
            args.input,
            sample_interval=args.sample_interval,
            max_frames=args.max_frames,
        )
        # Generate frame names for saving
        frame_names = [f"frame_{i:06d}.jpg" for i in range(len(all_frames))]
        logger.info(
            f"Loaded {len(all_frames)} frames from video "
            f"(source FPS: {source_fps:.1f})"
        )
    else:
        logger.info(f"Reading frame directory: {args.input}")
        if args.sample_interval is not None:
            logger.warning("--sample-interval is ignored for directory input")
        all_frames, frame_names = read_directory_frames(
            args.input, max_frames=args.max_frames,
        )
        logger.info(f"Loaded {len(all_frames)} frames from directory")

    if len(all_frames) == 0:
        logger.error("No frames loaded. Check your input path.")
        sys.exit(1)

    if windows_size == -1:
        windows_size = len(all_frames)

    # --- Set up video writer ---
    video_writer = None
    if args.save_video:
        video_out_path = os.path.join(output_root, "output.mp4")
        video_writer = VideoWriter(
            video_out_path, fps=args.video_fps, codec=args.video_codec,
        )
        logger.info(
            f"Video output enabled: {video_out_path} "
            f"(FPS={args.video_fps}, codec={args.video_codec})"
        )

    # --- Windowed inference loop ---
    start_time = time.time()
    vid_frames = []
    frame_indices = []
    instances = set()
    processed = 0
    all_predictions = {}  # frame_idx -> per-frame predictions

    for i, frame in enumerate(tqdm.tqdm(all_frames, desc="Processing")):
        vid_frames.append(frame)
        frame_indices.append(i)

        if len(vid_frames) == windows_size or i == len(all_frames) - 1:
            # do inference
            with torch.amp.autocast(device_type="cuda"):
                if processed == 0:
                    predictions, visualized_output = demo.run_on_video(
                        vid_frames, keep=False
                    )
                else:
                    predictions, visualized_output = demo.run_on_video(
                        vid_frames, keep=True
                    )

            # Extract per-frame prediction data for evaluation
            if save_predictions and 'pred_masks' in predictions:
                pred_masks = predictions['pred_masks']
                pred_scores = predictions.get('pred_scores', [])
                pred_labels = predictions.get('pred_labels', [])
                pred_ids = predictions.get('pred_ids', [])
                # pred_masks: list of N tensors, each (T, H, W)
                if len(pred_masks) > 0:
                    frame_masks_list = list(zip(*pred_masks))
                    for fi, idx in enumerate(frame_indices):
                        frame_preds = []
                        for obj_i in range(len(pred_scores)):
                            mask = frame_masks_list[fi][obj_i]
                            if isinstance(mask, torch.Tensor):
                                mask_np = mask.cpu().numpy()
                            else:
                                mask_np = np.array(mask)
                            # Compute bbox from mask
                            ys, xs = np.where(mask_np > 0)
                            if len(xs) == 0:
                                continue
                            x1, y1 = int(xs.min()), int(ys.min())
                            x2, y2 = int(xs.max()), int(ys.max())
                            frame_preds.append({
                                "id": int(pred_ids[obj_i]) if pred_ids else obj_i,
                                "label": int(pred_labels[obj_i]),
                                "score": float(pred_scores[obj_i]),
                                "bbox": [x1, y1, x2, y2],
                            })
                        all_predictions[idx] = frame_preds

            # do save
            for idx, _vis_output in zip(frame_indices, visualized_output):
                # Save individual frame
                if save_frames:
                    out_filename = os.path.join(output_root, frame_names[idx])
                    _vis_output.save(out_filename)
                # Add to video
                if video_writer is not None:
                    video_writer.add_frame(_vis_output.get_image())

            if 'pred_ids' in predictions.keys():
                for id in predictions['pred_ids']:
                    instances.add(id)

            processed += len(vid_frames)
            del visualized_output, vid_frames, frame_indices, predictions
            vid_frames = []
            frame_indices = []

    # --- Finalize ---
    if video_writer is not None:
        video_writer.close()
        logger.info(f"Output video saved: {os.path.join(output_root, 'output.mp4')}")

    if save_predictions and all_predictions:
        pred_path = os.path.join(output_root, "predictions.json")
        with open(pred_path, 'w') as f:
            json.dump(all_predictions, f)
        logger.info(f"Predictions saved: {pred_path} ({len(all_predictions)} frames)")

    elapsed = time.time() - start_time
    logger.info(
        "detected {} instances in {} frames in {:.2f}s ({:.1f} fps)".format(
            len(instances), processed, elapsed,
            processed / elapsed if elapsed > 0 else 0,
        )
    )
