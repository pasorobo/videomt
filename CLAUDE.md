# CLAUDE.md — VidEoMT

## Project Overview

VidEoMT (Video Encoder-only Mask Transformer) is a CVPR 2026 research project implementing a lightweight encoder-only model for online video segmentation built on plain Vision Transformers (ViT). It performs spatial-temporal reasoning within the ViT encoder without dedicated tracking modules, achieving 5–10x speed improvements (up to 160 FPS) over existing approaches.

**Supported tasks:** Video Instance Segmentation (VIS), Video Panoptic Segmentation (VPS), Video Semantic Segmentation (VSS).

## Repository Structure

```
videomt/                        # Main Python package
├── __init__.py                 # Public API exports
├── config.py                   # Detectron2-based configuration (add_videomt_config)
├── videomt.py                  # Main architecture: videomt, videomt_segmenter, videomt_online
├── criterion_videomt.py        # Loss functions (dice, cross-entropy, point rendering)
├── modeling/
│   ├── matcher.py              # Hungarian matcher for bipartite assignment
│   ├── two_stage_warmup_poly_schedule.py  # LR scheduler
│   └── backbone/
│       ├── videomt.py          # VidEoMT_CLASS backbone implementation
│       ├── vit.py              # ViT wrapper using timm
│       └── scale_block.py      # Upscaling blocks for mask generation
├── data_video/
│   ├── build.py                # Data loader builders
│   ├── combined_loader.py      # Multi-dataset loader with sampling ratios
│   ├── dataset_mapper.py       # YTVIS dataset mapping
│   ├── dataset_mapper_vps.py   # VIPSeg dataset mapping
│   ├── dataset_mapper_vss.py   # VSPW dataset mapping
│   ├── augmentation.py         # Data augmentation pipelines
│   ├── ytvis_eval.py           # YTVIS evaluation
│   ├── vps_eval.py             # VPS evaluation
│   ├── vss_eval.py             # VSS evaluation
│   └── datasets/               # Dataset registration and loading
│       ├── builtin.py          # Dataset registry
│       ├── ytvis.py            # YouTube-VIS handler
│       ├── vps.py              # VIPSeg handler
│       └── vss.py              # VSPW handler
└── utils/
    ├── misc.py                 # NestedTensor, utility functions
    └── memory.py               # Memory management decorators

train_net_video.py              # Main training/evaluation entry point
benchmark.py                    # FPS and GFLOPs measurement

configs/                        # YAML configs organized by dataset
├── ytvis19/                    # YouTube-VIS 2019
├── ytvis21/                    # YouTube-VIS 2021
├── ytvis22/                    # YouTube-VIS 2022
├── ovis/                       # OVIS
├── VIPSeg/                     # VIPSeg (panoptic)
└── VSPW/                       # VSPW (semantic)

utils/                          # Standalone evaluation scripts
├── eval_vpq_vspw.py
├── eval_stq_vspw.py
├── eval_miou_vspw.py
├── eval_vc_vspw.py
└── yt2022_evaluate.py

visualization/                  # Demo and visualization tools
├── video_demo.py
├── predictor.py
└── visualizer.py

model_zoo/                      # Model documentation and download links
datasets/                       # Dataset preparation documentation
docs/                           # Project website (HTML/CSS/JS)
```

## Environment Setup

```bash
conda create -n videomt python==3.12.3
conda activate videomt
pip install torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu126
python -m pip install --no-build-isolation 'git+https://github.com/facebookresearch/detectron2.git'
pip install git+https://github.com/cocodataset/panopticapi.git
python3 -m pip install -r requirements.txt
wandb login
```

**Key dependencies:** PyTorch 2.7, torchvision 0.22, detectron2, timm 1.0.20, transformers 4.56.1, wandb, fvcore, pycocotools, einops.

## Common Commands

### Evaluation

```bash
python train_net_video.py \
  --num-gpus 1 \
  --config-file configs/<dataset>/videomt/<config>.yaml \
  --eval-only MODEL.WEIGHTS /path/to/weight.pth \
  MODEL.MODEL.BACKBONE.TEST.WINDOW_SIZE 1 \
  OUTPUT_DIR /path/to/output
```

### Benchmark (FPS / GFLOPs)

```bash
# FPS
python benchmark.py --task fps --config-file <config.yaml> --model-weights <weight.pth> --warmup-iters 100

# GFLOPs (requires disabling fused attention)
export TIMM_FUSED_ATTN=0
python benchmark.py --task flops --config-file <config.yaml> --model-weights <weight.pth>
```

### Visualization

See `model_zoo/visualization.md` and `visualization/video_demo.py`.

## Architecture & Key Concepts

- **Detectron2 framework:** Uses detectron2's registry system (`META_ARCH_REGISTRY`), `CfgNode` config, `DefaultTrainer`, and data loading infrastructure.
- **Three registered meta-architectures:** `videomt`, `videomt_segmenter`, `videomt_online` (in `videomt/videomt.py`).
- **Backbone:** `VidEoMT_CLASS` wraps a timm ViT (DINOv2/DINOv3) with frame-agnostic queries and query propagation across frames.
- **Attention mask annealing:** Progressive suppression of cross-frame attention during training, controlled by `START_STEPS`/`END_STEPS` config.
- **Layer-wise LR decay (LLRD):** Earlier ViT blocks use lower learning rates; configured via `cfg.SOLVER.LLRD`.
- **Two-stage warmup + polynomial decay:** Custom LR scheduler in `modeling/two_stage_warmup_poly_schedule.py`.
- **Hungarian matching:** Bipartite assignment for loss computation in `modeling/matcher.py`.
- **Online inference:** Processes frames sequentially, propagating queries from previous frame without future information.

## Configuration System

YAML configs in `configs/` use detectron2's `CfgNode` with inheritance via `_BASE_`. Key parameters defined in `videomt/config.py`:

| Parameter | Description |
|---|---|
| `MODEL.BACKBONE.IMG_SIZE` | Input image size (640–1024) |
| `MODEL.BACKBONE.NUM_CLASSES` | Number of semantic classes |
| `MODEL.BACKBONE.NUM_Q` | Number of queries |
| `MODEL.BACKBONE.HIDDEN_DIM` | Hidden dimension |
| `MODEL.BACKBONE.MODEL_NAME` | timm model name (e.g. `vit_large_patch14_reg4_dinov2`) |
| `MODEL.BACKBONE.TRACKER_BLOCKS` | ViT blocks used for tracking |
| `MODEL.BACKBONE.SEGMENTER_BLOCKS` | ViT blocks used for segmentation |
| `MODEL.BACKBONE.ATTN_MASK_ANNEALING_ENABLED` | Enable attention mask annealing |
| `SOLVER.LLRD` | Layer-wise learning rate decay factor |
| `SOLVER.POLY_POWER` | Polynomial decay power |
| `DATASETS.DATASET_RATIO` | Sampling ratios for multi-dataset training |

## Code Conventions

- **Classes:** CamelCase (e.g. `VidEoMT_CLASS`, `VideoHungarianMatcher`)
- **Functions/methods:** snake_case (e.g. `batch_dice_loss`, `filter_empty_instances`)
- **Config constants:** UPPER_SNAKE_CASE
- **Copyright headers:** All files include attribution headers citing source repositories (MinVIS, EoMT, CAVIS, Mask2Former, Detectron2)
- **No formal test suite:** Validation is done through evaluation scripts and benchmark runs
- **No linter/formatter configured:** Code generally follows PEP 8
- **Experiment tracking:** wandb integration for logging metrics during training

## Dataset Layout

Datasets are expected under a `datasets/` directory (see `datasets/README.md`):
- YouTube-VIS 2019/2021/2022
- OVIS
- VIPSeg (panoptic video segmentation)
- VSPW (video semantic segmentation)

Each has a dedicated mapper (`dataset_mapper.py`, `dataset_mapper_vps.py`, `dataset_mapper_vss.py`) and evaluator.

## Important Notes for AI Assistants

- This is a **research codebase** — no CI/CD, no automated tests, no pre-commit hooks.
- The project uses **detectron2 heavily** — understand its registry, config, and trainer patterns before modifying core logic.
- **timm** provides the ViT backbone — model names reference timm's model registry.
- Training code is marked as "coming soon" in the README — `train_net_video.py` exists but full training configs/scripts may be incomplete.
- Config overrides can be passed via CLI args to `train_net_video.py` (detectron2 convention).
- The `videomt/` package is imported directly (no `setup.py` / `pip install -e .`), so always run scripts from the repository root.
- License: MIT.
