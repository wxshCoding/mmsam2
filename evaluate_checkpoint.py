"""Evaluate a training checkpoint without creating an optimizer or running training."""

import argparse
import logging
import os
from datetime import datetime

import torch
from torch.utils.data import DataLoader

from dataset import (
    POINT_PROMPT_EVAL_POLICY,
    FullDataset_new,
    FullDataset_new_bbox,
    collate_fn_bbox,
    collate_fn_multi_points,
)
# tf32 分支原实现（保留用于方案追踪）：
# from mmsam2 import MMSAM2, get_tf32_compute_policy
# 修改原因：master 是全 FP32 实现，不应依赖 TF32 策略函数。
from mmsam2 import MMSAM2
from train import TASK_CONFIGS, evaluate_valid_sets, load_training_checkpoint, seed_torch

# python evaluate_checkpoint.py   --checkpoint logs/Polyp/2026_08_26_154252/checkpoints/polyp_207_2026_08_26_154458.pth   --task Polyp
def build_parser():
    parser = argparse.ArgumentParser(
        "Evaluate one MMSAM2 training checkpoint",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--checkpoint", required=True, help="Training checkpoint path")
    parser.add_argument("--task", choices=TASK_CONFIGS.keys(), default="Polyp")
    parser.add_argument("--data_path", default=None, help="Override task dataset root")
    parser.add_argument("--valid_list", nargs="+", default=None, help="Override validation subsets")
    # 修改原因：独立验证只允许加载 --checkpoint 指定的训练权重，不再额外读取 SAM2
    # 预训练权重。MMSAM2 当前实现内部使用 CUDA，因此不暴露不可用的 CPU 选项。
    parser.add_argument("--device", default="cuda", choices=["cuda"])
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1024)
    parser.add_argument(
        "--resume_memory_bank",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Restore DMB by default; disabling it changes the evaluated model state",
    )
    parser.add_argument("--boundary_iou_ratio", type=float, default=0.02)
    parser.add_argument("--boundary_f_ratio", type=float, default=0.008)
    parser.add_argument("--nsd_ratio", type=float, default=0.008)
    parser.add_argument("--save_predictions", action="store_true")
    parser.add_argument("--prediction_root", default=None)
    parser.add_argument("--save_feature_vis", action="store_true")
    parser.add_argument("--feature_vis_dir", default="feature_vis_eval")
    parser.add_argument("--log_dir", default="./logs/checkpoint_eval")
    return parser


def create_eval_logger(log_dir):
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y_%m_%d_%H%M%S")
    log_path = os.path.join(log_dir, f"checkpoint_eval_{timestamp}.log")

    logger = logging.getLogger(f"checkpoint_eval.{timestamp}")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    formatter = logging.Formatter("%(asctime)-15s %(message)s")

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger, log_path


def build_validation_loaders(args, logger):
    point_loaders = {}
    bbox_loaders = {}

    for valid_name in args.valid_list:
        try:
            point_dataset = FullDataset_new(
                args.data_path,
                352,
                mode="valid",
                valid_file=valid_name,
            )
            point_loaders[valid_name] = DataLoader(
                point_dataset,
                batch_size=1,
                shuffle=False,
                num_workers=args.num_workers,
                drop_last=False,
                collate_fn=collate_fn_multi_points,
            )
            logger.info(
                f"[Checkpoint Eval] Loaded point subset {valid_name}: "
                f"{len(point_dataset)} images"
            )
        except Exception as exc:
            logger.warning(
                f"[Checkpoint Eval] Failed to load point subset {valid_name}: {exc}"
            )

        try:
            bbox_dataset = FullDataset_new_bbox(
                args.data_path,
                352,
                mode="valid",
                valid_file=valid_name,
            )
            bbox_loaders[valid_name] = DataLoader(
                bbox_dataset,
                batch_size=1,
                shuffle=False,
                num_workers=args.num_workers,
                drop_last=False,
                collate_fn=collate_fn_bbox,
            )
            logger.info(
                f"[Checkpoint Eval] Loaded bbox subset {valid_name}: "
                f"{len(bbox_dataset)} images"
            )
        except Exception as exc:
            logger.warning(
                f"[Checkpoint Eval] Failed to load bbox subset {valid_name}: {exc}"
            )

    return point_loaders, bbox_loaders


def resolve_args(args):
    task_config = TASK_CONFIGS[args.task]
    if args.data_path is None:
        args.data_path = task_config["data_path"]
    if args.valid_list is None:
        args.valid_list = list(task_config["valid_list"])

    if args.save_predictions and args.prediction_root is None:
        timestamp = datetime.now().strftime("%Y_%m_%d_%H%M%S")
        checkpoint_name = os.path.splitext(os.path.basename(args.checkpoint))[0]
        args.prediction_root = os.path.join(
            args.log_dir,
            "predictions",
            f"{checkpoint_name}_{timestamp}",
        )
    return args


def main(args):
    args = resolve_args(args)
    if not os.path.isfile(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")

    seed_torch(args.seed)
    device = torch.device(args.device)
    logger, log_path = create_eval_logger(args.log_dir)
    logger.info("--------------------Checkpoint evaluation starts--------------------")
    logger.info(f"Checkpoint: {args.checkpoint}")
    logger.info(f"Task: {args.task}")
    logger.info(f"Data path: {args.data_path}")
    logger.info(f"Validation subsets: {args.valid_list}")
    logger.info(f"Point prompt evaluation policy: {POINT_PROMPT_EVAL_POLICY}")

    # 修改原因：本脚本只构建模型并恢复推理所需状态，不创建 optimizer/scheduler，
    # 因此不会发生 optimizer.step、scheduler.step 或任何训练更新。
    # 原实现（保留用于方案追踪）：MMSAM2(args.hiera_path, ...)
    # 修改原因：只创建模型结构；模型参数唯一来源是随后加载的训练 checkpoint。
    model = MMSAM2(
        feature_vis_enabled=args.save_feature_vis,
        feature_vis_dir=args.feature_vis_dir,
    ).to(device)
    start_epoch = load_training_checkpoint(
        model=model,
        checkpoint_path=args.checkpoint,
        device=device,
        optimizer=None,
        scheduler=None,
        logger=logger,
        resume_memory_bank=args.resume_memory_bank,
    )
    model.eval()

    # tf32 分支原实现（保留用于方案追踪）：
    # logger.info(f"Compute policy: {get_tf32_compute_policy()}")
    # 修改原因：master 分支使用全 FP32 方案。MMSAM2 初始化时已禁用
    # CUDA matmul/cuDNN TF32，且评估不进入 autocast；
    # 独立验证记录实际运行时开关，便于核对指标计算路径。
    logger.info(
        "Compute policy: full FP32 "
        f"(autocast={torch.is_autocast_enabled()}, "
        f"cuda_matmul_tf32={torch.backends.cuda.matmul.allow_tf32}, "
        f"cudnn_tf32={torch.backends.cudnn.allow_tf32})"
    )
    logger.info(f"Restored DMB memories: {len(model.memory_bank.memories)}")
    if not args.resume_memory_bank:
        logger.warning(
            "DMB restore is disabled; metrics are not directly comparable with checkpoint "
            "metrics produced using the saved DMB."
        )

    point_loaders, bbox_loaders = build_validation_loaders(args, logger)
    if not point_loaders and not bbox_loaders:
        raise RuntimeError("No validation dataset could be loaded")

    # evaluate_valid_sets 内部覆盖 point、bbox、no-prompt 三种模式，并输出所有指标。
    with torch.inference_mode():
        evaluate_valid_sets(
            model=model,
            valid_dataloaders_point=point_loaders,
            valid_dataloaders_bbox=bbox_loaders,
            device=device,
            logger=logger,
            args=args,
            prediction_root=args.prediction_root,
            epoch=start_epoch,
        )

    logger.info("--------------------Checkpoint evaluation finished--------------------")
    logger.info(f"Evaluation log: {log_path}")
    if args.prediction_root is not None:
        logger.info(f"Predictions: {args.prediction_root}")


if __name__ == "__main__":
    main(build_parser().parse_args())
