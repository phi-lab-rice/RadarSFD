import argparse
import json
import os
import sys
from contextlib import nullcontext
from datetime import datetime
from pathlib import Path
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchmetrics.image import StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from tqdm.auto import tqdm

CURRENT_DIR = Path(__file__).resolve().parent
CODE_DIR = CURRENT_DIR.parent
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from dataloader import RadarLidarDataset, neg_pos_to_zero_one
from unet import marigoldUnet
from vae import VAE

DEFAULT_CONFIG_PATH = str(CODE_DIR / "config.yaml")


def load_config(config_path: str) -> Dict:
    with open(config_path, "r", encoding="utf-8") as handle:
        config = json.load(handle)

    if not isinstance(config, dict):
        raise ValueError(f"Config file must contain a top-level mapping: {config_path}")

    if config.get("device", "auto") == "auto":
        config["device"] = "cuda" if torch.cuda.is_available() else "cpu"

    return config


def resolve_device(config: Dict) -> torch.device:
    device_name = config.get("device", "cpu")
    if device_name.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested for evaluation but is not available.")
    return torch.device(device_name)


def autocast_context(device: torch.device, config: Dict):
    if device.type == "cuda" and config.get("use_fp16", False):
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return nullcontext()


class EvalDataset(Dataset):
    def __init__(self, data_root: str):
        self.dataset = RadarLidarDataset(data_root=data_root, split="test")

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int):
        radar_tensor, lidar_tensor = self.dataset[idx]
        sample_info = self.dataset.get_sample_info(idx)
        return radar_tensor, lidar_tensor, sample_info["radar_filename"]


class DiffusionModelEvaluator:
    def __init__(self, config: Dict, checkpoint_path: str):
        self.config = config
        self.checkpoint_path = os.path.abspath(checkpoint_path)
        self.device = resolve_device(config)
        self.metrics_log_path = None

        self._create_output_dirs()
        self._init_models()
        self._load_checkpoint()
        self._init_dataloader()
        self._init_losses()
        self._init_metrics_log()

    def _create_output_dirs(self) -> None:
        checkpoint_stem = Path(self.checkpoint_path).stem
        self.prediction_dir = os.path.join(
            self.config["output_dir"],
            "eval_predictions",
            checkpoint_stem,
        )
        os.makedirs(self.prediction_dir, exist_ok=True)

    def _init_models(self) -> None:
        unet_type = self.config.get("unet_type", "marigold").lower()
        if unet_type != "marigold":
            raise NotImplementedError(
                f"eval.py currently matches train.py's marigold path only, got unet_type={unet_type!r}."
            )

        self.vae = VAE(self.device, torch.float32, mode=self.config["vae_mode"])
        self.unet = marigoldUnet(
            use_pretrained=False,
            device=self.device,
            torch_dtype=torch.float32,
        ).unet
        self.unet.eval()

    def _load_checkpoint(self) -> None:
        checkpoint = torch.load(self.checkpoint_path, map_location="cpu")
        if isinstance(checkpoint, dict) and "unet_state_dict" in checkpoint:
            state_dict = checkpoint["unet_state_dict"]
        elif isinstance(checkpoint, dict):
            state_dict = checkpoint
        else:
            raise TypeError("Checkpoint must be a state dict or a dict containing `unet_state_dict`.")

        missing_keys, unexpected_keys = self.unet.load_state_dict(state_dict, strict=False)
        print(f"Loaded checkpoint from {self.checkpoint_path}")
        print(
            f"Checkpoint load summary: missing_keys={len(missing_keys)}, unexpected_keys={len(unexpected_keys)}"
        )
        if missing_keys:
            print(f"Missing key preview: {', '.join(missing_keys[:8])}")
        if unexpected_keys:
            print(f"Unexpected key preview: {', '.join(unexpected_keys[:8])}")

    def _init_dataloader(self) -> None:
        self.test_dataset = EvalDataset(data_root=self.config["data_root"])
        self.test_dataloader = DataLoader(
            self.test_dataset,
            batch_size=self.config["batch_size"],
            shuffle=False,
            num_workers=self.config["num_workers"],
            pin_memory=self.device.type == "cuda",
            drop_last=False,
        )

    def _init_losses(self) -> None:
        self.l1_loss = nn.L1Loss()
        self.ssim_loss = StructuralSimilarityIndexMeasure(data_range=1.0).to(self.device)
        self.lpips_loss = LearnedPerceptualImagePatchSimilarity(net_type="vgg").to(self.device)
        for param in self.lpips_loss.parameters():
            param.requires_grad = False

        self.inference_scheduler = DDIMScheduler(
            num_train_timesteps=self.config["num_train_timesteps"],
            beta_start=self.config["beta_start"],
            beta_end=self.config["beta_end"],
            beta_schedule=self.config["beta_schedule"],
        )
        self.inference_scheduler.set_timesteps(self.config.get("num_inference_steps", 50))

    def _init_metrics_log(self) -> None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.metrics_log_path = os.path.join(
            self.prediction_dir,
            f"eval_log_{timestamp}.txt",
        )
        with open(self.metrics_log_path, "w", encoding="utf-8") as handle:
            handle.write(f"Checkpoint: {self.checkpoint_path}\n")
            handle.write("l1,ssim,lpips,perceptual\n")

    def encode_to_latent(self, image: torch.Tensor) -> torch.Tensor:
        return self.vae.encode_latent(image)

    def decode_latent(self, latent: torch.Tensor):
        return self.vae.decode_latent(latent)

    def _empty_metric_totals(self) -> Dict[str, float]:
        return {
            "l1_loss": 0.0,
            "ssim_loss": 0.0,
            "lpips_loss": 0.0,
            "perceptual_loss": 0.0,
        }

    def _sample_batch(self, radar_images: torch.Tensor, lidar_images: torch.Tensor):
        radar_latents = self.encode_to_latent(radar_images)
        lidar_latents = self.encode_to_latent(lidar_images)
        resized_radar_latents = F.interpolate(
            radar_latents,
            size=lidar_latents.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

        latent = torch.randn_like(lidar_latents)
        null_text_embeddings = torch.zeros(
            (lidar_latents.shape[0], 77, 1024),
            device=self.device,
            dtype=latent.dtype,
        )

        for timestep in self.inference_scheduler.timesteps:
            concatenated_latents = torch.cat([latent, resized_radar_latents], dim=1)
            noise_pred = self.unet(
                sample=concatenated_latents,
                timestep=timestep,
                encoder_hidden_states=null_text_embeddings,
                return_dict=False,
            )[0]
            latent = self.inference_scheduler.step(
                model_output=noise_pred,
                timestep=int(timestep.item()) if isinstance(timestep, torch.Tensor) else int(timestep),
                sample=latent,
            ).prev_sample

        reconstructed_zero_one, reconstructed_neg_pos = self.decode_latent(latent)
        l1_loss = self.l1_loss(reconstructed_neg_pos.float(), lidar_images.float())
        ssim_loss = 1 - self.ssim_loss(
            reconstructed_zero_one.float(),
            neg_pos_to_zero_one(lidar_images.float()),
        )
        lpips_loss = self.lpips_loss(
            reconstructed_neg_pos.float(),
            lidar_images.float(),
        )
        perceptual_loss = (
            self.config["loss_weight_l1"] * l1_loss
            + self.config["loss_weight_ssim"] * ssim_loss
            + self.config["loss_weight_lpips"] * lpips_loss
        )

        metrics = {
            "l1_loss": float(l1_loss.item()),
            "ssim_loss": float(ssim_loss.item()),
            "lpips_loss": float(lpips_loss.item()),
            "perceptual_loss": float(perceptual_loss.item()),
        }
        return metrics, reconstructed_zero_one

    def _save_predictions(self, reconstructed_zero_one: torch.Tensor, radar_filenames) -> None:
        pred_uint8 = (255 * reconstructed_zero_one.mean(dim=1).clamp(0, 1)).to(torch.uint8)
        for pred_image, radar_filename in zip(pred_uint8, radar_filenames):
            save_name = f"pred_{radar_filename}"
            save_path = os.path.join(self.prediction_dir, save_name)
            Image.fromarray(pred_image.cpu().numpy()).save(save_path)

    def _append_metrics_log(self, metrics: Dict[str, float]) -> None:
        with open(self.metrics_log_path, "a", encoding="utf-8") as handle:
            handle.write(
                f"{metrics['l1_loss']:.6f},"
                f"{metrics['ssim_loss']:.6f},"
                f"{metrics['lpips_loss']:.6f},"
                f"{metrics['perceptual_loss']:.6f}\n"
            )

    def evaluate(self) -> None:
        totals = self._empty_metric_totals()
        num_batches = 0

        progress_bar = tqdm(
            self.test_dataloader,
            desc="Eval",
            disable=not self.config["enable_progress_bar"],
        )

        with torch.no_grad():
            for radar_images, lidar_images, radar_filenames in progress_bar:
                radar_images = radar_images.to(self.device)
                lidar_images = lidar_images.to(self.device)
                num_batches += 1

                with autocast_context(self.device, self.config):
                    metrics, reconstructed_zero_one = self._sample_batch(radar_images, lidar_images)

                for name, value in metrics.items():
                    totals[name] += value

                self._save_predictions(reconstructed_zero_one, radar_filenames)
                progress_bar.set_postfix(perceptual=f"{metrics['perceptual_loss']:.4f}")

        reduced_metrics = {
            name: value / max(num_batches, 1)
            for name, value in totals.items()
        }
        print(
            f"Eval metrics: l1={reduced_metrics['l1_loss']:.4f}, "
            f"ssim={reduced_metrics['ssim_loss']:.4f}, "
            f"lpips={reduced_metrics['lpips_loss']:.4f}, "
            f"perceptual={reduced_metrics['perceptual_loss']:.4f}"
        )
        self._append_metrics_log(reduced_metrics)
        print(f"Saved prediction images to {self.prediction_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default=DEFAULT_CONFIG_PATH,
        help="Path to the config file.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to the checkpoint file to evaluate.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    checkpoint_path = args.checkpoint or os.path.join(
        config["output_dir"],
        "checkpoints",
        "best_checkpoint.pt",
    )
    device = resolve_device(config)

    print(f"Loaded config from: {os.path.abspath(args.config)}")
    print(f"Using checkpoint: {os.path.abspath(checkpoint_path)}")
    print(f"Evaluation device: {device}")
    print(f"Mixed precision fp16 enabled: {device.type == 'cuda' and config.get('use_fp16', False)}")

    evaluator = DiffusionModelEvaluator(config, checkpoint_path)
    evaluator.evaluate()


if __name__ == "__main__":
    main()
