import argparse
from datetime import datetime
import json
import os
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchmetrics.image import StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from tqdm.auto import tqdm

from dataloader import create_data_loaders, neg_pos_to_zero_one
from unet import marigoldUnet
from vae import VAE

DEFAULT_CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.yaml")


def load_config(config_path: str) -> Dict:
    with open(config_path, "r", encoding="utf-8") as handle:
        config = json.load(handle)

    if not isinstance(config, dict):
        raise ValueError(f"Config file must contain a top-level mapping: {config_path}")

    if config.get("device", "auto") == "auto":
        config["device"] = "cuda" if torch.cuda.is_available() else "cpu"

    return config


def _resolve_mixed_precision(config: Dict) -> str:
    if torch.cuda.is_available() and config.get("use_fp16", False):
        return "fp16"
    return "no"


def create_accelerator(config: Dict) -> Accelerator:
    ddp_kwargs = DistributedDataParallelKwargs(
        find_unused_parameters=config.get("find_unused_parameters", False)
    )
    return Accelerator(
        gradient_accumulation_steps=config["gradient_accumulation_steps"],
        mixed_precision=_resolve_mixed_precision(config),
        kwargs_handlers=[ddp_kwargs],
    )


class DiffusionModelTrainer:
    def __init__(self, config: Dict, accelerator: Accelerator):
        self.config = config
        self.accelerator = accelerator
        self.device = accelerator.device
        self.best_val_loss = float("inf")
        self.log_path = None

        self._create_output_dirs()
        self._init_models()
        self._init_dataloaders()
        self._init_optimizer_and_scheduler()
        self._init_losses()
        self._prepare_training_components()
        self._init_log_file()

    def _create_output_dirs(self) -> None:
        if self.accelerator.is_main_process:
            os.makedirs(self.config["output_dir"], exist_ok=True)
            os.makedirs(os.path.join(self.config["output_dir"], "samples"), exist_ok=True)
            os.makedirs(os.path.join(self.config["output_dir"], "checkpoints"), exist_ok=True)
        self.accelerator.wait_for_everyone()

    def _init_log_file(self) -> None:
        if not self.accelerator.is_main_process:
            return

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_path = os.path.join(
            self.config["output_dir"],
            f"training_log_{timestamp}.txt",
        )
        with open(self.log_path, "w", encoding="utf-8") as handle:
            handle.write(f"Training log created: {timestamp}\n")
            handle.write(f"Output directory: {os.path.abspath(self.config['output_dir'])}\n")
            handle.write("epoch,train_total,train_noise,train_perceptual,val_total,val_noise,val_perceptual,lr\n")
        self.accelerator.print(f"Writing training log to {self.log_path}")

    def _init_models(self) -> None:
        self.vae = VAE(self.device, torch.float32, mode=self.config["vae_mode"])
        self.unet = marigoldUnet(
            use_pretrained=True,
            device=self.device,
            torch_dtype=torch.float32,
        ).unet
        self.unet.enable_gradient_checkpointing()

    def _init_dataloaders(self) -> None:
        train_dataloader, val_dataloader, test_dataloader = create_data_loaders(
            data_root=self.config["data_root"],
            batch_size=self.config["batch_size"],
            num_workers=self.config["num_workers"],
            shuffle_train=True,
            pin_memory=torch.cuda.is_available(),
            validation_split=self.config["validation_split"],
        )
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.test_dataloader = test_dataloader

    def _init_optimizer_and_scheduler(self) -> None:
        params_to_optimize = [p for p in self.unet.parameters() if p.requires_grad]
        self.optimizer = torch.optim.AdamW(
            params_to_optimize,
            lr=self.config["learning_rate"],
            betas=(self.config["adam_beta1"], self.config["adam_beta2"]),
            weight_decay=self.config["adam_weight_decay"],
            eps=self.config["adam_epsilon"],
        )
        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=self.config["num_train_timesteps"],
            beta_start=self.config["beta_start"],
            beta_end=self.config["beta_end"],
            beta_schedule=self.config["beta_schedule"],
            prediction_type=self.config["prediction_type"],
        )
        self.lr_scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            factor=0.5,
            patience=3,
        )

    def _init_losses(self) -> None:
        self.noise_loss = nn.MSELoss()
        self.l1_loss = nn.L1Loss()
        self.ssim_loss = None
        self.lpips_loss = None

        if self.config["use_perceptual_loss"]:
            self.ssim_loss = StructuralSimilarityIndexMeasure(data_range=1.0).to(self.device)
            self.lpips_loss = LearnedPerceptualImagePatchSimilarity(net_type="vgg").to(self.device)
            for param in self.lpips_loss.parameters():
                param.requires_grad = False

    def _prepare_training_components(self) -> None:
        self.unet, self.optimizer, self.train_dataloader, self.val_dataloader = self.accelerator.prepare(
            self.unet,
            self.optimizer,
            self.train_dataloader,
            self.val_dataloader,
        )

    def encode_to_latent(self, image: torch.Tensor) -> torch.Tensor:
        return self.vae.encode_latent(image)

    def decode_latent(self, latent: torch.Tensor):
        return self.vae.decode_latent(latent)

    def _empty_metric_totals(self) -> Dict[str, torch.Tensor]:
        return {
            "total_loss": torch.zeros(1, device=self.device),
            "noise_loss": torch.zeros(1, device=self.device),
            "perceptual_loss": torch.zeros(1, device=self.device),
            "l1_loss": torch.zeros(1, device=self.device),
            "ssim_loss": torch.zeros(1, device=self.device),
            "lpips_loss": torch.zeros(1, device=self.device),
        }

    def _compute_losses(self, radar_images: torch.Tensor, lidar_images: torch.Tensor) -> Dict[str, torch.Tensor]:
        radar_latents = self.encode_to_latent(radar_images)
        lidar_latents = self.encode_to_latent(lidar_images)

        noise = torch.randn_like(lidar_latents)
        timesteps = torch.randint(
            0,
            self.config["num_train_timesteps"],
            (lidar_latents.shape[0],),
            device=self.device,
        ).long()
        noisy_lidar_latents = self.noise_scheduler.add_noise(lidar_latents, noise, timesteps)

        resized_radar_latents = F.interpolate(
            radar_latents,
            size=noisy_lidar_latents.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        concatenated_latents = torch.cat([noisy_lidar_latents, resized_radar_latents], dim=1)
        null_text_embeddings = torch.zeros(
            (lidar_latents.shape[0], 77, 1024),
            device=self.device,
            dtype=concatenated_latents.dtype,
        )

        predicted_noise = self.unet(
            sample=concatenated_latents,
            timestep=timesteps,
            encoder_hidden_states=null_text_embeddings,
            return_dict=False,
        )[0]
        noise_loss = self.noise_loss(predicted_noise.float(), noise.float())

        l1_loss = torch.zeros(1, device=self.device)
        ssim_loss = torch.zeros(1, device=self.device)
        lpips_loss = torch.zeros(1, device=self.device)
        perceptual_loss = torch.zeros(1, device=self.device)
        total_loss = noise_loss

        if self.config["use_perceptual_loss"]:
            pred_clean_latents = []
            for index in range(lidar_latents.shape[0]):
                step_output = self.noise_scheduler.step(
                    model_output=predicted_noise[index:index + 1],
                    timestep=int(timesteps[index].item()),
                    sample=noisy_lidar_latents[index:index + 1],
                )
                pred_clean_latents.append(step_output.pred_original_sample)

            pred_clean_latents = torch.cat(pred_clean_latents, dim=0)
            reconstructed_lidar_zero_one, reconstructed_lidar_neg_pos = self.decode_latent(pred_clean_latents)

            l1_loss = self.l1_loss(reconstructed_lidar_neg_pos.float(), lidar_images.float())
            ssim_loss = 1 - self.ssim_loss(
                reconstructed_lidar_zero_one.float(),
                neg_pos_to_zero_one(lidar_images.float()),
            )
            lpips_loss = self.lpips_loss(
                reconstructed_lidar_neg_pos.float(),
                lidar_images.float(),
            )
            perceptual_loss = (
                self.config["loss_weight_l1"] * l1_loss
                + self.config["loss_weight_ssim"] * ssim_loss
                + self.config["loss_weight_lpips"] * lpips_loss
            )
            total_loss = 0.5 * noise_loss + 0.5 * perceptual_loss

        return {
            "total_loss": total_loss,
            "noise_loss": noise_loss,
            "perceptual_loss": perceptual_loss,
            "l1_loss": l1_loss,
            "ssim_loss": ssim_loss,
            "lpips_loss": lpips_loss,
        }

    def _reduce_metric_totals(
        self,
        totals: Dict[str, torch.Tensor],
        num_batches: int,
    ) -> Dict[str, float]:
        reduced_batches = self.accelerator.reduce(
            torch.tensor(float(num_batches), device=self.device),
            reduction="sum",
        )
        batch_count = max(int(reduced_batches.item()), 1)
        return {
            name: (
                self.accelerator.reduce(value.detach(), reduction="sum").item() / batch_count
            )
            for name, value in totals.items()
        }

    def _run_epoch(self, epoch: int, training: bool) -> Dict[str, float]:
        self.unet.train(training)
        totals = self._empty_metric_totals()
        num_batches = 0
        dataloader = self.train_dataloader if training else self.val_dataloader

        desc = f"Train {epoch + 1}" if training else f"Val {epoch + 1}"
        progress_bar = tqdm(
            dataloader,
            desc=desc,
            disable=not (self.accelerator.is_local_main_process and self.config["enable_progress_bar"]),
        )

        if training:
            self.optimizer.zero_grad(set_to_none=True)

        for step, (radar_images, lidar_images) in enumerate(progress_bar):
            num_batches += 1

            if training:
                with self.accelerator.accumulate(self.unet):
                    with self.accelerator.autocast():
                        metrics = self._compute_losses(radar_images, lidar_images)

                    self.accelerator.backward(metrics["total_loss"])

                    if self.accelerator.sync_gradients:
                        self.accelerator.clip_grad_norm_(self.unet.parameters(), self.config["max_grad_norm"])

                    self.optimizer.step()
                    self.optimizer.zero_grad(set_to_none=True)
            else:
                with torch.no_grad():
                    with self.accelerator.autocast():
                        metrics = self._compute_losses(radar_images, lidar_images)

            for name, value in metrics.items():
                totals[name] += value.detach().reshape(1)

            if self.accelerator.is_local_main_process:
                progress_bar.set_postfix(
                    loss=f"{metrics['total_loss'].detach().item():.4f}",
                    acc_step=f"{(step % self.accelerator.gradient_accumulation_steps) + 1}/{self.accelerator.gradient_accumulation_steps}",
                )

        metrics = self._reduce_metric_totals(totals, num_batches)
        phase = "Training" if training else "Validation"
        self.accelerator.print(
            f"{phase} epoch {epoch + 1}: total={metrics['total_loss']:.4f}, "
            f"noise={metrics['noise_loss']:.4f}, perceptual={metrics['perceptual_loss']:.4f}"
        )
        if self.config["use_perceptual_loss"]:
            self.accelerator.print(
                f"  perceptual parts: l1={metrics['l1_loss']:.4f}, "
                f"ssim={metrics['ssim_loss']:.4f}, lpips={metrics['lpips_loss']:.4f}"
            )
        return metrics

    def _append_epoch_log(self, epoch: int, train_metrics: Dict[str, float], val_metrics: Dict[str, float]) -> None:
        if not self.accelerator.is_main_process or self.log_path is None:
            return

        with open(self.log_path, "a", encoding="utf-8") as handle:
            handle.write(
                f"{epoch + 1},"
                f"{train_metrics['total_loss']:.6f},"
                f"{train_metrics['noise_loss']:.6f},"
                f"{train_metrics['perceptual_loss']:.6f},"
                f"{val_metrics['total_loss']:.6f},"
                f"{val_metrics['noise_loss']:.6f},"
                f"{val_metrics['perceptual_loss']:.6f},"
                f"{self.optimizer.param_groups[0]['lr']:.8f}\n"
            )

    def _checkpoint_payload(self, epoch: int) -> Dict:
        state_dict = self.accelerator.get_state_dict(self.unet)
        return {
            "epoch": epoch + 1,
            "unet_state_dict": {key: value.detach().cpu() for key, value in state_dict.items()},
            "config": self.config,
            "best_val_loss": self.best_val_loss,
        }

    def save_best_checkpoint(self, epoch: int, val_loss: float) -> None:
        self.accelerator.wait_for_everyone()
        if not self.accelerator.is_main_process or val_loss >= self.best_val_loss:
            return

        self.best_val_loss = val_loss
        best_checkpoint_path = os.path.join(
            self.config["output_dir"],
            "checkpoints",
            "best_checkpoint.pt",
        )
        self.accelerator.save(self._checkpoint_payload(epoch), best_checkpoint_path)
        self.accelerator.print(
            f"Saved best checkpoint (val_loss: {self.best_val_loss:.4f}) to {best_checkpoint_path}"
        )

    def _save_sample(self, epoch: int) -> None:
        self.accelerator.wait_for_everyone()
        if not self.accelerator.is_main_process:
            return

        unwrapped_unet = self.accelerator.unwrap_model(self.unet)
        unwrapped_unet.eval()

        with torch.no_grad():
            with self.accelerator.autocast():
                radar_images, lidar_images = next(iter(self.test_dataloader))
                actual_num_samples = min(radar_images.shape[0], self.config["num_samples"])
                radar_images = radar_images[:actual_num_samples].to(self.device)
                lidar_images = lidar_images[:actual_num_samples].to(self.device)

                radar_latents = self.encode_to_latent(radar_images)
                lidar_latents = self.encode_to_latent(lidar_images)
                resized_radar_latents = F.interpolate(
                    radar_latents,
                    size=lidar_latents.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )

                inference_scheduler = DDIMScheduler(
                    num_train_timesteps=self.config["num_train_timesteps"],
                    beta_start=self.config["beta_start"],
                    beta_end=self.config["beta_end"],
                    beta_schedule=self.config["beta_schedule"],
                )
                inference_scheduler.set_timesteps(50)

                latent = torch.randn_like(lidar_latents)
                null_text_embeddings = torch.zeros(
                    (lidar_latents.shape[0], 77, 1024),
                    device=self.device,
                    dtype=latent.dtype,
                )

                sample_progress = tqdm(
                    inference_scheduler.timesteps,
                    desc="Sampling",
                    leave=False,
                    disable=not self.config["enable_progress_bar"],
                )
                for timestep in sample_progress:
                    concatenated_latents = torch.cat([latent, resized_radar_latents], dim=1)
                    noise_pred = unwrapped_unet(
                        sample=concatenated_latents,
                        timestep=timestep,
                        encoder_hidden_states=null_text_embeddings,
                        return_dict=False,
                    )[0]
                    latent = inference_scheduler.step(
                        model_output=noise_pred,
                        timestep=int(timestep.item()) if isinstance(timestep, torch.Tensor) else int(timestep),
                        sample=latent,
                    ).prev_sample

                reconstructed_images_zero_one, _ = self.decode_latent(latent)

        radar_gt_vis = neg_pos_to_zero_one(radar_images).mean(dim=1, keepdim=True)
        lidar_gt_vis = neg_pos_to_zero_one(lidar_images).mean(dim=1, keepdim=True)
        reconstructed_vis = reconstructed_images_zero_one.mean(dim=1, keepdim=True)
        radar_gt_vis_resized = F.interpolate(
            radar_gt_vis,
            size=lidar_gt_vis.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

        grid = torch.cat([radar_gt_vis_resized, reconstructed_vis, lidar_gt_vis], dim=0)
        grid_image = torchvision.utils.make_grid(grid, nrow=actual_num_samples)
        save_path = os.path.join(
            self.config["output_dir"],
            "samples",
            f"test-sample-e{epoch + 1}.png",
        )
        torchvision.utils.save_image(grid_image, save_path)
        self.accelerator.print(f"Saved sample images to {save_path}")

    def train(self) -> None:
        for epoch in range(self.config["num_epochs"]):
            self.accelerator.print(f"\nStarting epoch {epoch + 1}/{self.config['num_epochs']}")
            train_metrics = self._run_epoch(epoch, training=True)
            val_metrics = self._run_epoch(epoch, training=False)

            self.lr_scheduler.step(val_metrics["total_loss"])
            self._append_epoch_log(epoch, train_metrics, val_metrics)
            self.save_best_checkpoint(epoch, val_metrics["total_loss"])
            self._save_sample(epoch)

        self.accelerator.wait_for_everyone()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default=DEFAULT_CONFIG_PATH,
        help="Path to the YAML config file.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    accelerator = create_accelerator(config)
    accelerator.print(f"Loaded config from: {os.path.abspath(args.config)}")
    accelerator.print(f"Using config: {config}")
    accelerator.print(
        f"Accelerate setup: processes={accelerator.num_processes}, "
        f"device={accelerator.device}, mixed_precision={accelerator.mixed_precision}"
    )

    trainer = DiffusionModelTrainer(config, accelerator)
    trainer.train()


if __name__ == "__main__":
    main()
