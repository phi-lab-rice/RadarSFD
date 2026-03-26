---
license: cc-by-nc-4.0
language:
  - en
tags:
  - radar
  - lidar
  - diffusion
  - robotics
  - point-cloud
  - image-to-image
base_model:
  - prs-eth/marigold-depth-v1-1
---

# RadarSFD

RadarSFD reconstructs dense LiDAR-like point cloud representations from a single radar frame using conditional latent diffusion with pretrained priors. This repository contains the released RadarSFD checkpoint for the paper *RadarSFD: Single-Frame Diffusion with Pretrained Priors for Radar Point Clouds*.

The released weights are intended to be used together with the official project code:

- GitHub: https://github.com/phi-lab-rice/RadarSFD

## Model description

RadarSFD conditions a latent diffusion denoiser on a single radar range-azimuth BEV image and predicts a dense LiDAR-like BEV reconstruction. The model transfers geometric priors from a pretrained Marigold U-Net and uses latent-space conditioning from the radar input during denoising.

In the current release, the uploaded checkpoint corresponds to the trained U-Net weights used by the project codebase.

## Intended use

This model is intended for:

- research on radar-based 3D perception
- single-frame radar-to-LiDAR-like reconstruction
- benchmarking on the RadarHD dataset
- qualitative and quantitative evaluation using the official repository

This model is not intended as a plug-and-play general-purpose Hugging Face inference API model. It is designed to be loaded by the RadarSFD codebase for evaluation and research experiments.

## Input and output

Input:

- a single radar BEV image from the RadarHD-style preprocessing pipeline
- grayscale radar input repeated to 3 channels by the project dataloader

Output:

- a reconstructed LiDAR-like BEV image
- downstream evaluation artifacts produced by the repository evaluation scripts

Within the official codebase, radar images are loaded as grayscale images and repeated to 3 channels. The training pipeline pairs radar and LiDAR BEV images from the RadarHD dataset.

## How to use

Clone the official repository and follow its environment setup and evaluation instructions:

```bash
git clone https://github.com/phi-lab-rice/RadarSFD.git
cd RadarSFD
```

Place the checkpoint in your desired location, update the dataset path in `Code/config.yaml`, and run evaluation with:

```bash
python3 Code/Eval/eval.py --config Code/config.yaml --checkpoint /path/to/RadarSFD.safetensors
```

Please refer to the repository for the expected dataset structure, environment dependencies, and evaluation workflow.

## Training data

RadarSFD is trained and evaluated on the RadarHD dataset:

- https://github.com/akarsh-prabhakara/RadarHD

The project uses paired radar and LiDAR BEV images for training, validation, and testing.

## Architecture and training

- Backbone initialization: `prs-eth/marigold-depth-v1-1`
- VAE mode in the released code: TAESD (`madebyollin/taesd`)
- Diffusion scheduler for training: DDPM
- Inference scheduler in evaluation: DDIM
- Training objective: latent diffusion with additional perceptual losses in image space

## Limitations

- This release is research-oriented and has only been validated within the official RadarSFD codebase.
- Performance depends on using the same preprocessing and dataset conventions as the training setup.
- The model is designed for single-frame radar reconstruction and does not use temporal accumulation or SAR.
- Outputs are LiDAR-like BEV reconstructions rather than fully post-processed 3D point clouds ready for deployment.

## License

This model is released under the `CC-BY-NC-4.0` license. Please review the dataset and upstream model licenses as well before use.

## Citation

If you use this model, please cite:

```bibtex
@inproceedings{zhao2026radarsfd,
  title     = {RadarSFD: Single-Frame Diffusion with Pretrained Priors for Radar Point Clouds},
  author    = {Zhao, Bin and Garg, Nakul},
  booktitle = {IEEE International Conference on Robotics and Automation (ICRA)},
  year      = {2026}
}
```
