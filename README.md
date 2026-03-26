# RadarSFD: Single-Frame Diffusion with Pretrained Priors for Radar Point Clouds

This is the repository for ICRA 2026 paper RadarSFD: Single-Frame Diffusion with Pretrained Priors for Radar Point Clouds.

RadarSFD reconstructs dense LiDAR-like point clouds from a single radar frame using conditional latent diffusion with pretrained priors.

## Dataset

The RadarHD dataset used in this project is available at:

https://github.com/akarsh-prabhakara/RadarHD

Before training or evaluation, update `data_root` in [`Code/config.yaml`](/Users/binzhao/Desktop/RadarSFD/Code/config.yaml) so it points to your local RadarHD dataset path.

## Code Structure

The main project code lives in [`Code/`](/Users/binzhao/Desktop/RadarSFD/Code):

- [`Code/train.py`](/Users/binzhao/Desktop/RadarSFD/Code/train.py): training with Hugging Face Accelerate
- [`Code/Eval/eval.py`](/Users/binzhao/Desktop/RadarSFD/Code/Eval/eval.py): inference on the test split and prediction image saving
- [`Code/Eval/record.py`](/Users/binzhao/Desktop/RadarSFD/Code/Eval/record.py): Chamfer Distance (CD) and Modified Hausdorff Distance (MHD) evaluation for saved predictions

## Training

Run training from the repository root:

```bash
accelerate launch Code/train.py --config Code/config.yaml
```

Training configuration is controlled through [`Code/config.yaml`](/Users/binzhao/Desktop/RadarSFD/Code/config.yaml), including dataset path, batch size, mixed precision, and output directory.

Checkpoints are saved to:

```bash
results/checkpoints/best_checkpoint.pt
```

and training logs are written to:

```bash
results/training_log_YYYYMMDD_HHMMSS.txt
```

## Inference

Run evaluation from the repository root:

```bash
python3 Code/Eval/eval.py --config Code/config.yaml --checkpoint results/checkpoints/best_checkpoint.pt
```

This will:

- load the checkpoint specified by `--checkpoint`
- run inference on the test loader defined by the dataset path in `Code/config.yaml`
- save one prediction image for each radar input image

Prediction images are saved under:

```bash
results/eval_predictions/<checkpoint_stem>/
```

For example:

```bash
results/eval_predictions/best_checkpoint/
```

## CD / MHD Evaluation

After inference, use [`Code/Eval/record.py`](/Users/binzhao/Desktop/RadarSFD/Code/Eval/record.py) to compute Chamfer Distance (CD) and Modified Hausdorff Distance (MHD) between the saved predictions and ground-truth lidar images.

The current `record.py` script uses paths set in its `__main__` block, so update:

- `ground_truth_dir`
- `predicted_dirs`

and then run:

```bash
cd Code/Eval
python3 record.py
```

The evaluation script saves the results as a CSV file in:

```bash
Code/Eval/csv/
```

with per-frame `CD` and `MHD` values plus summary statistics printed to the console.
