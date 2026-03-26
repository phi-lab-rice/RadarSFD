import argparse
from pathlib import Path
from typing import Dict

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert a PyTorch checkpoint (.pt) to a safetensors state dict."
    )
    parser.add_argument(
        "checkpoint",
        help="Path to the input checkpoint, for example results/checkpoints/best_checkpoint.pt",
    )
    parser.add_argument(
        "--output",
        help="Optional output path. Defaults to the input path with a .safetensors suffix.",
    )
    return parser.parse_args()


def extract_state_dict(checkpoint: object) -> Dict[str, torch.Tensor]:
    if isinstance(checkpoint, dict) and "unet_state_dict" in checkpoint:
        state_dict = checkpoint["unet_state_dict"]
    elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    elif isinstance(checkpoint, dict):
        state_dict = checkpoint
    else:
        raise TypeError(
            "Unsupported checkpoint format. Expected a state dict or a dict containing "
            "`unet_state_dict` or `state_dict`."
        )

    tensor_state_dict = {}
    for key, value in state_dict.items():
        if not isinstance(key, str):
            raise TypeError(f"State dict key must be a string, got {type(key).__name__!r}.")
        if not torch.is_tensor(value):
            raise TypeError(
                f"State dict entry {key!r} is not a tensor. "
                "This converter exports weights only."
            )
        tensor_state_dict[key] = value.detach().cpu().contiguous()

    if not tensor_state_dict:
        raise ValueError("No tensors were found in the checkpoint state dict.")

    return tensor_state_dict


def build_metadata(checkpoint_path: Path, checkpoint: object) -> Dict[str, str]:
    metadata = {
        "source_checkpoint": str(checkpoint_path.resolve()),
    }

    if isinstance(checkpoint, dict):
        if "epoch" in checkpoint:
            metadata["epoch"] = str(checkpoint["epoch"])
        if "best_val_loss" in checkpoint:
            metadata["best_val_loss"] = str(checkpoint["best_val_loss"])

    return metadata


def main() -> None:
    args = parse_args()
    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    output_path = (
        Path(args.output).expanduser().resolve()
        if args.output
        else checkpoint_path.with_suffix(".safetensors")
    )

    if checkpoint_path.suffix == ".safetensors":
        raise ValueError("Input file is already a .safetensors file.")

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = extract_state_dict(checkpoint)
    metadata = build_metadata(checkpoint_path, checkpoint)

    try:
        from safetensors.torch import save_file
    except ImportError as exc:
        raise ImportError(
            "safetensors is not installed. Run `python -m pip install safetensors` first."
        ) from exc

    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_file(state_dict, str(output_path), metadata=metadata)

    print(f"Loaded checkpoint: {checkpoint_path}")
    print(f"Saved safetensors: {output_path}")
    print(f"Tensor count: {len(state_dict)}")


if __name__ == "__main__":
    main()
