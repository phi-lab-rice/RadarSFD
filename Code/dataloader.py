import os
import glob
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms as transforms
from typing import Optional, Tuple, Callable, List


class RadarLidarDataset(Dataset):
    """
    Dataset for radar-lidar image pairs.
    Radar images are 64×256, lidar images are 512×256 (original dimensions preserved).
    For the image pair, the range, azimuth, and intensity arranged into an image with range (0-10 m) along rows,
    and azimuth (-90◦ to 90◦) along columns

    Args:
        data_root (str): Root directory containing the dataset
        split (str): Either 'train' or 'test'
    """

    def __init__(self, data_root: Optional[str] = None, split: str = "train"):
        # Auto-detect data root if not provided
        if data_root is None:
            # Try relative paths from both LDM directory and root directory
            possible_paths = ["dataset_5/"]

            for path in possible_paths:
                test_path = os.path.join(path, split, "radar")
                if os.path.exists(test_path):
                    data_root = path
                    break

            if data_root is None:
                raise FileNotFoundError(
                    f"Could not find data directory. Tried: {possible_paths}"
                )

        self.data_root = data_root
        self.split = split

        # Set up paths
        self.radar_dir = os.path.join(data_root, split, "radar")
        self.lidar_dir = os.path.join(data_root, split, "lidar")

        # Verify directories exist
        if not os.path.exists(self.radar_dir):
            raise FileNotFoundError(f"Radar directory not found: {self.radar_dir}")
        if not os.path.exists(self.lidar_dir):
            raise FileNotFoundError(f"Lidar directory not found: {self.lidar_dir}")

        # Get radar-lidar pairs
        self.pairs = self._get_paired_files()

        # Set up transform (convert to tensor and normalize)
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                lambda x: x.repeat(3, 1, 1),
                transforms.Normalize((0.5,), (0.5,)),
            ]
        )

    def _get_paired_files(self) -> List[Tuple[str, str]]:
        """Find matching radar-lidar file pairs."""
        radar_files = glob.glob(os.path.join(self.radar_dir, "R_*.png"))
        pairs = []

        for radar_file in radar_files:
            # Extract the identifier from radar filename (e.g., "140_932" from "R_140_932.png")
            radar_basename = os.path.basename(radar_file)
            identifier = radar_basename[2:-4]  # Remove "R_" prefix and ".png" suffix

            # Construct corresponding lidar filename
            lidar_file = os.path.join(self.lidar_dir, f"L_{identifier}.png")

            if os.path.exists(lidar_file):
                pairs.append((radar_file, lidar_file))

        if len(pairs) == 0:
            raise ValueError(
                f"No valid radar-lidar pairs found in {self.radar_dir} and {self.lidar_dir}"
            )

        return sorted(pairs)

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int):
        """
        Get a radar-lidar pair.

        Returns:
            tuple: (radar_image, lidar_image) as torch tensors
        """
        radar_path, lidar_path = self.pairs[idx]

        try:
            # Load images as grayscale (single channel)
            radar_img = Image.open(radar_path).convert("L")  # Convert to grayscale
            lidar_img = Image.open(lidar_path).convert("L")  # Convert to grayscale

            # Apply transform
            radar_tensor = self.transform(radar_img)
            lidar_tensor = self.transform(lidar_img)

            return radar_tensor, lidar_tensor

        except Exception as e:
            print(f"Error loading pair {idx}: {radar_path}, {lidar_path}")
            print(f"Error details: {str(e)}")
            raise

    def get_sample_info(self, idx: int) -> dict:
        """Get information about a specific sample."""
        radar_path, lidar_path = self.pairs[idx]
        return {
            "index": idx,
            "radar_path": radar_path,
            "lidar_path": lidar_path,
            "radar_filename": os.path.basename(radar_path),
            "lidar_filename": os.path.basename(lidar_path),
        }


def create_data_loaders(
    data_root: Optional[str] = None,
    batch_size: int = 16,
    num_workers: int = 4,
    shuffle_train: bool = True,
    pin_memory: bool = True,
    validation_split: float = 0.1,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train, validation, and test data loaders.

    Args:
        data_root (str): Root directory containing the dataset
        batch_size (int): Batch size for data loaders
        num_workers (int): Number of worker processes for data loading
        shuffle_train (bool): Whether to shuffle training data
        pin_memory (bool): Whether to pin memory for faster GPU transfer
        validation_split (float): The fraction of training data to use for validation.

    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """

    # Create datasets
    full_train_dataset = RadarLidarDataset(data_root=data_root, split="train")

    test_dataset = RadarLidarDataset(data_root=data_root, split="test")

    # Split training data into train and validation
    num_train = len(full_train_dataset)
    val_size = int(validation_split * num_train)
    train_size = num_train - val_size

    train_dataset, val_dataset = random_split(
        full_train_dataset, [train_size, val_size]
    )

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle_train,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,  # Drop last incomplete batch for training
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,  # No shuffling for test set
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,  # Keep all test samples
    )

    print(f"Created data loaders:")
    print(f"  Train: {len(train_dataset)} samples, {len(train_loader)} batches")
    print(f"  Validation: {len(val_dataset)} samples, {len(val_loader)} batches")
    print(f"  Test: {len(test_dataset)} samples, {len(test_loader)} batches")

    return train_loader, val_loader, test_loader


def repeat_to_3channel(x: torch.Tensor) -> torch.Tensor:
    """
    Repeat a 1-channel image tensor to 3 channels.
    Args:
        x: Tensor of shape (B, 1, H, W)
    Returns:
        Tensor of shape (B, 3, H, W)
    """
    if x.shape[1] == 1:
        return x.repeat(1, 3, 1, 1)
    elif x.shape[1] == 3:
        return x
    else:
        raise ValueError(f"Input tensor must have 1 or 3 channels, got {x.shape[1]}")


def zero_one_to_neg_pos(x: torch.Tensor) -> torch.Tensor:
    """
    Convert a tensor from [0, 1] range to [-1, 1] range.
    Args:
        x: Tensor of shape (B, C, H, W) in [0, 1] range
    Returns:
        Tensor of shape (B, C, H, W) in [-1, 1] range
    """
    temp = torch.abs(torch.flatten(x))
    mm = torch.max(torch.abs(temp))
    return torch.clamp((0.5 * x / mm - 0.5), -1, 1)


def neg_pos_to_zero_one(x: torch.Tensor) -> torch.Tensor:
    temp = torch.abs(torch.flatten(x))
    mm = torch.max(torch.abs(temp))
    return torch.clamp(((x + 1) / (2 * mm)), 0, 1)
