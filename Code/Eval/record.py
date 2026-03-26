from point_clouds import PointClouds
import os
import numpy as np
import pandas as pd
import re
from tqdm import tqdm
from PIL import Image
import multiprocessing as mp
from functools import partial


class Recorder:
    def __init__(self, ground_truth_dir, predicted_dir):
        self.ground_truth_dir = ground_truth_dir
        self.predicted_dir = predicted_dir
        self.ground_truth_files = sorted(
            [f for f in os.listdir(ground_truth_dir) if f.endswith(".png")]
        )
        self.predicted_files = sorted(
            [f for f in os.listdir(predicted_dir) if f.endswith(".png")]
        )
        self.converter = PointClouds()
        self.results = []

        print("Initialized Recorder")

    def _process_single_pair(self, pred_file, ground_truth_map):
        traj, frame = self.parse_filename(pred_file)
        if traj is None or frame is None:
            return None
        key = f"{traj}_{frame}"
        if key not in ground_truth_map:
            return None
        gt_file = ground_truth_map[key]
        gt_path = os.path.join(self.ground_truth_dir, gt_file)
        pred_path = os.path.join(self.predicted_dir, pred_file)
        if not os.path.exists(gt_path) or not os.path.exists(pred_path):
            return None
        gt_image = self.load_image(gt_path)
        pred_image = self.load_image(pred_path)
        gt_points = self.converter.extract_point_cloud_from_polar_image(gt_image)
        pred_points = self.converter.extract_point_cloud_from_polar_image(pred_image)
        cd_distance = self.converter.compute_chamfer(gt_points, pred_points)
        mhd_distance = self.converter.compute_modified_hausdorff_distance(
            gt_points, pred_points
        )
        return {
            "trajectory": traj,
            "frame_index": frame,
            "CD": cd_distance,
            "MHD": mhd_distance,
            "gt_file": gt_file,
            "pred_file": pred_file,
        }

    def parse_filename(self, filename):
        pattern = r"(?:L_|pred_)(\d+)_(\d+)\.png"
        match = re.match(pattern, filename)

        if match:
            trajectory = int(match.group(1))
            frame_index = int(match.group(2))
            return trajectory, frame_index
        else:
            return None, None

    def load_image(self, image_path):
        image = Image.open(image_path)
        image = np.asarray(image).astype(np.bool_)
        image = (image * 255).astype(np.uint8)
        return image

    def evaluate(self, num_processes=None):
        print(f"Found {len(self.ground_truth_files)} ground truth files")
        print(f"Found {len(self.predicted_files)} predicted files")
        ground_truth_map = {}
        for gt_file in self.ground_truth_files:
            traj, frame = self.parse_filename(gt_file)
            if traj is not None and frame is not None:
                ground_truth_map[f"{traj}_{frame}"] = gt_file
        if num_processes is None:
            num_processes = min(mp.cpu_count(), len(self.predicted_files))

        print(
            f"Processing {len(self.predicted_files)} predicted files using {num_processes} processes..."
        )
        with mp.Pool(processes=num_processes) as pool:
            process_func = partial(
                self._process_single_pair, ground_truth_map=ground_truth_map
            )
            results = list(
                tqdm(
                    pool.imap(process_func, self.predicted_files),
                    total=len(self.predicted_files),
                    desc="Evaluating",
                    unit="file",
                )
            )
        self.results = [result for result in results if result is not None]

        print(f"Evaluation completed. Processed {len(self.results)} image pairs.")

    def save_results(self, output_path="evaluation_results.csv"):
        if not self.results:
            print("No results to save!")
            return

        df = pd.DataFrame(self.results)
        df = df[["trajectory", "frame_index", "CD", "MHD"]]
        csv_dir = "csv"
        if not os.path.exists(csv_dir):
            os.makedirs(csv_dir)
            print(f"Created directory: {csv_dir}")
        csv_path = os.path.join(csv_dir, output_path)
        df.to_csv(csv_path, index=False)
        print(f"Results saved to: {csv_path}")
        print("\nSummary Statistics:")
        print(f"Total pairs evaluated: {len(df)}")
        print(f"Average CD: {df['CD'].mean():.6f}")
        print(f"Average MHD: {df['MHD'].mean():.6f}")
        print(f"CD Std Dev: {df['CD'].std():.6f}")
        print(f"MHD Std Dev: {df['MHD'].std():.6f}")

        return df

    def run_evaluation(self, output_path="evaluation_results.csv", num_processes=None):
        print("Starting evaluation...")
        self.evaluate(num_processes=num_processes)
        return self.save_results(output_path)


if __name__ == "__main__":
    ground_truth_dir = "./dataset_5/test/lidar"
    predicted_dirs = ["marigold_perceptual_images"]

    for predicted_dir in predicted_dirs:
        print(f"\n{'='*50}")
        print(f"Evaluating {predicted_dir}")
        print(f"{'='*50}")

        recorder = Recorder(ground_truth_dir, predicted_dir)
        recorder.run_evaluation(f"{predicted_dir}_results.csv", num_processes=8)
