import base64
import json
import os
import numpy as np
import cv2
import boto3
import torch
from io import BytesIO
from PIL import Image
from functools import lru_cache

from models.with_mobilenet import PoseEstimationWithMobileNet
from modules.load_state import load_state
from modules.keypoints import extract_keypoints, group_keypoints
from modules.pose import Pose
from demo import infer_fast


class PoseRunner:
    def __init__(self, bucket, checkpoint_path, result_prefix, height=256, stride=8, upsample_ratio=4, use_cpu=True):
        self.bucket = bucket
        self.checkpoint_path = checkpoint_path
        self.result_prefix = result_prefix
        self.height = height
        self.stride = stride
        self.upsample_ratio = upsample_ratio
        self.use_cpu = use_cpu
        self.s3 = boto3.client("s3")
        self._net = None

    @lru_cache(maxsize=1)
    def get_net(self):
        """
        Load the neural network model (cached for efficiency).
        
        Returns:
            The loaded and configured neural network model
        """
        if self._net is None:
            self._net = PoseEstimationWithMobileNet()
            load_state(self._net, torch.load(self.checkpoint_path, map_location="cpu"))
            self._net.eval()
        return self._net

    def run_pose(self, image_key):
        img_bytes = self.s3.get_object(Bucket=self.bucket, Key=image_key)["Body"].read()
        img_arr = np.array(Image.open(BytesIO(img_bytes)).convert("RGB"))

        net = self.get_net()
        heat, paf, scale, pad = infer_fast(
            net, img_arr, self.height, self.stride, self.upsample_ratio, cpu=self.use_cpu
        )
        
        num_kpts = Pose.num_kpts
        total, kpts_by_type = 0, []
        for k in range(num_kpts):
            total += extract_keypoints(heat[:, :, k], kpts_by_type, total)
        entries, all_kpts = group_keypoints(kpts_by_type, paf)
        
        for k in range(all_kpts.shape[0]):
            all_kpts[k, 0] = (all_kpts[k, 0] * self.stride / self.upsample_ratio - pad[1]) / scale
            all_kpts[k, 1] = (all_kpts[k, 1] * self.stride / self.upsample_ratio - pad[0]) / scale

        poses, json_out = [], []
        for e in entries:
            if not len(e):
                continue
            coords = np.ones((num_kpts, 2), dtype=int) * -1
            kp_json = []
            for k in range(num_kpts):
                if e[k] != -1:
                    x, y = map(int, all_kpts[int(e[k]), :2])
                    coords[k] = [x, y]
                    kp_json.append({"id": k, "x": x, "y": y})
            poses.append(Pose(coords, e[18]))
            json_out.append({"keypoints": kp_json})

        out = img_arr.copy()
        for p in poses:
            p.draw(out)
        out = cv2.addWeighted(img_arr, 0.6, out, 0.4, 0)
        for p in poses:
            x, y, w, h = p.bbox
            cv2.rectangle(out, (x, y), (x + w, y + h), (0, 255, 0), 2)

        base = os.path.basename(image_key)
        img_key = f"{self.result_prefix}/pose_{base}"
        json_key = f"{self.result_prefix}/pose_{os.path.splitext(base)[0]}.json"

        tmp_img = f"/tmp/{os.path.basename(img_key)}"
        tmp_json = f"/tmp/{os.path.basename(json_key)}"
        
        cv2.imwrite(tmp_img, cv2.cvtColor(out, cv2.COLOR_RGB2BGR))
        with open(tmp_json, "w") as f:
            json.dump(json_out, f, indent=2)
            
        with open(tmp_img, "rb") as fh:
            data_url = "data:image/jpeg;base64," + base64.b64encode(fh.read()).decode()

        self.s3.upload_file(tmp_img, self.bucket, img_key)
        self.s3.upload_file(tmp_json, self.bucket, json_key)
        
        return img_key, json_key, data_url


_runner = None


def initialize_runner(bucket, checkpoint_path, result_prefix, height=256, stride=8, upsample_ratio=4, use_cpu=True):
    global _runner
    _runner = PoseRunner(
        bucket=bucket,
        checkpoint_path=checkpoint_path,
        result_prefix=result_prefix,
        height=height,
        stride=stride,
        upsample_ratio=upsample_ratio,
        use_cpu=use_cpu
    )
    _runner.get_net()
    return _runner


def _get_net():
    if _runner is None:
        raise RuntimeError("Pose runner not initialized. Call initialize_runner() first.")
    return _runner.get_net()


def run_pose(image_key):
    if _runner is None:
        raise RuntimeError("Pose runner not initialized. Call initialize_runner() first.")
    return _runner.run_pose(image_key)


if __name__ == "__main__":
    import argparse
    
    def parse_arguments():
        parser = argparse.ArgumentParser(description="Pose estimation runner")
        parser.add_argument(
            "--bucket", type=str, default="ziad-pose-estimation",
            help="S3 bucket name for storing results"
        )
        parser.add_argument(
            "--checkpoint-path", type=str, default="models/human-pose.pth",
            help="Path to model checkpoint file"
        )
        parser.add_argument(
            "--result-prefix", type=str, default="results",
            help="Prefix for result files in S3"
        )
        parser.add_argument(
            "--height", type=int, default=256,
            help="Network input layer height"
        )
        parser.add_argument(
            "--stride", type=int, default=8,
            help="Model stride"
        )
        parser.add_argument(
            "--upsample-ratio", type=int, default=4,
            help="Model upsample ratio"
        )
        parser.add_argument(
            "--use-cpu", action="store_true", default=True,
            help="Use CPU for inference"
        )
        return parser.parse_args()
    
    args = parse_arguments()
    runner = initialize_runner(
        bucket=args.bucket,
        checkpoint_path=args.checkpoint_path,
        result_prefix=args.result_prefix,
        height=args.height,
        stride=args.stride,
        upsample_ratio=args.upsample_ratio,
        use_cpu=args.use_cpu
    )
    
    print(f"Pose Runner initialized with the following configuration:")
    print(f"- S3 Bucket: {args.bucket}")
    print(f"- Checkpoint Path: {args.checkpoint_path}")
    print(f"- Result Prefix: {args.result_prefix}")
    print(f"- Network Input Height: {args.height}")
    print(f"- Stride: {args.stride}")
    print(f"- Upsample Ratio: {args.upsample_ratio}")
    print(f"- Using CPU: {args.use_cpu}")
    print("Ready to process images. To use this module, import run_pose().")
