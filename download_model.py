"""
download_model.py
~~~~~~~~~~~~~~~~~
Download a trained nanogpt model checkpoint from a RunPod network volume
via its S3-compatible API to your local machine for inference.

Only downloads the files needed for inference (model + tokenizer).
The optimizer state (~1.5 GB) is skipped by default.

Credentials are read from .env in the project root:
  RUNPOD_S3_ENDPOINT   e.g. https://s3api-us-ca-2.runpod.io
  RUNPOD_S3_BUCKET     e.g. yrbayk4y
  RUNPOD_S3_ACCESS_KEY your RunPod S3 access key
  RUNPOD_S3_SECRET_KEY your RunPod S3 secret key

Usage:
  # Download latest SFT checkpoint + tokenizer
  python download_model.py

  # Download a specific step
  python download_model.py --step 44043

  # Custom paths
  python download_model.py \\
    --remote-checkpoint-dir sft_checkpoints_v2 \\
    --remote-tokenizer-dir  tokenizer_v2 \\
    --local-dir             ./local_model/

  # Also download optimizer state (needed to resume training, not for inference)
  python download_model.py --include-optimizer
"""

import argparse
import os
import sys
import re
from pathlib import Path

from size_utils import load_env

load_env()


def parse_args():
    p = argparse.ArgumentParser(description="Download nanogpt model from RunPod S3")
    p.add_argument("--endpoint",                type=str, default=None,
                   help="S3 endpoint URL (default: RUNPOD_S3_ENDPOINT from .env)")
    p.add_argument("--bucket",                  type=str, default=None,
                   help="S3 bucket name (default: RUNPOD_S3_BUCKET from .env)")
    p.add_argument("--remote-checkpoint-dir",   type=str, default="sft_checkpoints_v2",
                   help="Remote path to checkpoint directory (default: sft_checkpoints_v2)")
    p.add_argument("--remote-tokenizer-dir",    type=str, default="tokenizer_v2",
                   help="Remote path to tokenizer directory (default: tokenizer_v2)")
    p.add_argument("--local-dir",               type=str, default=".",
                   help="Local directory to download into (default: current dir)")
    p.add_argument("--step",                    type=int, default=None,
                   help="Checkpoint step to download (default: latest)")
    p.add_argument("--include-optimizer",       action="store_true",
                   help="Also download optimizer state (~1.5 GB, needed for training resume)")
    return p.parse_args()


def make_client(endpoint: str, access_key: str, secret_key: str):
    try:
        import boto3
        from botocore.config import Config
    except ImportError:
        print("boto3 not installed. Run: pip install boto3")
        sys.exit(1)

    return boto3.client(
        "s3",
        endpoint_url=endpoint,
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        region_name="us-east-1",
        config=Config(signature_version="s3v4"),
    )


def list_objects(client, bucket: str, prefix: str) -> list[dict]:
    objects = []
    paginator = client.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        objects.extend(page.get("Contents", []))
    return objects


def find_latest_step(objects: list[dict]) -> int | None:
    steps = []
    for obj in objects:
        key = obj["Key"]
        m = re.search(r"model_(\d+)\.pt$", key)
        if m:
            steps.append(int(m.group(1)))
    return max(steps) if steps else None


def format_size(n_bytes: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n_bytes < 1024:
            return f"{n_bytes:.1f} {unit}"
        n_bytes /= 1024
    return f"{n_bytes:.1f} GB"


def download_file(client, bucket: str, key: str, local_path: Path, size: int):
    try:
        from tqdm import tqdm
        has_tqdm = True
    except ImportError:
        has_tqdm = False

    local_path.parent.mkdir(parents=True, exist_ok=True)

    name = local_path.name
    print(f"  {name:40s}  {format_size(size):>10}", end="  ", flush=True)

    if has_tqdm:
        with tqdm(total=size, unit="B", unit_scale=True, leave=False, ncols=60) as pbar:
            client.download_file(
                bucket, key, str(local_path),
                Callback=lambda n: pbar.update(n),
            )
    else:
        client.download_file(bucket, key, str(local_path))

    print("done")


def main():
    args = parse_args()

    endpoint   = args.endpoint   or os.environ.get("RUNPOD_S3_ENDPOINT")
    bucket     = args.bucket     or os.environ.get("RUNPOD_S3_BUCKET")
    access_key = os.environ.get("RUNPOD_S3_ACCESS_KEY")
    secret_key = os.environ.get("RUNPOD_S3_SECRET_KEY")

    missing = []
    if not endpoint:   missing.append("RUNPOD_S3_ENDPOINT")
    if not bucket:     missing.append("RUNPOD_S3_BUCKET")
    if not access_key: missing.append("RUNPOD_S3_ACCESS_KEY")
    if not secret_key: missing.append("RUNPOD_S3_SECRET_KEY")
    if missing:
        print(f"Missing credentials: {', '.join(missing)}")
        print("Add them to .env or pass --endpoint / --bucket as flags.")
        sys.exit(1)

    local_dir = Path(args.local_dir)
    client    = make_client(endpoint, access_key, secret_key)

    # -------------------------------------------------------------------------
    # Checkpoint files

    ckpt_prefix = args.remote_checkpoint_dir.rstrip("/") + "/"
    print(f"\nListing {bucket}/{ckpt_prefix} ...")
    ckpt_objects = list_objects(client, bucket, ckpt_prefix)

    if not ckpt_objects:
        print(f"No objects found at {bucket}/{ckpt_prefix}")
        sys.exit(1)

    step = args.step or find_latest_step(ckpt_objects)
    if step is None:
        print("Could not find any model_*.pt file in the checkpoint directory.")
        sys.exit(1)

    print(f"Checkpoint step: {step:,}")

    wanted_suffixes = {f"model_{step}.pt", f"meta_{step}.json"}
    if args.include_optimizer:
        wanted_suffixes.add(f"optim_{step}_rank0.pt")

    print(f"\nDownloading checkpoint files (step {step}):")
    downloaded = 0
    for obj in ckpt_objects:
        key  = obj["Key"]
        name = Path(key).name
        if name in wanted_suffixes:
            local_path = local_dir / key
            download_file(client, bucket, key, local_path, obj["Size"])
            downloaded += 1

    if downloaded == 0:
        print(f"No checkpoint files found for step {step}.")
        sys.exit(1)

    # -------------------------------------------------------------------------
    # Tokenizer files

    tok_prefix = args.remote_tokenizer_dir.rstrip("/") + "/"
    print(f"\nDownloading tokenizer files ({bucket}/{tok_prefix}):")
    tok_objects = list_objects(client, bucket, tok_prefix)

    if not tok_objects:
        print(f"Warning: no tokenizer files found at {bucket}/{tok_prefix}")
    else:
        for obj in tok_objects:
            key  = obj["Key"]
            name = Path(key).name
            if name.startswith("."):
                continue
            local_path = local_dir / key
            download_file(client, bucket, key, local_path, obj["Size"])

    # -------------------------------------------------------------------------
    # Summary

    ckpt_local  = local_dir / args.remote_checkpoint_dir
    tok_local   = local_dir / args.remote_tokenizer_dir
    print(f"""
Done.
  Checkpoint : {ckpt_local}/
  Tokenizer  : {tok_local}/

Run the model locally:
  python chat_cli.py \\
    --model-dir {ckpt_local}/ \\
    --tokenizer-dir {tok_local}/
""")


if __name__ == "__main__":
    main()
