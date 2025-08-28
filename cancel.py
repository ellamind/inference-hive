import argparse
from pathlib import Path

import subprocess

from loguru import logger

from config import load_job_config

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    # args = argparse.Namespace()
    # args.run_dir = Path("fw2_annotations_run1")
    config = load_job_config(args.run_dir / "ih_config.yaml")
    shards = list(range(config.num_inference_servers))

    for shard in shards:
        cmd = ["scancel", "--name",f"{config.job_name}-{shard+1:06d}"]
        try:
            _ = subprocess.run(
                cmd,
                check=True,
                capture_output=True,
                text=True,
            )
        except subprocess.CalledProcessError as e:
            logger.error(f"scancel failed for shard {shard}: {e}")
            raise RuntimeError(f"scancel failed: {e.stderr or e.stdout}") from e

    logger.info("Done.")

if __name__ == "__main__":
    main()
