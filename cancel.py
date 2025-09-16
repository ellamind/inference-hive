import argparse
from pathlib import Path

import subprocess
import re

from loguru import logger

from config import load_job_config
from slurm_utils import get_current_jobs

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    # args = argparse.Namespace()
    # args.run_dir = Path("fw2_annotations_run1")
    config = load_job_config(args.run_dir / "ih_config.yaml")
    jobs = get_current_jobs(me=True, job_name=config.job_name)
    name_pattern = re.compile(rf"^{re.escape(config.job_name)}-\d{{6}}$")
    jobs = [job for job in jobs if name_pattern.match(job.get("name", ""))]
    job_ids = [str(job["job_id"]) for job in jobs if "job_id" in job]

    if not job_ids:
        logger.info("No matching jobs found to cancel.")
        return

    batch_size = 100
    for start in range(0, len(job_ids), batch_size):
        batch = job_ids[start:start + batch_size]
        cmd = ["scancel", *batch]
        try:
            _ = subprocess.run(
                cmd,
                check=True,
                capture_output=True,
                text=True,
            )
            logger.info(f"Cancelled {len(batch)} job(s): {' '.join(batch)}")
        except subprocess.CalledProcessError as e:
            logger.error(f"scancel failed for batch starting at index {start}: {e}")
            raise RuntimeError(f"scancel failed: {e.stderr or e.stdout}") from e

    logger.info("Done.")

if __name__ == "__main__":
    main()
