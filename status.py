import argparse
from pathlib import Path

import subprocess

from loguru import logger

from config import load_job_config
from slurm_utils import get_current_jobs, get_job_state_counts

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    config = load_job_config(args.run_dir / "ih_config.yaml")

    shards = list(range(config.num_inference_servers))

    try:
        jobs = get_current_jobs(job_name=config.job_name)
    except Exception as e:
        logger.error(f"Error getting current jobs: {e}")
        raise e

    if jobs:
        for job in jobs:
            job['shard'] = int(job["name"].split("-")[-1])
        jobs.sort(key=lambda x: x["shard"])

        logger.info(f"Found {len(jobs)} existing jobs for {config.job_name}")
        for state, count in get_job_state_counts(jobs).items():
            logger.info(f"    {count}/{len(jobs)} {state}")
    else:
        logger.info(f"No existing jobs found for {config.job_name}")
    
    shards_running = [job["shard"] for job in jobs if job["job_state"] == "RUNNING"]


if __name__ == "__main__":
    main()
