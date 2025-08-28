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
    job_script = args.run_dir / "ih_job.slurm"

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
    
    shards_in_queue = [job["shard"] for job in jobs]
    shards_to_submit = [shard for shard in shards if shard not in shards_in_queue]

    if shards_to_submit:
        logger.info(f"Submitting {len(shards_to_submit)} jobs...")
    else:
        logger.info("No jobs to submit.")
        return

    for i, shard in enumerate(shards_to_submit):
        logger.info(f"  {i+1}/{len(shards_to_submit)}: Submitting job for shard {shard}...")
        cmd = ["sbatch", "--array", f"{shard+1}", "--job-name", f"{config.job_name}-{shard:06d}", str(job_script)]
        try:
            result = subprocess.run(
                cmd,
                check=True,
                capture_output=True,
                text=True,
            )
        except subprocess.CalledProcessError as e:
            logger.error(f"sbatch failed for shard {shard}: {e}")
            raise RuntimeError(f"sbatch failed: {e.stderr or e.stdout}") from e
        logger.info(f"    {result.stdout.strip()} for shard {shard}")
    logger.info("Done.")

if __name__ == "__main__":
    main()
