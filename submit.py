import argparse
import time
from pathlib import Path

import subprocess

from loguru import logger

from inference_hive.config import load_job_config
from inference_hive.slurm_utils import get_current_jobs, get_job_state_counts


def get_shards_to_submit(config, run_dir):
    """Get list of shards that still need to be submitted."""
    shards = list(range(config.num_inference_servers))
    
    # Check completed shards
    shards_completed_file = run_dir / 'progress' / "shards_completed.log"
    if shards_completed_file.exists():
        shards_completed = [int(line.strip()) for line in shards_completed_file.read_text().splitlines()]
    else:
        shards_completed = []

    if shards_completed:
        logger.info(f"Shards completed: {len(shards_completed)}")

    # Check jobs in queue
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
    shards_to_submit = [shard for shard in shards if (shard not in shards_in_queue) and (shard not in shards_completed)]
    
    return shards_to_submit


def submit_shards(shards_to_submit, config, job_script):
    """Submit shards, return (submitted_count, remaining_shards) if QOS limit hit."""
    submitted = 0
    for i, shard in enumerate(shards_to_submit):
        logger.info(f"  {i+1}/{len(shards_to_submit)}: Submitting job for shard {shard}...")
        cmd = ["sbatch", "--array", f"{shard}", "--job-name", f"{config.job_name}-{shard:06d}", str(job_script)]
        try:
            result = subprocess.run(
                cmd,
                check=True,
                capture_output=True,
                text=True,
            )
            logger.info(f"    {result.stdout.strip()} for shard {shard}")
            submitted += 1
        except subprocess.CalledProcessError as e:
            error_msg = e.stderr or e.stdout or ""
            if "QOSMaxSubmitJobPerUserLimit" in error_msg or "Job violates accounting/QOS policy" in error_msg:
                logger.warning(f"QOS limit reached after submitting {submitted} jobs. Remaining: {len(shards_to_submit) - i}")
                return submitted, shards_to_submit[i:]
            else:
                logger.error(f"sbatch failed for shard {shard}: {e}")
                raise RuntimeError(f"sbatch failed: {error_msg}") from e
    
    return submitted, []


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=None,
                        help="Maximum number of jobs to submit per cycle")
    parser.add_argument("--retry-interval", type=int, default=None,
                        help="Retry interval in minutes. If set, script will keep retrying until all shards are submitted.")
    parser.add_argument("--max-retries", type=int, default=None,
                        help="Maximum number of retry cycles (default: unlimited)")
    args = parser.parse_args()
    
    config = load_job_config(args.run_dir / "ih_config.yaml")
    job_script = args.run_dir / "ih_job.slurm"
    
    retry_count = 0
    
    while True:
        if retry_count > 0:
            logger.info(f"=== Retry cycle {retry_count} ===")
        
        shards_to_submit = get_shards_to_submit(config, args.run_dir)
        logger.info(f"Jobs to submit: {len(shards_to_submit)}")

        if not shards_to_submit:
            logger.info("No jobs to submit. All shards are either completed or in queue.")
            return
        
        logger.info(f"Submitting {len(shards_to_submit)} jobs...")
        
        if args.limit:
            logger.warning(f"Applying limit of {args.limit} per cycle.")
            shards_to_submit = shards_to_submit[:args.limit]

        submitted, remaining = submit_shards(shards_to_submit, config, job_script)
        
        if not remaining:
            logger.info(f"Done. Submitted {submitted} jobs.")
            if args.retry_interval:
                # Re-check in case more shards became available (completed jobs freed slots)
                shards_to_submit = get_shards_to_submit(config, args.run_dir)
                if not shards_to_submit:
                    logger.info("All shards submitted or in queue. Exiting.")
                    return
                # Continue to retry logic below
                remaining = shards_to_submit
            else:
                return
        
        # If we have remaining shards and retry is enabled
        if remaining and args.retry_interval:
            retry_count += 1
            if args.max_retries and retry_count >= args.max_retries:
                logger.warning(f"Max retries ({args.max_retries}) reached. {len(remaining)} shards still pending.")
                return
            
            logger.info(f"Waiting {args.retry_interval} minutes before next retry... ({len(remaining)} shards remaining)")
            time.sleep(args.retry_interval * 60)
        elif remaining:
            # No retry enabled, exit with error
            raise RuntimeError(f"QOS limit reached. {len(remaining)} shards not submitted. Use --retry-interval to auto-retry.")

if __name__ == "__main__":
    main()
