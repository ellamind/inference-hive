import argparse
from pathlib import Path
import time

from loguru import logger
from tabulate import tabulate
import polars as pl

from config import load_job_config
from slurm_utils import get_current_jobs, get_job_state_counts


def read_progress_file(file_path: str) -> pl.DataFrame:
    """Read JSON lines progress file and parse into a Polars DataFrame"""
    df = pl.read_ndjson(file_path)
    
    # Flatten the nested progress and throughput data
    df_expanded = df.with_columns([
        # Extract progress fields
        pl.col("progress").struct.field("completed").alias("completed"),
        pl.col("progress").struct.field("total").alias("total"),
        pl.col("progress").struct.field("new").alias("new"),
        pl.col("progress").struct.field("existing").alias("existing"),
        pl.col("progress").struct.field("eta_seconds").alias("eta_seconds"),
        pl.col("progress").struct.field("eta_formatted").alias("eta_formatted"),
        
        # Extract overall throughput fields (per second)
        pl.col("throughput").struct.field("overall").struct.field("requests_per_second").alias("overall_requests_ps"),
        pl.col("throughput").struct.field("overall").struct.field("total_tokens_per_second").alias("overall_total_tps"),
        pl.col("throughput").struct.field("overall").struct.field("prompt_tokens_per_second").alias("overall_prompt_tps"),
        pl.col("throughput").struct.field("overall").struct.field("completion_tokens_per_second").alias("overall_completion_tps"),
        
        # Convert timestamp to datetime for display
        pl.from_epoch(pl.col("timestamp")).alias("datetime")
    ]).select([
        "shard", "num_shards", "timestamp", "datetime", "completed", "total", "new", "existing",
        "eta_seconds", "eta_formatted", "overall_requests_ps", "overall_total_tps", 
        "overall_prompt_tps", "overall_completion_tps"
    ])
    
    return df_expanded


def load_and_combine_files(file_paths: list[str]) -> pl.DataFrame:
    """Load multiple progress files and combine them into a single DataFrame"""
    all_dataframes = []
    
    for file_path in file_paths:
        try:
            df = read_progress_file(file_path)
            all_dataframes.append(df)
        except Exception as e:
            print(f"✗ Failed to load {Path(file_path).name}: {e}")
            continue
    
    if not all_dataframes:
        raise ValueError("No valid data files could be loaded")
    
    # Combine all dataframes
    combined_df = pl.concat(all_dataframes)
    return combined_df


def calculate_current_total_throughput(df: pl.DataFrame, cutoff_timestamp: float) -> dict:
    """Calculate current total throughput from recent data across all active shards using a precomputed cutoff timestamp"""
    if len(df) == 0:
        return {"active_shards": 0, "requests_ps": 0, "total_tps": 0, "prompt_tps": 0, "completion_tps": 0}
    
    # Filter to recent data only
    recent_df = df.filter(pl.col("timestamp") >= cutoff_timestamp)
    
    if len(recent_df) == 0:
        return {"active_shards": 0, "requests_ps": 0, "total_tps": 0, "prompt_tps": 0, "completion_tps": 0}
    
    # Get the most recent record for each shard (latest data point per shard)
    latest_per_shard = recent_df.group_by("shard").agg([
        pl.col("timestamp").max().alias("latest_timestamp"),
        pl.col("overall_requests_ps").last().alias("requests_ps"),
        pl.col("overall_total_tps").last().alias("total_tps"),
        pl.col("overall_prompt_tps").last().alias("prompt_tps"),
        pl.col("overall_completion_tps").last().alias("completion_tps")
    ])
    
    # Sum across all active shards
    totals = latest_per_shard.select([
        pl.col("requests_ps").sum().alias("total_requests_ps"),
        pl.col("total_tps").sum().alias("total_total_tps"),
        pl.col("prompt_tps").sum().alias("total_prompt_tps"),
        pl.col("completion_tps").sum().alias("total_completion_tps"),
        pl.col("shard").n_unique().alias("active_shards")
    ]).row(0, named=True)
    
    return {
        "active_shards": totals["active_shards"],
        "requests_ps": totals["total_requests_ps"] or 0,
        "total_tps": totals["total_total_tps"] or 0,
        "prompt_tps": totals["total_prompt_tps"] or 0,
        "completion_tps": totals["total_completion_tps"] or 0
    }


def calculate_per_shard_stats(df: pl.DataFrame, cutoff_timestamp: float) -> pl.DataFrame:
    """Calculate statistics for each shard using a precomputed cutoff timestamp"""
    if len(df) == 0:
        return pl.DataFrame()
    
    # Get total number of shards from the data
    num_shards = df.select(pl.col("num_shards").first()).item()
    
    # Get the latest record for each shard that has data
    latest_per_shard = df.group_by("shard").agg([
        pl.col("timestamp").max().alias("latest_timestamp"),
        pl.col("datetime").last().alias("latest_datetime"),
        pl.col("completed").last().alias("completed"),
        pl.col("total").last().alias("total"),
        pl.col("eta_formatted").last().alias("eta_formatted"),
        # Number of entries (rows) in the shard's log
        pl.len().alias("entries"),
        pl.col("overall_requests_ps").mean().alias("avg_requests_ps"),
        pl.col("overall_total_tps").mean().alias("avg_total_tps"),
        pl.col("overall_prompt_tps").mean().alias("avg_prompt_tps"),
        pl.col("overall_completion_tps").mean().alias("avg_completion_tps")
    ]).with_columns([
        # Calculate progress percentage
        (pl.col("completed") / pl.col("total") * 100).alias("progress_pct"),
        # Determine if shard is currently active based on recent activity
        pl.when(pl.col("latest_timestamp") >= cutoff_timestamp)
        .then(pl.lit("active"))
        .otherwise(pl.lit("inactive"))
        .alias("status")
    ])
    
    # Create a complete list of all shards (0 to num_shards-1)
    all_shards = pl.DataFrame({"shard": list(range(num_shards))})
    
    # Left join to include all shards, even those without data
    complete_stats = all_shards.join(latest_per_shard, on="shard", how="left").with_columns([
        # Set status for shards with no data
        pl.when(pl.col("latest_timestamp").is_null())
        .then(pl.lit("No Data"))
        .otherwise(pl.col("status"))
        .alias("status"),
        # For shards with no data, entries should be 0
        pl.when(pl.col("entries").is_null())
        .then(pl.lit(0))
        .otherwise(pl.col("entries"))
        .alias("entries")
    ]).sort("shard")
    
    return complete_stats



def print_per_shard_stats(shard_stats: pl.DataFrame, shards_completed: list[int]):
    """Print per-shard statistics table including completion status"""
    if len(shard_stats) == 0:
        print("\nNo shard data available")
        return
    
    # Prepare table data
    table_data = []
    active_shards = 0
    
    for row in shard_stats.iter_rows(named=True):
        status = row['status'] if row['status'] is not None else "No Data"
        
        # Count active shards
        if status == "active":
            active_shards += 1
        
        # Check if this shard has data
        if row['latest_datetime'] is None:
            # Shard has no data
            table_data.append([
                f"Shard {row['shard']}",
                str(row['shard'] in shards_completed),
                "-",
                "0",
                status,
                "-",
                "-",
                "-", 
                "-",
                "-",
                "-"
            ])
        else:
            # Shard has data
            progress_pct = f"{row['progress_pct']:.1f}%" if row['progress_pct'] is not None else "-"
            table_data.append([
                f"Shard {row['shard']}",
                str(row['shard'] in shards_completed),
                progress_pct,
                str(row['entries']) if row['entries'] is not None else "0",
                status,
                f"{row['avg_requests_ps']:.1f}" if row['avg_requests_ps'] is not None else "-",
                f"{row['avg_total_tps']:.1f}" if row['avg_total_tps'] is not None else "-",
                f"{row['avg_prompt_tps']:.1f}" if row['avg_prompt_tps'] is not None else "-",
                f"{row['avg_completion_tps']:.1f}" if row['avg_completion_tps'] is not None else "-",
                row['latest_datetime'].strftime('%Y-%m-%d %H:%M:%S'),
                row['eta_formatted'] or "N/A"
            ])
    
    headers = [
        "Shard", "Completed", "Progress %", "Entries", "Status", "RPS", "Total TPS", 
        "Prompt TPS", "Compl TPS", "Last Update", "ETA"
    ]
    
    total_shards = len(shard_stats)
    print(f"\nPER-SHARD STATISTICS ({active_shards}/{total_shards} shards active)")
    print(f"{'='*90}")
    print(tabulate(table_data, headers=headers, tablefmt="grid"))


def print_current_total_throughput(totals: dict, recent_minutes: float):
    """Print current total throughput table"""
    table_data = [
        ["RPS", f"{totals['requests_ps']:,.1f}"],
        ["Total TPS", f"{totals['total_tps']:,.1f}"],
        ["Prompt TPS", f"{totals['prompt_tps']:,.1f}"],
        ["Completion TPS", f"{totals['completion_tps']:,.1f}"]
    ]
    
    print(f"\nCURRENT TOTAL THROUGHPUT ({totals['active_shards']} active shards, last {recent_minutes:.1f} min)")
    print(f"{'='*60}")
    print(tabulate(table_data, headers=["Metric", "Rate"], tablefmt="grid"))


def print_summary_info(df: pl.DataFrame, num_files: int, shards_completed=None, cutoff_timestamp: float = None):
    """Print additional summary information, including completed shard count. Uses a precomputed cutoff timestamp for activity."""
    if len(df) == 0:
        return
    
    # Calculate overall summary
    first_timestamp = df.select(pl.col("timestamp").min()).item()
    last_timestamp = df.select(pl.col("timestamp").max()).item()
    duration_minutes = (last_timestamp - first_timestamp) / 60
    
    # Get total configured shards vs active shards
    total_shards = df.select(pl.col("num_shards").first()).item()
    shards_with_data = df.select(pl.col("shard").n_unique()).item()
    
    # Calculate active shards (recent activity)
    if cutoff_timestamp is None:
        cutoff_timestamp = time.time()
    active_shards = df.filter(pl.col("timestamp") >= cutoff_timestamp).select(pl.col("shard").n_unique()).item() or 0
    
    # Get overall progress considering ALL shards (including those not started)
    latest_per_shard = df.group_by("shard").agg([
        pl.col("completed").last().alias("completed"),
        pl.col("total").last().alias("total"),
        pl.col("eta_seconds").last().alias("eta_seconds"),
        pl.col("eta_formatted").last().alias("eta_formatted")
    ])
    
    # Sum completed requests from shards that have data
    total_completed = latest_per_shard.select(pl.col("completed").sum()).item()
    
    # Calculate total expected requests across ALL shards
    # Assuming each shard has the same number of total requests
    requests_per_shard = latest_per_shard.select(pl.col("total").first()).item() if len(latest_per_shard) > 0 else 0
    total_requests_all_shards = requests_per_shard * total_shards if requests_per_shard else 0
    
    overall_progress = (total_completed / total_requests_all_shards * 100) if total_requests_all_shards > 0 else 0
    
    # Calculate longest ETA
    # If there are unstarted shards, longest ETA is infinity
    if active_shards < total_shards:
        longest_eta = "∞"
    else:
        # All shards have data, find the maximum ETA
        max_eta_seconds = latest_per_shard.filter(pl.col("eta_seconds").is_not_null()).select(pl.col("eta_seconds").max()).item()
        if max_eta_seconds is not None and max_eta_seconds > 0:
            # Find the corresponding formatted ETA
            max_eta_row = latest_per_shard.filter(pl.col("eta_seconds") == max_eta_seconds).select(pl.col("eta_formatted")).item()
            longest_eta = max_eta_row if max_eta_row else "N/A"
        else:
            longest_eta = "N/A"
    
    completed_shards_count = len(shards_completed) if shards_completed else 0

    summary_data = [
        ["Total shards", f"{total_shards}"],
        ["Shards completed", f"{completed_shards_count}"],
        ["Shards w/ data", f"{shards_with_data}"],
        ["Active shards", f"{active_shards}"],
        ["Duration", f"{duration_minutes:.1f} minutes"],
        ["Total completed", f"{total_completed:,}"],
        ["Total requests", f"{total_requests_all_shards:,}"],
        ["Overall progress", f"{overall_progress:.1f}%"],
        ["Longest ETA", longest_eta]
    ]
    
    print("\nSUMMARY")
    print(f"{'='*30}")
    print(tabulate(summary_data, tablefmt="simple"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--recent-minutes", type=float, default=5.0)
    parser.add_argument("--detailed", action="store_true", default=False)
    args = parser.parse_args()

    # args=argparse.Namespace()
    # args.run_dir = Path("fw2_annotations_run3")
    # args.recent_minutes = 2

    config = load_job_config(args.run_dir / "ih_config.yaml")

    shards = list(range(config.num_inference_servers))

    try:
        jobs = get_current_jobs(job_name=config.job_name)
    except Exception as e:
        logger.error(f"Error getting current jobs: {e}")
        raise e

    if jobs:
        for job in jobs:
            job['shard'] = int(job["name"].split("-")[-1]) -1
        jobs.sort(key=lambda x: x["shard"])

        logger.info(f"Found {len(jobs)} existing jobs for {config.job_name}")
        for state, count in get_job_state_counts(jobs).items():
            logger.info(f"    {count}/{len(jobs)} {state}")
    else:
        logger.info(f"No existing jobs found for {config.job_name}")
    
    shards_running = [job["shard"] for job in jobs if "RUNNING" in job["job_state"]]
    shards_not_running = [job["shard"] for job in jobs if "RUNNING" not in job["job_state"]]
    
    logger.info(f"Shards running: {shards_running}")
    logger.info(f"Shards in queue: {shards_not_running}")

    shards_completed_file = Path(args.run_dir / "shards_completed.log")

    if shards_completed_file.exists():
        shards_completed = [int(line.strip()) for line in shards_completed_file.read_text().splitlines()]
    else:
        shards_completed = []
    
    job_by_shard = {job["shard"]: job for job in jobs}
    headers = ["Shard", "Completed", "State", "JobID"]
    rows = []
    for shard in shards:
        job = job_by_shard.get(shard)
        state = job.get("job_state") if job else "-"
        job_id = job.get("job_id", "-") if job else "-"
        completed = str(shard in shards_completed)
        rows.append([shard, completed, state, job_id])
    table_str = "\n" + tabulate(rows, headers=headers, tablefmt="github") + "\n"
    print(table_str)

    if args.detailed:

        # Capture a single reference timestamp once before loading data to avoid processing-delay drift
        reference_timestamp = time.time()
        cutoff_timestamp = reference_timestamp - (args.recent_minutes * 60)
        cutoff_human = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(cutoff_timestamp))
        reference_human = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(reference_timestamp))
        logger.info(f"Using cutoff based on system time: cutoff={cutoff_human}, reference={reference_human}, window={args.recent_minutes} min")

        progress_files = []
        for shard in shards:
            progress_file = Path(args.run_dir / "progress" / f"{shard+1}-progress.jsonl")
            if progress_file.exists() and progress_file.stat().st_size > 0:
                progress_files.append(progress_file)

        logger.info(f"Found {len(progress_files)} progress files")
        if not progress_files:
            logger.info("No progress files found - exiting.")
            return
        
        logger.info(f"Loading {len(progress_files)} progress files")
        df = load_and_combine_files(progress_files)
        logger.info(f"Total data points: {len(df):_}")

        shard_stats = calculate_per_shard_stats(df, cutoff_timestamp)
        print_per_shard_stats(shard_stats, shards_completed)

        current_totals = calculate_current_total_throughput(df, cutoff_timestamp)
        print_current_total_throughput(current_totals, args.recent_minutes)

        print_summary_info(df, len(progress_files), shards_completed, cutoff_timestamp)
        

if __name__ == "__main__":
    main()
