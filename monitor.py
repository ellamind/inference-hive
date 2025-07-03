import polars as pl
import argparse
from pathlib import Path
from tabulate import tabulate
from datetime import datetime
import glob


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


def discover_progress_files(directory: str) -> list[str]:
    """Discover all inference stats JSONL files in a directory"""
    directory_path = Path(directory)
    if not directory_path.exists():
        raise FileNotFoundError(f"Directory '{directory}' not found")
    
    pattern = str(directory_path / "*-inference-stats.jsonl")
    files = glob.glob(pattern)
    
    if not files:
        raise FileNotFoundError(f"No inference stats files found in '{directory}'")
    
    return sorted(files)


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


def calculate_current_total_throughput(df: pl.DataFrame, recent_minutes: float = 2.0) -> dict:
    """Calculate current total throughput from recent data across all active shards"""
    if len(df) == 0:
        return {"active_shards": 0, "requests_ps": 0, "total_tps": 0, "prompt_tps": 0, "completion_tps": 0}
    
    # Use current time as reference point for "recent" data
    import time
    current_timestamp = time.time()
    cutoff_timestamp = current_timestamp - (recent_minutes * 60)
    
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


def calculate_per_shard_stats(df: pl.DataFrame, recent_minutes: float = 2.0) -> pl.DataFrame:
    """Calculate statistics for each shard"""
    if len(df) == 0:
        return pl.DataFrame()
    
    # Get total number of shards from the data
    num_shards = df.select(pl.col("num_shards").first()).item()
    
    # Calculate cutoff for "running" status (recent activity)
    import time
    current_timestamp = time.time()
    cutoff_timestamp = current_timestamp - (recent_minutes * 60)
    
    # Get the latest record for each shard that has data
    latest_per_shard = df.group_by("shard").agg([
        pl.col("timestamp").max().alias("latest_timestamp"),
        pl.col("datetime").last().alias("latest_datetime"),
        pl.col("completed").last().alias("completed"),
        pl.col("total").last().alias("total"),
        pl.col("eta_formatted").last().alias("eta_formatted"),
        pl.col("overall_requests_ps").mean().alias("avg_requests_ps"),
        pl.col("overall_total_tps").mean().alias("avg_total_tps"),
        pl.col("overall_prompt_tps").mean().alias("avg_prompt_tps"),
        pl.col("overall_completion_tps").mean().alias("avg_completion_tps")
    ]).with_columns([
        # Calculate progress percentage
        (pl.col("completed") / pl.col("total") * 100).alias("progress_pct"),
        # Determine if shard is currently running based on recent activity
        pl.when(pl.col("latest_timestamp") >= cutoff_timestamp)
        .then(pl.lit("Running"))
        .otherwise(pl.lit("Not Running"))
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
        .alias("status")
    ]).sort("shard")
    
    return complete_stats


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


def print_per_shard_stats(shard_stats: pl.DataFrame):
    """Print per-shard statistics table"""
    if len(shard_stats) == 0:
        print("\nNo shard data available")
        return
    
    # Prepare table data
    table_data = []
    running_shards = 0
    
    for row in shard_stats.iter_rows(named=True):
        status = row['status'] if row['status'] is not None else "No Data"
        
        # Count running shards
        if status == "Running":
            running_shards += 1
        
        # Check if this shard has data
        if row['latest_datetime'] is None:
            # Shard has no data
            table_data.append([
                f"Shard {row['shard']}",
                "-",
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
                progress_pct,
                status,
                f"{row['avg_requests_ps']:.1f}" if row['avg_requests_ps'] is not None else "-",
                f"{row['avg_total_tps']:.1f}" if row['avg_total_tps'] is not None else "-",
                f"{row['avg_prompt_tps']:.1f}" if row['avg_prompt_tps'] is not None else "-",
                f"{row['avg_completion_tps']:.1f}" if row['avg_completion_tps'] is not None else "-",
                row['latest_datetime'].strftime('%Y-%m-%d %H:%M:%S'),
                row['eta_formatted'] or "N/A"
            ])
    
    headers = [
        "Shard", "Progress %", "Status", "RPS", "Total TPS", 
        "Prompt TPS", "Compl TPS", "Last Update", "ETA"
    ]
    
    total_shards = len(shard_stats)
    print(f"\nPER-SHARD STATISTICS ({running_shards}/{total_shards} shards running)")
    print(f"{'='*90}")
    print(tabulate(table_data, headers=headers, tablefmt="grid"))


def print_summary_info(df: pl.DataFrame, num_files: int, recent_minutes: float = 2.0):
    """Print additional summary information"""
    if len(df) == 0:
        return
    
    # Calculate overall summary
    first_timestamp = df.select(pl.col("timestamp").min()).item()
    last_timestamp = df.select(pl.col("timestamp").max()).item()
    duration_minutes = (last_timestamp - first_timestamp) / 60
    
    # Get total configured shards vs active shards vs running shards
    total_shards = df.select(pl.col("num_shards").first()).item()
    active_shards = df.select(pl.col("shard").n_unique()).item()
    
    # Calculate running shards (recent activity)
    import time
    current_timestamp = time.time()
    cutoff_timestamp = current_timestamp - (recent_minutes * 60)
    running_shards = df.filter(pl.col("timestamp") >= cutoff_timestamp).select(pl.col("shard").n_unique()).item() or 0
    
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
    
    summary_data = [
        ["Total Shards", f"{total_shards}"],
        ["Shards w/ Data", f"{active_shards}"],
        ["Running Shards", f"{running_shards}"],
        ["Duration", f"{duration_minutes:.1f} minutes"],
        ["Total Completed", f"{total_completed:,}"],
        ["Total Requests", f"{total_requests_all_shards:,}"],
        ["Overall Progress", f"{overall_progress:.1f}%"],
        ["Longest ETA", longest_eta]
    ]
    
    print(f"\nSUMMARY")
    print(f"{'='*30}")
    print(tabulate(summary_data, tablefmt="simple"))


def main():
    parser = argparse.ArgumentParser(
        description="Analyze multi-shard inference progress and throughput statistics",
    )
    
    parser.add_argument(
        "path",
        help="Path to directory containing JSONL files or single JSONL file"
    )
    
    parser.add_argument(
        "--recent-minutes",
        type=float,
        default=2.0,
        help="Time window in minutes for calculating current throughput (default: 2.0)"
    )
    
    args = parser.parse_args()
    
    # Check if input is file or directory
    input_path = Path(args.path)
    if not input_path.exists():
        parser.error(f"Path '{input_path}' not found")
    
    try:
        if input_path.is_file():
            # Single file mode
            print(f"Processing single file: {input_path.name}")
            df = read_progress_file(str(input_path))
            num_files = 1
            print(f"Loaded {len(df):,} data points")
        else:
            # Directory mode
            print(f"Discovering files in directory: {input_path}")
            file_paths = discover_progress_files(str(input_path))
            print(f"Found {len(file_paths)} files")
            
            df = load_and_combine_files(file_paths)
            num_files = len(file_paths)
            print(f"Total data points: {len(df):,}")
        
        # Calculate and display per-shard statistics
        shard_stats = calculate_per_shard_stats(df, args.recent_minutes)
        print_per_shard_stats(shard_stats)
        
        # Calculate and display current total throughput
        current_totals = calculate_current_total_throughput(df, args.recent_minutes)
        print_current_total_throughput(current_totals, args.recent_minutes)
        
        # Display summary information
        print_summary_info(df, num_files, args.recent_minutes)
        
    except Exception as e:
        parser.error(f"Error processing files: {e}")


if __name__ == "__main__":
    main() 