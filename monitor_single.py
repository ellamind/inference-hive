import polars as pl
import argparse
from datetime import datetime
from pathlib import Path
from tabulate import tabulate


def read_progress_file(file_path: str) -> pl.DataFrame:
    """Read JSON lines progress file and parse into a Polars DataFrame"""
    
    # Read the NDJSON file directly with polars
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
        
        # Extract overall throughput fields (per hour)
        pl.col("throughput").struct.field("overall").struct.field("total_tokens_per_hour").alias("overall_total_tph"),
        pl.col("throughput").struct.field("overall").struct.field("prompt_tokens_per_hour").alias("overall_prompt_tph"),
        pl.col("throughput").struct.field("overall").struct.field("completion_tokens_per_hour").alias("overall_completion_tph"),
        
        # Convert timestamp to datetime
        pl.from_epoch(pl.col("timestamp")).alias("datetime")
    ]).select([
        "shard", "num_shards", "timestamp", "datetime", "completed", "total", "new", "existing",
        "eta_seconds", "eta_formatted", "overall_requests_ps", "overall_total_tps", 
        "overall_prompt_tps", "overall_completion_tps", "overall_total_tph", "overall_prompt_tph", "overall_completion_tph"
    ])
    
    return df_expanded


def calculate_current_throughput(df: pl.DataFrame, recent_minutes: float = 2.0) -> dict:
    """Calculate current throughput from the most recent data point"""
    if len(df) == 0:
        return {"status": "No Data", "requests_ps": 0, "total_tps": 0, "prompt_tps": 0, "completion_tps": 0}
    
    # Use current time as reference point for "recent" data
    import time
    current_timestamp = time.time()
    cutoff_timestamp = current_timestamp - (recent_minutes * 60)
    
    # Get the most recent record
    latest_record = df.sort("timestamp").tail(1).row(0, named=True)
    
    # Determine status based on recent activity
    if latest_record['timestamp'] >= cutoff_timestamp:
        status = "Running"
    else:
        status = "Not Running"
    
    return {
        "status": status,
        "requests_ps": latest_record.get('overall_requests_ps', 0) or 0,
        "total_tps": latest_record.get('overall_total_tps', 0) or 0,
        "prompt_tps": latest_record.get('overall_prompt_tps', 0) or 0,
        "completion_tps": latest_record.get('overall_completion_tps', 0) or 0,
        "last_update": latest_record.get('datetime'),
        "progress_pct": (latest_record.get('completed', 0) / latest_record.get('total', 1) * 100) if latest_record.get('total', 0) > 0 else 0,
        "eta_formatted": latest_record.get('eta_formatted', 'N/A'),
        "completed": latest_record.get('completed', 0),
        "total_requests": latest_record.get('total', 0)
    }


def print_current_throughput(current_stats: dict, recent_minutes: float):
    """Print current throughput information (only when running)"""
    is_running = current_stats['status'] == "Running"
    
    if is_running:
        # Show actual current throughput table
        title = f"CURRENT THROUGHPUT (last {recent_minutes:.1f} min)"
        table_data = [
            ["RPS", f"{current_stats['requests_ps']:,.1f}"],
            ["Total TPS", f"{current_stats['total_tps']:,.1f}"],
            ["Prompt TPS", f"{current_stats['prompt_tps']:,.1f}"],
            ["Completion TPS", f"{current_stats['completion_tps']:,.1f}"]
        ]
        
        print(f"\n{title}")
        print(f"{'='*50}")
        print(tabulate(table_data, headers=["Metric", "Value"], tablefmt="grid"))


def print_shard_progress(current_stats: dict, df: pl.DataFrame):
    """Print shard progress and status information"""
    if len(df) == 0:
        print("\nNo data available")
        return
    
    # Get shard information
    first_record = df.row(0, named=True)
    shard_id = first_record.get('shard', 'Unknown')
    
    # Calculate duration
    first_timestamp = df.select(pl.col("timestamp").min()).item()
    last_timestamp = df.select(pl.col("timestamp").max()).item()
    duration_minutes = (last_timestamp - first_timestamp) / 60
    
    table_data = [
        ["Status", current_stats['status']],
        ["Shard ID", f"{shard_id}"],
        ["Progress", f"{current_stats['progress_pct']:.1f}%"],
        ["Completed", f"{current_stats['completed']:,}"],
        ["Total Requests", f"{current_stats['total_requests']:,}"],
        ["ETA", current_stats['eta_formatted']],
        ["Last Update", current_stats['last_update'].strftime('%Y-%m-%d %H:%M:%S') if current_stats['last_update'] else 'N/A'],
        ["Duration", f"{duration_minutes:.1f} minutes"]
    ]
    
    print(f"\nSHARD PROGRESS & STATUS")
    print(f"{'='*40}")
    print(tabulate(table_data, headers=["Field", "Value"], tablefmt="grid"))


def print_throughput_summary(df: pl.DataFrame, use_hours=False):
    """Print a summary table of key throughput statistics"""
    
    if len(df) == 0:
        print("No data to analyze")
        return
    
    # Choose columns based on time unit
    if use_hours:
        total_col = "overall_total_tph"
        prompt_col = "overall_prompt_tph"
        completion_col = "overall_completion_tph"
        unit = "per Hour"
    else:
        total_col = "overall_total_tps"
        prompt_col = "overall_prompt_tps"
        completion_col = "overall_completion_tps"
        unit = "per Second"
    
    # Calculate statistics
    stats = df.select([
        pl.col(total_col).mean().alias("avg_total"),
        pl.col(total_col).max().alias("max_total"),
        pl.col(total_col).min().alias("min_total"),
        pl.col(prompt_col).mean().alias("avg_prompt"),
        pl.col(prompt_col).max().alias("max_prompt"),
        pl.col(prompt_col).min().alias("min_prompt"),
        pl.col(completion_col).mean().alias("avg_completion"),
        pl.col(completion_col).max().alias("max_completion"),
        pl.col(completion_col).min().alias("min_completion"),
    ]).row(0, named=True)
    
    metric_suffix = "TPH" if use_hours else "TPS"
    
    # Prepare data for table
    table_data = [
        [f"Total {metric_suffix}", f"{stats['avg_total']:,.0f}", f"{stats['min_total']:,.0f}", f"{stats['max_total']:,.0f}"],
        [f"Prompt {metric_suffix}", f"{stats['avg_prompt']:,.0f}", f"{stats['min_prompt']:,.0f}", f"{stats['max_prompt']:,.0f}"],
        [f"Completion {metric_suffix}", f"{stats['avg_completion']:,.0f}", f"{stats['min_completion']:,.0f}", f"{stats['max_completion']:,.0f}"]
    ]
    
    print(f"\nTHROUGHPUT STATISTICS (Tokens {unit})")
    print(f"{'='*50}")
    print(tabulate(table_data, headers=["Metric", "Average", "Minimum", "Maximum"], tablefmt="grid"))


def create_horizontal_throughput_charts(df: pl.DataFrame, plot_lines=None, bar_width=50, use_hours=False):
    """Create three separate horizontal bar charts with time on y-axis using time-based aggregation"""
    
    if len(df) == 0:
        return
    
    # Determine number of plot lines to use
    n_points = min(len(df), plot_lines)
    use_aggregation = len(df) > plot_lines
    
    if n_points <= 1:
        print("Not enough data points for chart")
        return
    
    # Choose columns based on time unit
    if use_hours:
        total_col = "overall_total_tph"
        prompt_col = "overall_prompt_tph"
        completion_col = "overall_completion_tph"
        unit_suffix = "per Hour"
    else:
        total_col = "overall_total_tps"
        prompt_col = "overall_prompt_tps"
        completion_col = "overall_completion_tps"
        unit_suffix = "per Second"
    
    if use_aggregation:
        # Time-based aggregation approach
        min_timestamp = df.select(pl.col("timestamp").min()).item()
        max_timestamp = df.select(pl.col("timestamp").max()).item()
        
        # Create time buckets
        bucket_duration = (max_timestamp - min_timestamp) / n_points
        
        # Add bucket assignment to dataframe
        df_with_buckets = df.with_columns([
            ((pl.col("timestamp") - min_timestamp) / bucket_duration).floor().cast(pl.Int32).alias("bucket")
        ])
        
        # Aggregate data by bucket
        aggregated_df = df_with_buckets.group_by("bucket").agg([
            pl.col("timestamp").mean().alias("timestamp"),
            pl.col(total_col).mean().alias(total_col),
            pl.col(prompt_col).mean().alias(prompt_col),
            pl.col(completion_col).mean().alias(completion_col)
        ]).sort("bucket")
        
        # Get the aggregated data series
        total_tps = aggregated_df.select(total_col).to_series().to_list()
        prompt_tps = aggregated_df.select(prompt_col).to_series().to_list()
        completion_tps = aggregated_df.select(completion_col).to_series().to_list()
        timestamps = aggregated_df.select("timestamp").to_series().to_list()
        
    else:
        # Simple sampling for smaller datasets
        step = len(df) // n_points
        sampled_df = df.gather_every(step).head(n_points)
        
        total_tps = sampled_df.select(total_col).to_series().to_list()
        prompt_tps = sampled_df.select(prompt_col).to_series().to_list()
        completion_tps = sampled_df.select(completion_col).to_series().to_list()
        timestamps = sampled_df.select("timestamp").to_series().to_list()
    
    # Create time labels with date for jobs that might run longer than 24 hours
    time_labels = [datetime.fromtimestamp(ts).strftime('%m-%d %H:%M:%S') for ts in timestamps]
    
    def create_horizontal_bar_chart(values, title, max_val=None):
        if max_val is None:
            max_val = max(values) if values else 1
        
        aggregation_note = " (Time-Aggregated)" if use_aggregation else ""
        print(f"\n{title}{aggregation_note}")
        print("=" * len(title + aggregation_note))
        print(f"{'Time':<12} {'Value':<8} Bar (0 to {max_val:,.0f})")
        print("-" * (bar_width + 27))
        
        for i, (time_label, value) in enumerate(zip(time_labels, values)):
            # Calculate bar length
            if max_val > 0:
                bar_length = int((value / max_val) * bar_width)
            else:
                bar_length = 0
            
            # Create the bar
            bar = "█" * bar_length
            
            print(f"{time_label:<12} {value:>7,.0f} |{bar:<{bar_width}}")
    
    # Find global max for consistent scaling
    all_values = total_tps + prompt_tps + completion_tps
    global_max = max(all_values) if all_values else 1
    
    # Create the three charts
    create_horizontal_bar_chart(total_tps, f"Total Tokens {unit_suffix} Over Time", global_max)
    create_horizontal_bar_chart(prompt_tps, f"Prompt Tokens {unit_suffix} Over Time", global_max)
    create_horizontal_bar_chart(completion_tps, f"Completion Tokens {unit_suffix} Over Time", global_max)


def main():
    parser = argparse.ArgumentParser(
        description="Analyze throughput data from JSON lines progress file",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument(
        "progress_file",
        help="Path to the JSON lines progress file to analyze"
    )
    
    parser.add_argument(
        "--plot-lines",
        type=int,
        default=10,
        help="Maximum number of lines to display in charts. If more data points exist, they will be aggregated into time buckets (default: 10)"
    )
    
    parser.add_argument(
        "--bar-width",
        type=int,
        default=50,
        help="Width of horizontal bars in characters (default: 50)"
    )
    
    parser.add_argument(
        "--hours",
        action="store_true",
        help="Display throughput values per hour instead of per second"
    )
    
    parser.add_argument(
        "--recent-minutes",
        type=float,
        default=2.0,
        help="Time window in minutes for calculating current throughput (default: 2.0)"
    )
    
    args = parser.parse_args()
    
    # Check if file exists
    file_path = Path(args.progress_file)
    if not file_path.exists():
        parser.error(f"File '{file_path}' not found")
    
    try:
        df = read_progress_file(str(file_path))
        print(f"Loaded {len(df):,} data points from {file_path.name}")
        
        # Calculate and display current status
        current_stats = calculate_current_throughput(df, args.recent_minutes)
        print_current_throughput(current_stats, args.recent_minutes)
        print_shard_progress(current_stats, df)
        
        # Display throughput statistics
        print_throughput_summary(df, use_hours=args.hours)
        
        # Create charts
        create_horizontal_throughput_charts(df, plot_lines=args.plot_lines, bar_width=args.bar_width, use_hours=args.hours)
        
    except Exception as e:
        parser.error(f"Error processing file: {e}")


if __name__ == "__main__":
    main() 