import argparse
import asyncio
import json
import os
import signal
import time

from loguru import logger
from openai import AsyncOpenAI
import udf

from data_utils import DatasetReader, DatasetWriter, NoDatasetFilesError, load_data
from schemas import CHAT_COMPLETION_SCHEMA, COMPLETION_SCHEMA
from validate_data import validate_input_data_format
from config import load_inference_config

def _setup_signal_handlers(writer=None):
    """Setup signal handlers to gracefully close writer and exit cleanly"""
    def handle_shutdown_signal(signum, frame):
        signal_name = signal.Signals(signum).name
        logger.info(f"Received {signal_name}. Emergency shutdown - preserving checkpoints for recovery.")
        if writer is not None and not writer._closed:
            try:
                # Use emergency mode to shutdown asap.
                writer.close(emergency=True)
                logger.info("Writer emergency close completed successfully.")
            except Exception as e:
                logger.error(f"Error during emergency close: {e}")
        logger.info(f"Exiting due to {signal_name}. Checkpoint files preserved for recovery.")
        os._exit(1)
    
    signal.signal(signal.SIGTERM, handle_shutdown_signal)
    signal.signal(signal.SIGINT, handle_shutdown_signal)
    logger.debug("Configured signal handler.")


def format_time(seconds):
    """Format seconds into a readable time string"""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.1f}h"


class APIFailureCounter:
    """Simple counter to track consecutive API failures"""

    def __init__(self, max_failures: int):
        self.max_failures = max_failures
        self.consecutive_failures = 0

    def record_failure(self):
        """Record an API failure"""
        self.consecutive_failures += 1
        if self.consecutive_failures >= self.max_failures:
            raise RuntimeError(
                f"API appears to be down: {self.consecutive_failures} consecutive failures. Terminating to preserve data."
            )

    def record_success(self):
        """Record a successful API call"""
        if self.consecutive_failures > 0:
            logger.info(f"API recovered after {self.consecutive_failures} failures")
            self.consecutive_failures = 0


class ProgressLogger:
    """Logger for structured progress data in JSONL format"""
    
    def __init__(self, log_file_path: str, shard: int, num_shards: int):
        self.log_file_path = log_file_path
        self.shard = shard
        self.num_shards = num_shards
        self.file_handle = None
        
        # Token tracking for overall stats
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.total_tokens = 0
        
        # Token tracking for interval stats
        self.interval_prompt_tokens = 0
        self.interval_completion_tokens = 0
        self.interval_tokens = 0
    
    def __enter__(self):
        self.file_handle = open(self.log_file_path, 'w')
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.file_handle:
            self.file_handle.close()
    
    def add_token_usage(self, usage_dict):
        """Add token usage from an API response"""
        if usage_dict:
            prompt_tokens = usage_dict.get('prompt_tokens', 0)
            completion_tokens = usage_dict.get('completion_tokens', 0)
            total_tokens = usage_dict.get('total_tokens', 0)
            
            # Add to overall totals
            self.total_prompt_tokens += prompt_tokens
            self.total_completion_tokens += completion_tokens
            self.total_tokens += total_tokens
            
            # Add to interval totals
            self.interval_prompt_tokens += prompt_tokens
            self.interval_completion_tokens += completion_tokens
            self.interval_tokens += total_tokens
    
    def reset_interval_tokens(self):
        """Reset interval token counters"""
        self.interval_prompt_tokens = 0
        self.interval_completion_tokens = 0
        self.interval_tokens = 0
    
    def log_progress(self, completed, total, new, existing, eta_seconds, eta_formatted,
                    overall_rate, interval_rate, interval_duration, interval_new, 
                    interval_existing, start_time):
        """Log a progress entry in JSONL format"""
        if not self.file_handle:
            return
            
        current_time = time.time()
        elapsed_time = current_time - start_time
        
        # Calculate token rates
        overall_prompt_rate = self.total_prompt_tokens / elapsed_time if elapsed_time > 0 else 0
        overall_completion_rate = self.total_completion_tokens / elapsed_time if elapsed_time > 0 else 0
        overall_token_rate = self.total_tokens / elapsed_time if elapsed_time > 0 else 0
        
        interval_prompt_rate = self.interval_prompt_tokens / interval_duration if interval_duration > 0 else 0
        interval_completion_rate = self.interval_completion_tokens / interval_duration if interval_duration > 0 else 0
        interval_token_rate = self.interval_tokens / interval_duration if interval_duration > 0 else 0
        
        progress_entry = {
            "timestamp": current_time,
            "shard": self.shard,
            "num_shards": self.num_shards,
            "progress": {
                "completed": completed,
                "total": total,
                "new": new,
                "existing": existing,
                "eta_seconds": eta_seconds,
                "eta_formatted": eta_formatted
            },
            "throughput": {
                "overall": {
                    "requests_per_second": overall_rate,
                    "total_tokens_per_second": overall_token_rate,
                    "prompt_tokens_per_second": overall_prompt_rate,
                    "completion_tokens_per_second": overall_completion_rate,
                    "requests_per_hour": overall_rate * 3600,
                    "total_tokens_per_hour": overall_token_rate * 3600,
                    "prompt_tokens_per_hour": overall_prompt_rate * 3600,
                    "completion_tokens_per_hour": overall_completion_rate * 3600
                },
                "last_interval": {
                    "duration_seconds": interval_duration,
                    "new": interval_new,
                    "existing": interval_existing,
                    "requests_per_second": interval_rate,
                    "total_tokens_per_second": interval_token_rate,
                    "prompt_tokens_per_second": interval_prompt_rate,
                    "completion_tokens_per_second": interval_completion_rate,
                    "requests_per_hour": interval_rate * 3600,
                    "total_tokens_per_hour": interval_token_rate * 3600,
                    "prompt_tokens_per_hour": interval_prompt_rate * 3600,
                    "completion_tokens_per_hour": interval_completion_rate * 3600
                }
            }
        }
        
        # Write JSONL entry
        json.dump(progress_entry, self.file_handle)
        self.file_handle.write('\n')
        self.file_handle.flush()


def _load_and_validate_dataset(config, args: argparse.Namespace):
    """Load dataset and validate format, returning dataset and total length."""
    ds = load_data(config, args.shard, args.num_shards)
    logger.info(f"Dataset:\n{ds}")

    # Build a small concrete sample (no generator) for validation
    sample_rows = []
    try:
        udf_func = getattr(udf, config.apply_udf) if config.apply_udf else None
    except AttributeError:
        logger.error(f"UDF function '{config.apply_udf}' not found in udf.py")
        raise
    for row in ds:
        if udf_func is not None:
            try:
                row = udf_func(row, **(config.apply_udf_kwargs or {}))
            except Exception as e:
                logger.error(f"UDF error during validation: {e}")
                raise
        sample_rows.append(row)
        if len(sample_rows) >= 1000:
            break

    validate_input_data_format(
        sample_rows,
        config.input_column_name,
        config.id_column_name,
        config.api_type,
        log_samples=False,
    )
    return ds


def _read_existing_ids(output_path: str, shard: int) -> set:
    """Read existing IDs from an existing output dataset directory."""
    existing_ids = set()
    try:
        reader = DatasetReader(output_path, columns=['id'], glob_pattern=f"**/shard{shard:06d}*.parquet")
        logger.info(
            f"Found existing output in {output_path} with {len(reader):_} rows"
        )
        logger.info("Reading existing IDs")
        for row in reader:
            existing_ids.add(row["id"])
        logger.info(f"Existing IDs: {len(existing_ids):_}")
    except NoDatasetFilesError:
        logger.info(f"No existing output found in {output_path}")
    return existing_ids


async def run_inference_async(config, args: argparse.Namespace):
    """Run the asynchronous inference pipeline."""
    _load_and_validate_dataset(config, args)
    existing_ids = _read_existing_ids(config.output_path, args.shard)

    api_key = os.environ.get("API_KEY", "EMPTY")
    client = AsyncOpenAI(
        base_url=config.api_base_url, api_key=api_key, max_retries=config.max_retries
    )

    schema = (
        CHAT_COMPLETION_SCHEMA
        if config.api_type == "chat-completion"
        else COMPLETION_SCHEMA
    )
    writer = DatasetWriter(dataset_dir=config.output_path, schema=schema, shard=args.shard)

    # Setup signal handlers to ensure writer is closed on shutdown
    _setup_signal_handlers(writer)

    # Initialize progress logger if specified
    progress_logger = None
    if args.log_file:
        progress_logger = ProgressLogger(args.log_file, args.shard, args.num_shards)

    try:
        semaphore = asyncio.Semaphore(config.max_connections)

        # Build raw dataset for processing (apply UDF per-row in workers)
        raw_ds = DatasetReader(config.dataset_path, shard=args.shard, num_shards=args.num_shards, **(config.dataset_kwargs or {}))

        # Progress tracking
        total_rows = len(raw_ds)
        processed_rows = 0
        skipped_rows = 0
        progress_lock = asyncio.Lock()
        first_completion_time = None

        # Track progress for interval-based rate calculation
        last_report_time = None
        last_report_processed_rows = 0

        api_failure_counter = APIFailureCounter(config.max_consecutive_failures)
        fatal_error_event = asyncio.Event()

        # Resolve UDF function once
        udf_func = None
        if config.apply_udf:
            try:
                udf_func = getattr(udf, config.apply_udf)
            except AttributeError:
                logger.error(f"UDF function '{config.apply_udf}' not found in udf.py")
                raise

        async def request(row):
            nonlocal processed_rows, skipped_rows, first_completion_time

            async with semaphore:
                row_id = row[config.id_column_name]
                if row_id in existing_ids:
                    async with progress_lock:
                        skipped_rows += 1
                    return

                try:
                    # Apply UDF per-row (guarding against bad text/formatting)
                    if udf_func is not None:
                        row = udf_func(row, **(config.apply_udf_kwargs or {}))

                    if config.api_type == "chat-completion":
                        response = await client.chat.completions.create(
                            model=config.model, messages=row[config.input_column_name], **config.completions_kwargs
                        )
                    elif config.api_type == "completion":
                        response = await client.completions.create(
                            model=config.model, prompt=row[config.input_column_name], **config.completions_kwargs
                        )
                    else:
                        raise ValueError(f"Invalid API type: {config.api_type}")

                    api_failure_counter.record_success()
                    output_dict = {"response": response.model_dump(
                        exclude=response.model_extra.keys()
                    )}
                    output_dict["id"] = row_id
                    writer.add_row(output_dict)

                    if progress_logger and hasattr(response, 'usage') and response.usage:
                        usage_dict = response.usage.model_dump() if hasattr(response.usage, 'model_dump') else response.usage
                        progress_logger.add_token_usage(usage_dict)

                    async with progress_lock:
                        processed_rows += 1
                        current_time = time.time()
                        if first_completion_time is None:
                            first_completion_time = current_time

                except Exception as e:
                    error_str = str(e).lower()
                    is_api_error = any(
                        keyword in error_str
                        for keyword in [
                            "connection",
                            "timeout",
                            "network",
                            "unreachable",
                            "refused",
                            "reset",
                            "unavailable",
                            "service temporarily unavailable",
                        ]
                    )

                    if is_api_error:
                        try:
                            api_failure_counter.record_failure()
                        except RuntimeError as fatal:
                            # Signal fatal outage so the run can abort with non-zero exit
                            fatal_error_event.set()
                            logger.error(str(fatal))
                            raise
                        logger.warning(f"API connection issue: {e}")
                        raise
                    else:
                        logger.error(f"API Error, skipping request: {e}")
                        async with progress_lock:
                            skipped_rows += 1

        async def run_inference():
            async def report_progress(is_final=False):
                nonlocal last_report_time, last_report_processed_rows
                async with progress_lock:
                    current_time = time.time()
                    total_completed = processed_rows + skipped_rows

                    if processed_rows > 0 and first_completion_time is not None:
                        elapsed_since_first_completion = current_time - first_completion_time
                        overall_rate = processed_rows / elapsed_since_first_completion
                        overall_rate_per_hour = overall_rate * 3600

                        interval_rate = 0.0
                        interval_rate_per_hour = 0.0
                        if last_report_time is not None:
                            interval_duration = current_time - last_report_time
                            if interval_duration > 0:
                                interval_processed = processed_rows - last_report_processed_rows
                                interval_rate = interval_processed / interval_duration
                                interval_rate_per_hour = interval_rate * 3600

                        remaining = total_rows - total_completed
                        eta_interval = remaining / interval_rate if interval_rate > 0 else 0

                        failure_msg = ""
                        if api_failure_counter.consecutive_failures > 0:
                            failure_msg = f", {api_failure_counter.consecutive_failures} consecutive failures"

                        prefix = "Final Progress" if is_final else "Progress"

                        if is_final:
                            logger.info(
                                f"{prefix}: {total_completed:_}/{total_rows:_} completed "
                                f"({processed_rows:_} new, {skipped_rows:_} existing), "
                                f"Overall: {overall_rate:.1f} reqs/s ({overall_rate_per_hour:_.0f} reqs/h){failure_msg}",
                                flush=True
                            )
                        else:
                            if last_report_time is None:
                                last_report_time = first_completion_time
                                last_report_processed_rows = 0
                                interval_duration = current_time - last_report_time
                                if interval_duration > 0:
                                    interval_processed = processed_rows - last_report_processed_rows
                                    interval_rate = interval_processed / interval_duration
                                    interval_rate_per_hour = interval_rate * 3600
                                    eta_interval = remaining / interval_rate if interval_rate > 0 else 0

                            logger.info(
                                f"{prefix}: {total_completed:_}/{total_rows:_} completed "
                                f"({processed_rows:_} new, {skipped_rows:_} existing), "
                                f"Overall: {overall_rate:.1f} reqs/s ({overall_rate_per_hour:_.0f} reqs/h), "
                                f"Last {args.progress_report_interval}s: {interval_rate:.1f} reqs/s ({interval_rate_per_hour:_.0f} reqs/h), "
                                f"ETA: {format_time(eta_interval)}{failure_msg}",
                                flush=True
                            )

                        if progress_logger and processed_rows > 0 and first_completion_time is not None:
                            if last_report_time is not None and not is_final:
                                interval_duration = current_time - last_report_time
                                interval_new = processed_rows - last_report_processed_rows
                                interval_existing = 0
                            else:
                                interval_duration = current_time - first_completion_time
                                interval_new = processed_rows
                                interval_existing = skipped_rows

                            progress_logger.log_progress(
                                completed=total_completed,
                                total=total_rows,
                                new=processed_rows,
                                existing=skipped_rows,
                                eta_seconds=eta_interval,
                                eta_formatted=format_time(eta_interval),
                                overall_rate=overall_rate,
                                interval_rate=interval_rate,
                                interval_duration=interval_duration,
                                interval_new=interval_new,
                                interval_existing=interval_existing,
                                start_time=first_completion_time
                            )

                        if progress_logger and not is_final:
                            progress_logger.reset_interval_tokens()

                        if not is_final:
                            last_report_time = current_time
                            last_report_processed_rows = processed_rows
                    elif processed_rows == 0:
                        logger.info(
                            f"Progress: {total_completed:_}/{total_rows:_} completed "
                            f"({processed_rows:_} new, {skipped_rows:_} existing), "
                            f"Waiting for first API completion...",
                            flush=True
                        )

            async def progress_reporter(scheduling_done_event: asyncio.Event, inflight_tasks: set):
                while True:
                    await asyncio.sleep(args.progress_report_interval)
                    if fatal_error_event.is_set():
                        logger.info("Fatal error detected, stopping progress reporter")
                        break
                    await report_progress()
                    if scheduling_done_event.is_set() and not inflight_tasks:
                        logger.info("All rows processed, exiting")
                        break

            logger.info("Starting inference")
            start_time = time.time()

            # Iterator and bounded inflight scheduling
            ds_iter = iter(raw_ds)
            inflight: set[asyncio.Task] = set()
            scheduling_done = asyncio.Event()

            async def start_next() -> bool:
                if fatal_error_event.is_set():
                    return False
                try:
                    row = next(ds_iter)
                except StopIteration:
                    scheduling_done.set()
                    return False
                task = asyncio.create_task(request(row))
                inflight.add(task)
                def _discard(t: asyncio.Task):
                    inflight.discard(t)
                task.add_done_callback(_discard)
                return True

            # Prime up to max_connections
            for _ in range(config.max_connections):
                created = await start_next()
                if not created:
                    break

            reporter_task = asyncio.create_task(progress_reporter(scheduling_done, inflight))

            # Drive pipeline
            while inflight and not fatal_error_event.is_set():
                done, _ = await asyncio.wait(inflight, return_when=asyncio.FIRST_COMPLETED)
                # For each completed task, try to start one new task
                for t in done:
                    exc = t.exception()
                    if exc is not None and fatal_error_event.is_set():
                        break
                    await start_next()

            # Handle fatal outage: cancel remaining tasks
            if fatal_error_event.is_set():
                for t in list(inflight):
                    if not t.done():
                        t.cancel()
                if inflight:
                    await asyncio.gather(*inflight, return_exceptions=True)
                if not reporter_task.done():
                    reporter_task.cancel()
                raise RuntimeError("API appears to be down: aborting shard to preserve data")

            # Await remaining tasks
            if inflight:
                await asyncio.gather(*inflight, return_exceptions=True)

            # Final report and stop reporter
            if not reporter_task.done():
                await report_progress(is_final=True)
                reporter_task.cancel()

            total_time = time.time() - start_time
            if first_completion_time is not None:
                processing_time = time.time() - first_completion_time
                logger.info(
                    f"Done. {processed_rows:_} processed, {skipped_rows:_} skipped, "
                    f"Total time: {format_time(total_time)}, Processing time: {format_time(processing_time)}"
                )
            else:
                logger.info(
                    f"Done. {processed_rows:_} processed, {skipped_rows:_} skipped, Total time: {format_time(total_time)}"
                )

            # Return progress counters for completion verification
            return processed_rows, skipped_rows, total_rows

        if progress_logger:
            with progress_logger:
                processed_rows, skipped_rows, total_rows = await run_inference()
        else:
            processed_rows, skipped_rows, total_rows = await run_inference()


        logger.info("Closing writer...")
        writer.close()
        logger.info("Writer closed successfully")
        
        # If we exit without processing the full shard, treat as failure so the
        # job wrapper won't mark the shard as completed.
        if (processed_rows + skipped_rows) < total_rows:
            raise RuntimeError(
                f"Shard incomplete: completed {processed_rows + skipped_rows:_}/{total_rows:_} rows"
            )


    except Exception as e:
        logger.error(f"An error occurred during inference: {e}")
        if writer is not None and not writer._closed:
            logger.info("Closing writer with emergency flag...")
            writer.close(emergency=True)
            logger.info("Writer emergency close completed successfully.")
        raise
    finally:
        logger.info("Shutdown completed")


def main(config, args: argparse.Namespace):
    asyncio.run(run_inference_async(config, args))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=str, required=True, help="Path to YAML configuration file"
    )
    parser.add_argument(
        "--num-shards", type=int, default=1, help="Total number of shards"
    )
    parser.add_argument("--shard", type=int, default=0, help="Shard number")
    parser.add_argument(
        "--progress-report-interval", 
        type=int, 
        default=60,
        help="Interval in seconds for progress reports"
    )
    parser.add_argument(
        "--log-file",
        type=str,
        help="Path to write structured progress logs in JSONL format"
    )
    args = parser.parse_args()

    config = load_inference_config(args.config)

    # Validate shard parameter
    if args.shard >= args.num_shards or args.shard < 0:
        raise ValueError(
            f"shard ({args.shard}) must be between 0 and {args.num_shards - 1}"
        )

    logger.info(f"Configuration:\n{config.model_dump_json(indent=4)}")
    logger.info(f"Dataset shard {args.shard}")
    logger.info(f"Number of shards: {args.num_shards}")

    main(config, args)
