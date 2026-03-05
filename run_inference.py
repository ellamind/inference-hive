import argparse
import asyncio
import concurrent.futures
import json
import multiprocessing
import os
import signal
import sys
import time

from loguru import logger
from openai import AsyncOpenAI, DefaultAioHttpClient
from inference_hive import udf

from inference_hive.data_utils import DatasetReader, DatasetWriter, NoDatasetFilesError, load_data
from inference_hive.schemas import CHAT_COMPLETION_SCHEMA, COMPLETION_SCHEMA
from validate_data import validate_input_data_format
from inference_hive.config import load_inference_config

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
    """Run the asynchronous inference pipeline (single-worker path)."""
    _load_and_validate_dataset(config, args)
    existing_ids = _read_existing_ids(config.output_path, args.shard)

    api_key = os.environ.get("API_KEY", "EMPTY")
    api_base_url = os.environ.get('API_BASE_URL', config.api_base_url)

    schema = (
        CHAT_COMPLETION_SCHEMA
        if config.api_type == "chat-completion"
        else COMPLETION_SCHEMA
    )
    writer = DatasetWriter(dataset_dir=config.output_path, schema=schema, shard=args.shard, batch_size=10_000)

    _setup_signal_handlers(writer)

    progress_logger = None
    if args.log_file:
        progress_logger = ProgressLogger(args.log_file, args.shard, args.num_shards)

    client = AsyncOpenAI(
        base_url=api_base_url,
        api_key=api_key,
        max_retries=config.max_retries,
        http_client=DefaultAioHttpClient(),
    )

    try:
        semaphore = asyncio.Semaphore(config.max_connections)

        ds = load_data(config, args.shard, args.num_shards)

        # Progress tracking
        total_rows = len(ds)
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

                try:
                    # Apply UDF per-row (guarding against bad text/formatting)
                    if udf_func is not None:
                        row = udf_func(row, **(config.apply_udf_kwargs or {}))

                    if config.api_type == "chat-completion":
                        coro = client.chat.completions.create(
                            model=config.model, messages=row[config.input_column_name], **config.completions_kwargs
                        )
                    elif config.api_type == "completion":
                        coro = client.completions.create(
                            model=config.model, prompt=row[config.input_column_name], **config.completions_kwargs
                        )
                    else:
                        raise ValueError(f"Invalid API type: {config.api_type}")

                    response = await asyncio.wait_for(coro, timeout=300)

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

                except asyncio.TimeoutError:
                    logger.warning(f"Request timed out after 300s for row {row_id}, skipping")
                    try:
                        api_failure_counter.record_failure()
                    except RuntimeError as fatal:
                        fatal_error_event.set()
                        logger.error(str(fatal))
                        raise
                    async with progress_lock:
                        skipped_rows += 1
                except Exception as e:
                    error_str = str(e).lower()

                    is_context_length_error = any(
                        kw in error_str for kw in ["context length", "context_length", "too many tokens"]
                    )
                    if is_context_length_error:
                        logger.warning(f"Row {row_id} exceeds model context length, skipping permanently: {e}")
                        async with progress_lock:
                            skipped_rows += 1
                        return

                    is_api_error = any(
                        keyword in error_str
                        for keyword in [
                            "connection",
                            "timeout",
                            "timed out",
                            "request timed out",
                            "read timed out",
                            "gateway timeout",
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
                            fatal_error_event.set()
                            logger.error(str(fatal))
                            raise
                        logger.warning(f"API connection issue for row {row_id}, skipping: {e}")
                        async with progress_lock:
                            skipped_rows += 1
                        return
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
            ds_iter = iter(ds)
            inflight: set[asyncio.Task] = set()
            scheduling_done = asyncio.Event()

            async def start_next() -> bool:
                nonlocal skipped_rows
                if fatal_error_event.is_set():
                    return False
                while True:
                    try:
                        row = next(ds_iter)
                    except StopIteration:
                        scheduling_done.set()
                        return False
                    row_id = row[config.id_column_name]
                    if row_id in existing_ids:
                        skipped_rows += 1
                        continue
                    break
                task = asyncio.create_task(request(row))
                inflight.add(task)
                def _discard(t: asyncio.Task):
                    inflight.discard(t)
                task.add_done_callback(_discard)
                return True

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
                logger.error("Fatal error detected, cancelling remaining tasks and shutting down...")
                for t in list(inflight):
                    if not t.done():
                        t.cancel()
                if inflight:
                    # Use wait_for with timeout to avoid hanging forever on stubborn tasks
                    try:
                        await asyncio.wait_for(
                            asyncio.gather(*inflight, return_exceptions=True),
                            timeout=10.0
                        )
                    except asyncio.TimeoutError:
                        logger.warning("Timeout waiting for tasks to cancel, proceeding with shutdown")
                if not reporter_task.done():
                    reporter_task.cancel()
                    try:
                        await asyncio.wait_for(asyncio.shield(reporter_task), timeout=2.0)
                    except (asyncio.TimeoutError, asyncio.CancelledError):
                        pass
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
        await client.close()
        logger.info("Shutdown completed")


def _worker_main(read_queue, write_queue, config_path, args_dict, effective_connections, shared_token_counts=None, error_count=None):
    """Worker subprocess: pulls rows from read_queue, makes API calls, pushes results to write_queue."""
    config = load_inference_config(config_path)

    api_key = os.environ.get("API_KEY", "EMPTY")
    api_base_url = os.environ.get('API_BASE_URL', config.api_base_url)

    udf_func = None
    if config.apply_udf:
        try:
            udf_func = getattr(udf, config.apply_udf)
        except AttributeError:
            logger.error(f"UDF function '{config.apply_udf}' not found in udf.py")
            write_queue.put(None)
            sys.exit(1)

    api_failure_counter = APIFailureCounter(config.max_consecutive_failures)

    async def _run():
        loop = asyncio.get_running_loop()
        loop.set_default_executor(concurrent.futures.ThreadPoolExecutor(max_workers=128))
        client = AsyncOpenAI(
            base_url=api_base_url,
            api_key=api_key,
            max_retries=config.max_retries,
            http_client=DefaultAioHttpClient(),
        )
        try:
            semaphore = asyncio.Semaphore(effective_connections)
            fatal_error_event = asyncio.Event()
            inflight: set[asyncio.Task] = set()
            scheduling_done = asyncio.Event()

            # Pre-fetch buffer: decouple IPC reads from the scheduling loop
            # so start_next() never blocks on multiprocessing queue get()
            local_queue = asyncio.Queue(maxsize=effective_connections * 2)

            async def prefetch_rows():
                try:
                    while not fatal_error_event.is_set():
                        row = await asyncio.to_thread(read_queue.get)
                        await local_queue.put(row)
                        if row is None:
                            break
                except Exception as e:
                    logger.error(f"Prefetch error: {e}")
                    await local_queue.put(None)

            prefetch_task = asyncio.create_task(prefetch_rows())

            async def do_request(row):
                row_id = row[config.id_column_name]
                output_dict = None
                try:
                    if udf_func is not None:
                        row = udf_func(row, **(config.apply_udf_kwargs or {}))

                    async with semaphore:
                        if config.api_type == "chat-completion":
                            coro = client.chat.completions.create(
                                model=config.model,
                                messages=row[config.input_column_name],
                                **config.completions_kwargs,
                            )
                        elif config.api_type == "completion":
                            coro = client.completions.create(
                                model=config.model,
                                prompt=row[config.input_column_name],
                                **config.completions_kwargs,
                            )
                        else:
                            raise ValueError(f"Invalid API type: {config.api_type}")

                        response = await asyncio.wait_for(coro, timeout=300)

                    api_failure_counter.record_success()
                    output_dict = {
                        "id": row_id,
                        "response": response.model_dump(exclude=response.model_extra.keys()),
                    }

                    if shared_token_counts is not None and hasattr(response, 'usage') and response.usage:
                        pt = response.usage.prompt_tokens or 0
                        ct = response.usage.completion_tokens or 0
                        tt = response.usage.total_tokens or 0
                        with shared_token_counts[0].get_lock():
                            shared_token_counts[0].value += pt
                        with shared_token_counts[1].get_lock():
                            shared_token_counts[1].value += ct
                        with shared_token_counts[2].get_lock():
                            shared_token_counts[2].value += tt

                except asyncio.TimeoutError:
                    logger.warning(f"Request timed out after 300s for row {row_id}, skipping")
                    try:
                        api_failure_counter.record_failure()
                    except RuntimeError:
                        fatal_error_event.set()
                        raise
                    if error_count is not None:
                        with error_count.get_lock():
                            error_count.value += 1
                except Exception as e:
                    error_str = str(e).lower()

                    is_context_length_error = any(
                        kw in error_str for kw in ["context length", "context_length", "too many tokens"]
                    )
                    if is_context_length_error:
                        logger.warning(f"Row {row_id} exceeds model context length, skipping permanently: {e}")
                        if error_count is not None:
                            with error_count.get_lock():
                                error_count.value += 1
                        return

                    is_api_error = any(
                        kw in error_str
                        for kw in [
                            "connection", "timeout", "timed out",
                            "request timed out", "read timed out",
                            "gateway timeout", "network", "unreachable",
                            "refused", "reset", "unavailable",
                            "service temporarily unavailable",
                        ]
                    )
                    if is_api_error:
                        try:
                            api_failure_counter.record_failure()
                        except RuntimeError:
                            fatal_error_event.set()
                            raise
                        logger.warning(f"API error for row {row_id}, skipping: {e}")
                    else:
                        logger.error(f"API Error, skipping request: {e}")
                    if error_count is not None:
                        with error_count.get_lock():
                            error_count.value += 1

                if output_dict is not None:
                    await asyncio.to_thread(write_queue.put, output_dict)

            async def start_next() -> bool:
                if fatal_error_event.is_set():
                    return False
                row = await local_queue.get()
                if row is None:
                    scheduling_done.set()
                    return False
                task = asyncio.create_task(do_request(row))
                inflight.add(task)
                task.add_done_callback(lambda t: inflight.discard(t))
                return True

            for _ in range(effective_connections):
                if not await start_next():
                    break

            while inflight and not fatal_error_event.is_set():
                done, _ = await asyncio.wait(inflight, return_when=asyncio.FIRST_COMPLETED)
                for t in done:
                    exc = t.exception()
                    if exc is not None and fatal_error_event.is_set():
                        break
                    if not scheduling_done.is_set():
                        await start_next()

            if fatal_error_event.is_set():
                for t in list(inflight):
                    if not t.done():
                        t.cancel()
                if inflight:
                    try:
                        await asyncio.wait_for(
                            asyncio.gather(*inflight, return_exceptions=True),
                            timeout=10.0,
                        )
                    except asyncio.TimeoutError:
                        pass
                raise RuntimeError("API appears to be down")

            if inflight:
                await asyncio.gather(*inflight, return_exceptions=True)
        finally:
            prefetch_task.cancel()
            await client.close()

    logger.info(f"Worker started (PID {os.getpid()}, {effective_connections} connections)")
    try:
        asyncio.run(_run())
    except Exception as e:
        logger.error(f"Worker failed (PID {os.getpid()}): {e}")
        sys.exit(1)
    finally:
        write_queue.put(None)
    logger.info(f"Worker finished (PID {os.getpid()})")


def _writer_main(write_queue, config_path, shard, num_workers, processed_count):
    """Writer subprocess: drains write_queue into DatasetWriter, increments shared counter."""
    config = load_inference_config(config_path)
    schema = (
        CHAT_COMPLETION_SCHEMA
        if config.api_type == "chat-completion"
        else COMPLETION_SCHEMA
    )
    writer = DatasetWriter(
        dataset_dir=config.output_path, schema=schema, shard=shard, batch_size=10_000,
    )
    _setup_signal_handlers(writer)

    workers_done = 0
    logger.info(f"Writer started (PID {os.getpid()})")
    try:
        while workers_done < num_workers:
            item = write_queue.get()
            if item is None:
                workers_done += 1
                logger.debug(f"Writer: worker sentinel received ({workers_done}/{num_workers})")
                continue
            writer.add_row(item)
            with processed_count.get_lock():
                processed_count.value += 1

        logger.info(f"Writer closing: {processed_count.value:_} rows written")
        writer.close()
        logger.info("Writer closed successfully")
    except Exception as e:
        logger.error(f"Writer failed (PID {os.getpid()}): {e}")
        if not writer._closed:
            writer.close(emergency=True)
        sys.exit(1)


def main(config, args: argparse.Namespace):
    cli_workers = getattr(args, 'num_workers', None)
    num_workers = cli_workers if cli_workers is not None else config.num_workers

    if num_workers <= 1:
        try:
            asyncio.run(run_inference_async(config, args))
        except RuntimeError as e:
            if "API appears to be down" in str(e):
                logger.error(f"Exiting due to API failure: {e}")
                sys.exit(1)
            raise
        except Exception as e:
            logger.error(f"Unhandled exception in main: {e}")
            sys.exit(1)
        return

    # --- Multi-worker queue-based pipeline ---
    effective_connections = config.max_connections // num_workers
    logger.info(
        f"Starting queue-based pipeline: {num_workers} workers, "
        f"{effective_connections} connections/worker"
    )

    _load_and_validate_dataset(config, args)
    existing_ids = _read_existing_ids(config.output_path, args.shard)

    ds = load_data(config, args.shard, args.num_shards)
    total_rows = len(ds)

    read_queue = multiprocessing.Queue(maxsize=effective_connections * 2)
    read_queue.cancel_join_thread()
    write_queue = multiprocessing.Queue()
    processed_count = multiprocessing.Value('i', 0)
    error_count = multiprocessing.Value('i', 0)

    shared_token_counts = (
        multiprocessing.Value('l', 0),  # prompt tokens
        multiprocessing.Value('l', 0),  # completion tokens
        multiprocessing.Value('l', 0),  # total tokens
    )

    progress_logger = None
    if args.log_file:
        progress_logger = ProgressLogger(args.log_file, args.shard, args.num_shards)
        progress_logger.file_handle = open(progress_logger.log_file_path, 'w')

    writer_proc = multiprocessing.Process(
        target=_writer_main,
        args=(write_queue, args.config, args.shard, num_workers, processed_count),
    )
    writer_proc.start()

    worker_procs = []
    for _ in range(num_workers):
        p = multiprocessing.Process(
            target=_worker_main,
            args=(read_queue, write_queue, args.config, vars(args), effective_connections, shared_token_counts, error_count),
        )
        p.start()
        worker_procs.append(p)

    shutting_down = False

    def _shutdown_handler(signum, frame):
        nonlocal shutting_down
        if shutting_down:
            return
        shutting_down = True
        signal_name = signal.Signals(signum).name
        logger.info(f"Received {signal_name}, shutting down pipeline...")
        try:
            while not read_queue.empty():
                read_queue.get_nowait()
        except Exception:
            pass
        for _ in range(num_workers):
            try:
                read_queue.put(None, timeout=5)
            except Exception:
                pass
        for p in worker_procs:
            p.join(timeout=30)
            if p.is_alive():
                p.terminate()
        writer_proc.join(timeout=30)
        if writer_proc.is_alive():
            writer_proc.terminate()
        logger.info(f"Shutdown complete after {signal_name}")
        os._exit(1)

    signal.signal(signal.SIGTERM, _shutdown_handler)
    signal.signal(signal.SIGINT, _shutdown_handler)

    skipped = 0
    fed = 0
    start_time = time.time()
    last_report_time = start_time
    last_report_processed = 0
    last_report_prompt_tokens = 0
    last_report_completion_tokens = 0
    last_report_total_tokens = 0
    all_workers_dead = False

    for row in ds:
        if shutting_down or all_workers_dead:
            break
        row_id = row[config.id_column_name]
        if row_id in existing_ids:
            skipped += 1
            continue

        while not shutting_down and not all_workers_dead:
            try:
                read_queue.put(row, timeout=30)
                break
            except Exception:
                if not any(p.is_alive() for p in worker_procs):
                    logger.error("All workers died, aborting reader")
                    all_workers_dead = True
        fed += 1

        now = time.time()
        if now - last_report_time >= args.progress_report_interval:
            done_count = processed_count.value
            completed = done_count + skipped
            elapsed = now - start_time
            rate = done_count / elapsed if elapsed > 0 else 0
            interval_duration = now - last_report_time
            interval_new = done_count - last_report_processed
            interval_rate = interval_new / interval_duration if interval_duration > 0 else 0
            remaining = total_rows - completed
            eta = remaining / interval_rate if interval_rate > 0 else 0
            logger.info(
                f"Progress: {completed:_}/{total_rows:_} completed "
                f"({done_count:_} new, {skipped:_} existing), "
                f"Overall: {rate:.1f} reqs/s ({rate * 3600:_.0f} reqs/h), "
                f"Last {args.progress_report_interval}s: {interval_rate:.1f} reqs/s ({interval_rate * 3600:_.0f} reqs/h), "
                f"ETA: {format_time(eta)}"
            )

            if progress_logger:
                cur_pt = shared_token_counts[0].value
                cur_ct = shared_token_counts[1].value
                cur_tt = shared_token_counts[2].value
                progress_logger.total_prompt_tokens = cur_pt
                progress_logger.total_completion_tokens = cur_ct
                progress_logger.total_tokens = cur_tt
                progress_logger.interval_prompt_tokens = cur_pt - last_report_prompt_tokens
                progress_logger.interval_completion_tokens = cur_ct - last_report_completion_tokens
                progress_logger.interval_tokens = cur_tt - last_report_total_tokens
                progress_logger.log_progress(
                    completed=completed,
                    total=total_rows,
                    new=done_count,
                    existing=skipped,
                    eta_seconds=eta,
                    eta_formatted=format_time(eta),
                    overall_rate=rate,
                    interval_rate=interval_rate,
                    interval_duration=interval_duration,
                    interval_new=interval_new,
                    interval_existing=0,
                    start_time=start_time,
                )
                last_report_prompt_tokens = cur_pt
                last_report_completion_tokens = cur_ct
                last_report_total_tokens = cur_tt

            last_report_time = now
            last_report_processed = done_count

    logger.info(f"Reader finished: {fed:_} rows sent, {skipped:_} existing skipped")

    for _ in range(num_workers):
        try:
            read_queue.put(None, timeout=5)
        except Exception:
            pass

    for p in worker_procs:
        p.join()

    writer_proc.join()

    elapsed = time.time() - start_time
    total_processed = processed_count.value
    total_errors = error_count.value
    rate = total_processed / elapsed if elapsed > 0 else 0
    logger.info(
        f"Done. {total_processed:_} processed, {skipped:_} skipped, {total_errors:_} errors, "
        f"Total time: {format_time(elapsed)}, "
        f"Overall: {rate:.1f} reqs/s ({rate * 3600:_.0f} reqs/h)"
    )

    if progress_logger:
        completed = total_processed + skipped
        cur_pt = shared_token_counts[0].value
        cur_ct = shared_token_counts[1].value
        cur_tt = shared_token_counts[2].value
        progress_logger.total_prompt_tokens = cur_pt
        progress_logger.total_completion_tokens = cur_ct
        progress_logger.total_tokens = cur_tt
        progress_logger.interval_prompt_tokens = cur_pt - last_report_prompt_tokens
        progress_logger.interval_completion_tokens = cur_ct - last_report_completion_tokens
        progress_logger.interval_tokens = cur_tt - last_report_total_tokens
        final_interval = time.time() - last_report_time
        final_interval_new = total_processed - last_report_processed
        final_interval_rate = final_interval_new / final_interval if final_interval > 0 else 0
        progress_logger.log_progress(
            completed=completed,
            total=total_rows,
            new=total_processed,
            existing=skipped,
            eta_seconds=0,
            eta_formatted="0s",
            overall_rate=rate,
            interval_rate=final_interval_rate,
            interval_duration=final_interval,
            interval_new=final_interval_new,
            interval_existing=0,
            start_time=start_time,
        )
        progress_logger.file_handle.close()

    failed = any(p.exitcode != 0 for p in worker_procs) or writer_proc.exitcode != 0
    if failed:
        logger.error("One or more processes failed")
        sys.exit(1)

    if (total_processed + skipped + total_errors) < total_rows:
        logger.error(
            f"Shard incomplete: {total_processed + skipped + total_errors:_}/{total_rows:_} rows"
        )
        sys.exit(1)


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
    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="Number of parallel worker processes for API calls. "
             "Overrides the config value. Uses a queue-based pipeline when > 1.",
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
