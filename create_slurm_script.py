import argparse
import shutil
from pathlib import Path

from loguru import logger

from config import load_job_config


SBATCH_TEMPLATE = """#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --partition={partition}
#SBATCH --account={account}
#SBATCH --qos={qos}
##SBATCH --array=1-{num_inference_servers}
#SBATCH --nodes={num_nodes_per_inference_server}
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task={cpus_per_node}
#SBATCH --mem={memory_per_node}
#SBATCH --gres={gres_per_node}
#SBATCH --time={time_limit}
#SBATCH --signal=B:SIGUSR1@120
#SBATCH --output={log_dir}/%a-%A-%N.log
#SBATCH --error={log_dir}/%a-%A-%N.log
{additional_sbatch_lines}

# Print job information
echo "=== SLURM Job Information ==="
echo "SLURM_JOB_NAME: ${{SLURM_JOB_NAME}}"
echo "SLURM_JOB_ID: ${{SLURM_JOB_ID}}"
echo "SLURM_ARRAY_JOB_ID: ${{SLURM_ARRAY_JOB_ID}}"
echo "SLURM_ARRAY_TASK_ID: ${{SLURM_ARRAY_TASK_ID}}"
echo "SLURM_JOB_NUM_NODES: ${{SLURM_JOB_NUM_NODES}}"
echo "SLURM_JOB_NODELIST: ${{SLURM_JOB_NODELIST}}"
echo "SLURM_JOB_PARTITION: ${{SLURM_JOB_PARTITION}}"
echo "SLURM_JOB_ACCOUNT: ${{SLURM_JOB_ACCOUNT}}"
echo "============================="

# Check if this shard is already completed
CURRENT_SHARD=$((SLURM_ARRAY_TASK_ID - 1))
COMPLETED_SHARDS_FILE="{log_dir}/shards_completed.log"
FAILED_SHARDS_FILE="{log_dir}/shards_failed.log"


set -e

log() {{
   local level="$1"
   local message="$2"
   echo "$(date '+%Y-%m-%d %H:%M:%S') [$level] $message"
}}

log_failed_shard() {{
    local reason="$1"
    local additional_info="$2"
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    local job_info="${{SLURM_ARRAY_JOB_ID:-${{SLURM_JOB_ID}}}}-${{SLURM_ARRAY_TASK_ID}}"
    
    # Create failed shards log entry: shard_number timestamp reason job_info hostname additional_info
    echo "${{CURRENT_SHARD}} ${{timestamp}} ${{reason}} ${{job_info}} ${{additional_info}}" >> "${{FAILED_SHARDS_FILE}}"
    log "ERROR" "Marked shard ${{CURRENT_SHARD}} as failed: ${{reason}} ${{additional_info}}"
}}

# Check if this shard is already completed
if [ -f "$COMPLETED_SHARDS_FILE" ]; then
    # Use grep with word boundaries to match exact shard number
    if grep -q "^${{CURRENT_SHARD}}$" "$COMPLETED_SHARDS_FILE"; then
        log "INFO" "Shard ${{CURRENT_SHARD}} is already completed. Exiting."
        exit 0
    else
        log "INFO" "Shard ${{CURRENT_SHARD}} not found in completed shards. Proceeding with inference."
    fi
else
    log "INFO" "No completed shards log found. Proceeding with inference."
fi

cleanup() {{
    local signal=$1
    if [ "$signal" = "SIGUSR1" ]; then
        log "WARN" "Job is about to hit time limit, shutting down gracefully..."
    elif [ "$signal" = "SIGTERM" ]; then
        log "WARN" "Job received cancellation signal, shutting down gracefully..."
    else
        log "WARN" "Job received signal $signal, shutting down gracefully..."
    fi
    log "INFO" "Initiating graceful shutdown of processes..."
    
    # Send SIGINT to inference process group
    if [ ! -z "$INFERENCE_PID" ] && kill -0 $INFERENCE_PID 2>/dev/null; then
        log "INFO" "Sending SIGINT to inference process group (PID: $INFERENCE_PID)"
        # Send signal to the entire process group using negative PID
        kill -INT -$INFERENCE_PID 2>/dev/null || kill -INT $INFERENCE_PID
    fi
    
    # give the inference script some time before shutting down the inference server
    sleep 10 

    # Send SIGINT to inference server process group
    if [ ! -z "$INFERENCE_SERVER_PID" ] && kill -0 $INFERENCE_SERVER_PID 2>/dev/null; then
        log "INFO" "Sending SIGINT to inference server process group (PID: $INFERENCE_SERVER_PID)"
        # Send signal to the entire process group using negative PID
        kill -INT -$INFERENCE_SERVER_PID 2>/dev/null || kill -INT $INFERENCE_SERVER_PID
    fi

    log "INFO" "Waiting for processes to finish gracefully..."

    # Wait for processes to finish with timeout
    local wait_timeout=60
    local wait_count=0
    
    while [ $wait_count -lt $wait_timeout ]; do
        local processes_running=0
        
        # Check if processes are still running
        if [ ! -z "$INFERENCE_SERVER_PID" ] && kill -0 $INFERENCE_SERVER_PID 2>/dev/null; then
            processes_running=$((processes_running + 1))
        fi
        if [ ! -z "$HEALTHCHECK_PID" ] && kill -0 $HEALTHCHECK_PID 2>/dev/null; then
            processes_running=$((processes_running + 1))
        fi
        if [ ! -z "$INFERENCE_PID" ] && kill -0 $INFERENCE_PID 2>/dev/null; then
            processes_running=$((processes_running + 1))
        fi
        
        if [ $processes_running -eq 0 ]; then
            log "INFO" "All processes have finished gracefully"
            break
        fi
        
        log "INFO" "Waiting for $processes_running processes to finish... ($wait_count/$wait_timeout)"
        sleep 1
        wait_count=$((wait_count + 1))
    done
    
    # Force kill any remaining processes
    if [ $wait_count -eq $wait_timeout ]; then
        log "WARN" "Timeout reached, force killing remaining processes"
        if [ ! -z "$INFERENCE_SERVER_PID" ] && kill -0 $INFERENCE_SERVER_PID 2>/dev/null; then
            # Force kill the entire process group
            kill -KILL -$INFERENCE_SERVER_PID 2>/dev/null || kill -KILL $INFERENCE_SERVER_PID 2>/dev/null
        fi
        if [ ! -z "$INFERENCE_PID" ] && kill -0 $INFERENCE_PID 2>/dev/null; then
            # Force kill the entire process group
            kill -KILL -$INFERENCE_PID 2>/dev/null || kill -KILL $INFERENCE_PID 2>/dev/null
        fi
    fi
    
    # Only resubmit if this was a time limit signal, not a manual cancellation
    # Resubmit instead of requeue since requeue is disabled on many clusters.
    if [ "$signal" = "SIGUSR1" ]; then
        log "INFO" "Resubmitting task ${{SLURM_ARRAY_TASK_ID}} automatically due to time limit..."
        sbatch_output=$(sbatch --array=${{SLURM_ARRAY_TASK_ID}} "{log_dir}/{job_name}.slurm" 2>&1)
        if [ $? -eq 0 ]; then
            new_job_id=$(echo "$sbatch_output" | grep -o '[0-9]*')
            log "INFO" "Task ${{SLURM_ARRAY_TASK_ID}} resubmitted successfully as job ${{new_job_id}}. Progress will resume from where it left off."
        else
            log "ERROR" "Failed to resubmit task ${{SLURM_ARRAY_TASK_ID}}. Error: $sbatch_output"
            log "ERROR" "You may need to resubmit manually: sbatch --array=${{SLURM_ARRAY_TASK_ID}} {log_dir}/{job_name}.slurm"
            # Mark as failed since resubmission failed
            log_failed_shard "resubmission_failed" "Failed to resubmit after time limit: $sbatch_output"
        fi
    else
        # Mark as failed for any signal other than SIGUSR1 (time limit)
        if [ "$signal" = "SIGTERM" ]; then
            log_failed_shard "manual_cancellation" "Job was manually cancelled"
        else
            log_failed_shard "unexpected_signal" "Job received signal: $signal"
        fi
        log "INFO" "Task ${{SLURM_ARRAY_TASK_ID}} was manually cancelled - not resubmitting automatically."
        log "INFO" "To restart this task later, run: sbatch --array=${{SLURM_ARRAY_TASK_ID}} {log_dir}/{job_name}.slurm"
    fi
    
    exit 0
}}

trap 'cleanup SIGUSR1' SIGUSR1 # send before timelimit is hit
trap 'cleanup SIGTERM' SIGTERM # send by scancel

# Setup env

# Activate pixi environment
log "INFO" "Activating pixi environment: {pixi_env}"
eval "$(pixi shell-hook --manifest-path {pixi_manifest} -e {pixi_env} --no-install)"
log "INFO" "python path: $(which python)"

# Add env exports
{env_exports}
export API_BASE_URL="{api_base_url}"
MASTER_NODE=$(scontrol show hostname ${{SLURM_JOB_NODELIST}} | head -n1)
export MASTER_NODE

# Validate dataset before starting inference server
log "INFO" "Validating dataset format for shard ${{CURRENT_SHARD}}"
python validate_data.py --config {config_path} --shard ${{CURRENT_SHARD}} --num-shards {num_data_shards}
VALIDATION_EXIT_CODE=$?

if [ $VALIDATION_EXIT_CODE -ne 0 ]; then
    log "ERROR" "Dataset validation failed for shard ${{CURRENT_SHARD}}. Exiting."
    log_failed_shard "data_validation_failed" "Dataset validation failed before starting inference server"
    exit 1
fi

log "INFO" "Dataset validation passed for shard ${{CURRENT_SHARD}}"

# Start inference server
log "INFO" "Starting inference server on ${{SLURM_JOB_NUM_NODES}} nodes"
INFERENCE_SERVER_LOG="{log_dir}/${{SLURM_ARRAY_TASK_ID}}-${{SLURM_JOB_ID}}-%N-inference-server.log"
INFERENCE_SERVER_COMMAND="{inference_server_command}"
log "INFO" "Inference server command: ${{INFERENCE_SERVER_COMMAND}}"
setsid srun --output="$INFERENCE_SERVER_LOG" --error="$INFERENCE_SERVER_LOG" \\
    bash -c "${{INFERENCE_SERVER_COMMAND}}" &
INFERENCE_SERVER_PID=$!

# Give the inference server srun command time to actually start
sleep 5

# Check if the srun command is still running (not the inference server itself, but the srun process)
if ! kill -0 $INFERENCE_SERVER_PID 2>/dev/null; then
    # The srun command itself has exited, check its exit code
    wait $INFERENCE_SERVER_PID
    INFERENCE_SERVER_EXIT_CODE=$?
    if [ $INFERENCE_SERVER_EXIT_CODE -ne 0 ]; then
        log "ERROR" "Inference server srun command failed with exit code $INFERENCE_SERVER_EXIT_CODE. Check the inference server logs for details."
        log "ERROR" "This usually means the inference server failed to start. Exiting to prevent resource waste."
        log_failed_shard "inference_server_startup_failed" "srun command failed with exit code $INFERENCE_SERVER_EXIT_CODE"
        exit 1
    fi
fi

# Wait for inference server to become healthy
wait_for_api_server() {{
    local max_attempts=$(({health_check_max_wait_minutes} * 60 / {health_check_interval_seconds}))  # Calculate attempts based on total time and interval
    log "INFO" "Maximum health check wait time: {health_check_max_wait_minutes} minutes"
    local attempt=0
    local health_url="${{API_BASE_URL%/v1}}/health"
    
    while [ $attempt -lt $max_attempts ]; do
        log "INFO" "Health check attempt $((attempt + 1))/${{max_attempts}} for ${{health_url}}"
        
        # Check if the inference server srun process is still running
        if ! kill -0 $INFERENCE_SERVER_PID 2>/dev/null; then
            log "ERROR" "Inference server srun process (PID: $INFERENCE_SERVER_PID) has terminated during health checks"
            # Wait for the process to get its exit code
            wait $INFERENCE_SERVER_PID 2>/dev/null
            INFERENCE_SERVER_EXIT_CODE=$?
            log "ERROR" "Inference server srun command exited with code $INFERENCE_SERVER_EXIT_CODE"
            log_failed_shard "inference_server_terminated_during_healthcheck" "srun process died during health checks with exit code $INFERENCE_SERVER_EXIT_CODE"
            return 1
        fi
        
        if curl -s --connect-timeout 5 --max-time 10 "${{health_url}}" >/dev/null 2>&1; then
            log "INFO" "Inference server is healthy and ready!"
            return 0
        fi
        
        log "INFO" "Inference server not ready, waiting {health_check_interval_seconds} seconds..."
        sleep {health_check_interval_seconds}
        attempt=$((attempt + 1))
    done
    
    log "ERROR" "Error: inference server did not become healthy within timeout"
    return 1
}}
log "INFO" "Waiting for inference server to become healthy"
wait_for_api_server
HEALTHCHECK_EXIT_CODE=$?

# If health check failed, exit immediately
if [ $HEALTHCHECK_EXIT_CODE -ne 0 ]; then
    log "ERROR" "Health check failed. Exiting job to prevent resource waste."
    log_failed_shard "health_check_failed" "Inference server did not become healthy within timeout"
    exit 1
fi

# Run inference
log "INFO" "Running inference"
INFERENCE_PROGRESS_LOG="{progress_dir}/${{SLURM_ARRAY_TASK_ID}}-progress.jsonl"
# Start inference in a new process group so we can kill the entire group
setsid python run_inference.py --config {config_path} --num-shards {num_data_shards} --shard ${{CURRENT_SHARD}} --log-file $INFERENCE_PROGRESS_LOG &
INFERENCE_PID=$!
wait $INFERENCE_PID
INFERENCE_EXIT_CODE=$?

if [ $INFERENCE_EXIT_CODE -eq 0 ]; then
    log "INFO" "Inference completed successfully, recording shard completion"
    echo "$((SLURM_ARRAY_TASK_ID - 1))" >> "{log_dir}/completed_shards.log"
    log "INFO" "Done"
else
    log "ERROR" "Inference failed with exit code $INFERENCE_EXIT_CODE"
    log_failed_shard "inference_script_failed" "run_inference.py exited with code $INFERENCE_EXIT_CODE"
fi
"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", "-c",
        type=str,
        required=True,
        help="Path to the YAML configuration file"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        required=True,
        help="Directory where the slurm job script and config will be copied to, and slurm logs will be written to."
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Allow overwriting existing output directory"
    )
    args = parser.parse_args()
    
    # Validate that config file exists
    config_path = Path(args.config)
    if not config_path.exists():
        logger.error(f"Configuration file '{args.config}' does not exist.")
        return 1
    
    # Load configuration using the shared config module
    try:
        config = load_job_config(config_path)
        # Convert to dict for template formatting
        config_dict = config.model_dump()
    except Exception as e:
        logger.error(f"Configuration validation error: {e}")
        return 1

    # Ensure the output_path directory exists (where results will be written to)
    output_path = Path(config.output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    # Create output directory
    output_dir = Path(args.output)
    if output_dir.exists():
        if not args.force:
            logger.error(f"Output directory '{output_dir}' already exists. Use --force to overwrite.")
            return 1
        else:
            logger.warning(f"Output directory '{output_dir}' already exists. Overwriting due to --force flag.")
    
    # Ensure directory exists (in case --force was used but directory was removed)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {output_dir}")

    progress_dir = output_dir / "progress"
    progress_dir.mkdir(exist_ok=True)

    # Use output directory as log directory
    config_dict["log_dir"] = str(output_dir)
    config_dict["progress_dir"] = str(progress_dir)

    # num_inference_servers is the number of data shards
    config_dict["num_data_shards"] = config.num_inference_servers

    # Generate additional SBATCH lines
    additional_sbatch_lines = ""
    if config.additional_sbatch_args:
        for key, value in config.additional_sbatch_args.items():
            # Ensure key starts with -- if not already present
            if not key.startswith("--"):
                key = f"--{key}"
            additional_sbatch_lines += f"#SBATCH {key}={value}\n"
    config_dict["additional_sbatch_lines"] = additional_sbatch_lines

    # Generate environment variable exports
    env_exports = ""
    if config.env_vars:
        for key, value in config.env_vars.items():
            env_exports += f"export {key}=\"{value}\"\n"
    config_dict["env_exports"] = env_exports

    # Copy config file to output directory for reproducibility
    config_copy_path = output_dir / "ih_config.yaml"
    try:
        shutil.copy2(config_path, config_copy_path)
        logger.info(f"Config copied to: {config_copy_path}")
    except Exception as e:
        logger.warning(f"Could not copy config file: {e}")
    config_dict["config_path"] = str(config_copy_path)

    # Copy udf.py file to output directory for reproducibility if apply_udf is used
    if config.apply_udf:
        udf_path = Path(__file__).parent / "udf.py"
        udf_copy_path = output_dir / "udf.py"
        try:
            shutil.copy2(udf_path, udf_copy_path)
            logger.info(f"UDF file copied to: {udf_copy_path}")
        except Exception as e:
            logger.warning(f"Could not copy UDF file: {e}")

    sbatch_script = SBATCH_TEMPLATE.format(**config_dict)
    
    job_script_path = output_dir / "ih_job.slurm"
    try:
        with open(job_script_path, "w") as f:
            f.write(sbatch_script)
        logger.info(f"SLURM job script generated successfully: {job_script_path}")
    except Exception as e:
        logger.error(f"Error writing job script '{job_script_path}': {e}")
        return 1

    logger.info(f"To submit the job: sbatch {job_script_path}")
    logger.info(f"To cancel all jobs: scancel --name={config_dict['job_name']}")
    logger.info(f"To check job status: squeue -u $USER --name={config_dict['job_name']}")
    
    return 0


if __name__ == "__main__":
    exit(main())
