#!/bin/bash
#SBATCH --job-name=fw2-annotations-deu_Latn-1M
#SBATCH --partition=boost_usr_prod
#SBATCH --account=AIFAC_L01_028
#SBATCH --qos=normal
#SBATCH --array=1-8%8
#SBATCH --nodes=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=0
#SBATCH --gres=gpu:4
#SBATCH --time=24:00:00
#SBATCH --signal=B:SIGUSR1@120
#SBATCH --output=fw2_annotations_run1/%A-%a-%N.log
#SBATCH --error=fw2_annotations_run1/%A-%a-%N.log
#SBATCH --exclude=lrdn[0032,0701]


# Print job information
echo "=== SLURM Job Information ==="
echo "SLURM_JOB_NAME: ${SLURM_JOB_NAME}"
echo "SLURM_JOB_ID: ${SLURM_JOB_ID}"
echo "SLURM_ARRAY_JOB_ID: ${SLURM_ARRAY_JOB_ID}"
echo "SLURM_ARRAY_TASK_ID: ${SLURM_ARRAY_TASK_ID}"
echo "SLURM_JOB_NUM_NODES: ${SLURM_JOB_NUM_NODES}"
echo "SLURM_JOB_NODELIST: ${SLURM_JOB_NODELIST}"
echo "SLURM_JOB_PARTITION: ${SLURM_JOB_PARTITION}"
echo "SLURM_JOB_ACCOUNT: ${SLURM_JOB_ACCOUNT}"
echo "============================="

# Check if this shard is already completed
CURRENT_SHARD=$((SLURM_ARRAY_TASK_ID - 1))
COMPLETED_SHARDS_FILE="fw2_annotations_run1/shards_completed.log"
FAILED_SHARDS_FILE="fw2_annotations_run1/shards_failed.log"

if [ -f "$COMPLETED_SHARDS_FILE" ]; then
    # Use grep with word boundaries to match exact shard number
    if grep -q "^${CURRENT_SHARD}$" "$COMPLETED_SHARDS_FILE"; then
        echo "$(date '+%Y-%m-%d %H:%M:%S') [INFO] Shard ${CURRENT_SHARD} is already completed. Exiting."
        exit 0
    else
        echo "$(date '+%Y-%m-%d %H:%M:%S') [INFO] Shard ${CURRENT_SHARD} not found in completed shards. Proceeding with inference."
    fi
else
    echo "$(date '+%Y-%m-%d %H:%M:%S') [INFO] No completed shards log found. Proceeding with inference."
fi

set -e

log() {
   local level="$1"
   local message="$2"
   echo "$(date '+%Y-%m-%d %H:%M:%S') [$level] $message"
}

log_failed_shard() {
    local reason="$1"
    local additional_info="$2"
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    local job_info="${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID}}-${SLURM_ARRAY_TASK_ID}"
    
    # Create failed shards log entry: shard_number timestamp reason job_info hostname additional_info
    echo "${CURRENT_SHARD} ${timestamp} ${reason} ${job_info} ${additional_info}" >> "${FAILED_SHARDS_FILE}"
    log "ERROR" "Marked shard ${CURRENT_SHARD} as failed: ${reason} ${additional_info}"
}

cleanup() {
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
        log "INFO" "Resubmitting task ${SLURM_ARRAY_TASK_ID} automatically due to time limit..."
        sbatch_output=$(sbatch --array=${SLURM_ARRAY_TASK_ID} "fw2_annotations_run1/fw2-annotations-deu_Latn-1M.slurm" 2>&1)
        if [ $? -eq 0 ]; then
            new_job_id=$(echo "$sbatch_output" | grep -o '[0-9]*')
            log "INFO" "Task ${SLURM_ARRAY_TASK_ID} resubmitted successfully as job ${new_job_id}. Progress will resume from where it left off."
        else
            log "ERROR" "Failed to resubmit task ${SLURM_ARRAY_TASK_ID}. Error: $sbatch_output"
            log "ERROR" "You may need to resubmit manually: sbatch --array=${SLURM_ARRAY_TASK_ID} fw2_annotations_run1/fw2-annotations-deu_Latn-1M.slurm"
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
        log "INFO" "Task ${SLURM_ARRAY_TASK_ID} was manually cancelled - not resubmitting automatically."
        log "INFO" "To restart this task later, run: sbatch --array=${SLURM_ARRAY_TASK_ID} fw2_annotations_run1/fw2-annotations-deu_Latn-1M.slurm"
    fi
    
    exit 0
}

trap 'cleanup SIGUSR1' SIGUSR1 # send before timelimit is hit
trap 'cleanup SIGTERM' SIGTERM # send by scancel

# Setup env
export HF_HUB_OFFLINE="1"
export VLLM_DISABLE_COMPILE_CACHE="1"

export API_BASE_URL="http://localhost:64776/v1"
MASTER_NODE=$(scontrol show hostname ${SLURM_JOB_NODELIST} | head -n1)
export MASTER_NODE
log "INFO" "python path: $(pixi run --manifest-path pixi.toml -e cuda-vllm --no-install which python)"

# Validate dataset before starting inference server
log "INFO" "Validating dataset format for shard ${CURRENT_SHARD}"
pixi run --manifest-path pixi.toml -e cuda-vllm --no-install \
    python validate_data.py --config fw2_annotations_run1/config_fw2_annotations.yaml --shard ${CURRENT_SHARD} --num-shards 8
VALIDATION_EXIT_CODE=$?

if [ $VALIDATION_EXIT_CODE -ne 0 ]; then
    log "ERROR" "Dataset validation failed for shard ${CURRENT_SHARD}. Exiting."
    log_failed_shard "data_validation_failed" "Dataset validation failed before starting inference server"
    exit 1
fi

log "INFO" "Dataset validation passed for shard ${CURRENT_SHARD}"

# Start inference server
log "INFO" "Starting inference server on ${SLURM_JOB_NUM_NODES} nodes"
INFERENCE_SERVER_LOG="fw2_annotations_run1/${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID}}-${SLURM_ARRAY_TASK_ID}-${HOSTNAME}-inference-server.log"
setsid pixi run --manifest-path pixi.toml -e cuda-vllm --no-install \
    python -m vllm.entrypoints.openai.api_server --host=localhost --port=64776 --model="Qwen/Qwen3-4B" --tensor-parallel-size=1 --data-parallel-size=4 --max-model-len=16384 --trust-remote-code --gpu-memory-utilization=0.9 --disable-log-requests \
    > "${INFERENCE_SERVER_LOG}" 2>&1 &
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
wait_for_api_server() {
    local max_attempts=$((10 * 60 / 20))  # Calculate attempts based on total time and interval
    local attempt=0
    local health_url="${API_BASE_URL%/v1}/health"
    
    while [ $attempt -lt $max_attempts ]; do
        log "INFO" "Health check attempt $((attempt + 1))/${max_attempts} for ${health_url}"
        
        if curl -s --connect-timeout 5 --max-time 10 "${health_url}" >/dev/null 2>&1; then
            log "INFO" "Inference server is healthy and ready!"
            return 0
        fi
        
        log "INFO" "Inference server not ready, waiting 20 seconds..."
        sleep 20
        attempt=$((attempt + 1))
    done
    
    log "ERROR" "Error: inference server did not become healthy within timeout"
    return 1
}
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
INFERENCE_PROGRESS_LOG="fw2_annotations_run1/${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID}}-${SLURM_ARRAY_TASK_ID}-${HOSTNAME}-inference-stats.jsonl"
# Start inference in a new process group so we can kill the entire group
setsid pixi run --manifest-path pixi.toml -e cuda-vllm --no-install \
    python run_inference.py --config fw2_annotations_run1/config_fw2_annotations.yaml --num-shards 8 --shard $((SLURM_ARRAY_TASK_ID - 1)) --log-file $INFERENCE_PROGRESS_LOG &
INFERENCE_PID=$!
wait $INFERENCE_PID
INFERENCE_EXIT_CODE=$?

if [ $INFERENCE_EXIT_CODE -eq 0 ]; then
    log "INFO" "Inference completed successfully, recording shard completion"
    echo "$((SLURM_ARRAY_TASK_ID - 1))" >> "fw2_annotations_run1/completed_shards.log"
    log "INFO" "Done"
else
    log "ERROR" "Inference failed with exit code $INFERENCE_EXIT_CODE"
    log_failed_shard "inference_script_failed" "run_inference.py exited with code $INFERENCE_EXIT_CODE"
fi
