import json
import subprocess
from collections import Counter

ACTIVE_JOB_STATES = {"PENDING", "RUNNING", "CONFIGURING", "COMPLETING"}

def _normalize_job_state(job_state):
    """Normalize job_state to a list, handling both string and list formats across Slurm versions."""
    if isinstance(job_state, list):
        return job_state
    return [job_state]

def _is_job_active(job):
    """Check if a job is in an active state."""
    states = _normalize_job_state(job["job_state"])
    return any(state in ACTIVE_JOB_STATES for state in states)

def get_current_jobs(me: bool=True, job_name: str=None):
    cmd = ["squeue", "--json"]
    if me:
        cmd.append("--me")
    try:
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"squeue failed: {e.stderr or e.stdout}") from e

    try:
        payload = json.loads(result.stdout or "{}")
    except json.JSONDecodeError as e:
        raise RuntimeError("Failed to parse squeue JSON output") from e

    jobs = payload.get("jobs")
    if not isinstance(jobs, list):
        raise RuntimeError("Unexpected squeue JSON structure: 'jobs' is not a list")
    if job_name:
        jobs = [job for job in jobs if job["name"][:-7] == job_name]
    jobs = [job for job in jobs if _is_job_active(job)]
    return jobs


def get_job_state_counts(jobs: list[dict]) -> Counter:
    return Counter(_normalize_job_state(job["job_state"])[0] for job in jobs)