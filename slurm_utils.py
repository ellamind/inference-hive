import json
import subprocess
from collections import Counter

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
    jobs = [job for job in jobs if job["job_state"] in {"PENDING", "RUNNING", "CONFIGURING", "COMPLETING"}]
    return jobs


def get_job_state_counts(jobs: list[dict]) -> Counter:
    return Counter(job["job_state"][0] for job in jobs)