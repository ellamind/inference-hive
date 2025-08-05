import yaml
from pathlib import Path
from pydantic import BaseModel, Field, field_validator, model_validator
from typing import Optional, Dict, Any, Literal


class BaseConfig(BaseModel):
    """Base configuration with common fields used by both job creation and inference"""
    
    # API Configuration
    api_base_url: str = Field(..., description="Base URL for the API server")
    api_type: Literal["completion", "chat-completion"] = Field(
        ..., description="Type of API (completion or chat-completion)"
    )
    model: str = Field(..., description="Model identifier")

    # Dataset Configuration
    dataset_path: Path = Field(..., description="Path to the dataset")
    input_column_name: str = Field(
        default="prompt", description="Name of the input column"
    )
    id_column_name: str = Field(
        default="id", description="Name of the ID column. Must contain unique string identifiers for each row."
    )
    use_load_from_disk: bool = Field(
        ..., description="Whether to load dataset from disk"
    )

    # Output Configuration
    output_path: Path = Field(..., description="Path for output files")

    # Dataset Loading Arguments
    load_dataset_kwargs: Optional[Dict[str, Any]] = Field(
        default=None, description="Additional dataset loading arguments"
    )

    # Completion Arguments
    completions_kwargs: Optional[Dict[str, Any]] = Field(
        default=None, description="Arguments for completions"
    )

    # Connection Settings
    max_connections: int = Field(
        ..., gt=0, description="Maximum number of connections"
    )
    max_retries: int = Field(default=3, ge=0, description="Maximum number of retries")

    @field_validator("dataset_path", mode="before")
    @classmethod
    def convert_dataset_path_to_path(cls, v):
        return Path(v)

    @field_validator("output_path", mode="before")
    @classmethod
    def convert_output_path_to_path(cls, v):
        return Path(v).absolute()

    @field_validator("load_dataset_kwargs", "completions_kwargs", mode="before")
    @classmethod
    def convert_none_to_dict(cls, v):
        return v if v is not None else {}


class JobConfig(BaseConfig):
    """Configuration for SLURM job creation (extends BaseConfig)"""

    # SLURM Configuration
    job_name: str = Field(..., description="Name of the SLURM job")
    partition: str = Field(..., description="SLURM partition to use")
    account: str = Field(..., description="SLURM account to charge")
    qos: str = Field(..., description="Quality of Service for the job")
    num_inference_servers: int = Field(
        ..., gt=0, description="Number of inference servers to start"
    )
    num_nodes_per_inference_server: int = Field(
        ..., gt=0, description="Number of nodes per inference server"
    )
    num_data_shards: Optional[int] = Field(
        default=None, gt=0, description="Total number of data shards. If not specified, defaults to num_inference_servers. For very large datasets you can use more shards than inference servers for checkpointing. However, only num_inference_servers shards will be processed in parallel."
    )
    cpus_per_node: int = Field(..., gt=0, description="Number of CPUs per node")
    memory_per_node: str = Field(..., description="Memory per node in GB")
    gres_per_node: str = Field(..., description="Generic resources per node (e.g., 'gpu:4', 'nvgpu:2', 'a100:1')")
    time_limit: str = Field(..., description="Time limit for the job (HH:MM format)")
    additional_sbatch_args: Optional[Dict[str, str]] = Field(
        default=None, description="Additional SBATCH arguments as key-value pairs"
    )

    # Environment Configuration
    env_vars: Optional[Dict[str, str]] = Field(
        default=None, description="Environment variables as key-value pairs"
    )
    pixi_manifest: str = Field(
        ..., description="Path to the pixi manifest file"
    )
    pixi_env: str = Field(
        ..., description="Pixi environment to use"
    )

    # Inference Server Configuration
    inference_server_command: str = Field(
        ..., description="Command to start the inference server"
    )

    # Health Check Configuration
    health_check_max_wait_minutes: int = Field(
        default=10, gt=0, description="Maximum time to wait for server health"
    )
    health_check_interval_seconds: int = Field(
        default=20, gt=0, description="Interval between health checks"
    )

    @field_validator("time_limit")
    @classmethod
    def validate_time_limit(cls, v):
        """Validate time limit format according to SLURM documentation.
        Acceptable formats: "minutes", "minutes:seconds", "hours:minutes:seconds", 
        "days-hours", "days-hours:minutes", "days-hours:minutes:seconds".
        A time limit of zero requests that no time limit be imposed.
        """
        if not isinstance(v, str):
            raise ValueError("time_limit must be a string")

        # Special case: zero means no time limit
        if v == "0":
            return v

        # Check for days-hours formats (contains dash)
        if "-" in v:
            # Split on dash first
            dash_parts = v.split("-")
            if len(dash_parts) != 2:
                raise ValueError("time_limit with days format must have exactly one dash")
            
            days_part, hours_part = dash_parts
            
            # Validate days part is a number
            try:
                int(days_part)
            except ValueError:
                raise ValueError("days part must be a number")
            
            # hours_part can be "hours", "hours:minutes", or "hours:minutes:seconds"
            colon_parts = hours_part.split(":")
            if len(colon_parts) not in [1, 2, 3]:
                raise ValueError("hours part must be in format 'hours', 'hours:minutes', or 'hours:minutes:seconds'")
            
            try:
                for part in colon_parts:
                    int(part)
            except ValueError:
                raise ValueError("all time components must be numbers")
                
        else:
            # No dash, so it's minutes, minutes:seconds, or hours:minutes:seconds
            colon_parts = v.split(":")
            if len(colon_parts) not in [1, 2, 3]:
                raise ValueError("time_limit must be in format 'minutes', 'minutes:seconds', or 'hours:minutes:seconds'")
            
            try:
                for part in colon_parts:
                    int(part)
            except ValueError:
                raise ValueError("all time components must be numbers")

        return v

    @field_validator("inference_server_command")
    @classmethod
    def validate_inference_server_command(cls, v):
        """Strip whitespace and newlines from inference server command."""
        return v.strip()

    @field_validator("additional_sbatch_args")
    @classmethod
    def validate_additional_sbatch_args(cls, v):
        """Validate that additional_sbatch_args don't conflict with existing args."""
        if v is None:
            return v
        
        # List of SBATCH arguments that are already used and cannot be overridden
        reserved_args = {
            # Arguments explicitly used in the template
            "job-name", "partition", "account", "qos", "array", "nodes", 
            "cpus-per-task", "mem", "gres", "time", "signal", "output", "error",
            # Arguments that would conflict with script logic
            "dependency",   # Could interfere with array job management
            "requeue", "no-requeue",  # Conflicts with built-in requeue logic
            "wait",         # Changes submission behavior
            # GPU-related options (conflict with --gres)
            "gpus", "gpus-per-node", "gpus-per-socket", "gpus-per-task",
            "gpu-bind", "gpu-freq",
            # Memory-related options (conflict with --mem)
            "mem-per-cpu", "mem-per-gpu", "mem-bind",
            # CPU/task-related options (conflict with resource allocation)
            "cpus-per-gpu", "cores-per-socket", "sockets-per-node", "threads-per-core",
            "ntasks", "ntasks-per-node", "ntasks-per-core", "ntasks-per-socket", "ntasks-per-gpu",
            "hint", "extra-node-info",
            # TRES-related options
            "tres-bind", "tres-per-task",
        }
        
        for key in v.keys():
            # Remove leading dashes if present
            clean_key = key.lstrip("-")
            if clean_key in reserved_args:
                raise ValueError(f"SBATCH argument '{key}' is reserved and cannot be overridden. Reserved args: {sorted(reserved_args)}")
        
        return v

    @model_validator(mode="after")
    def validate_and_set_num_data_shards(self):
        """Validate and set num_data_shards with appropriate warnings."""
        from loguru import logger
        
        # Set default if not specified
        if self.num_data_shards is None:
            self.num_data_shards = self.num_inference_servers
            logger.info(f"num_data_shards not specified, defaulting to num_inference_servers ({self.num_inference_servers})")
        
        # Ensure num_data_shards is at least num_inference_servers
        elif self.num_data_shards < self.num_inference_servers:
            logger.warning(f"num_data_shards ({self.num_data_shards}) is smaller than num_inference_servers ({self.num_inference_servers}). "
                          f"Setting num_data_shards to {self.num_inference_servers} to avoid idle inference servers.")
            self.num_data_shards = self.num_inference_servers
        
        return self


class InferenceConfig(BaseConfig):
    """Configuration for distributed offline inference (extends BaseConfig)"""

    # Additional inference-specific fields only
    max_consecutive_failures: int = Field(
        default=20, description="Maximum consecutive API failures before terminating"
    )


def load_job_config(config_path: str | Path) -> JobConfig:
    """Load and validate job configuration from YAML file"""
    with open(config_path, "r") as f:
        config_data = yaml.safe_load(f)
    return JobConfig(**config_data)


def load_inference_config(config_path: str | Path) -> InferenceConfig:
    """Load and validate inference configuration from YAML file"""
    with open(config_path, "r") as f:
        config_data = yaml.safe_load(f)
    return InferenceConfig(**config_data)


def load_config_for_validation(config_path: str) -> Dict[str, Any]:
    """Load and extract only the fields needed for data validation"""
    with open(config_path, "r") as f:
        config_data = yaml.safe_load(f)
    
    # Extract only the fields we need for validation (no defaults, explicit config required)
    required_fields = ['api_type', 'dataset_path', 'use_load_from_disk']
    for field in required_fields:
        if field not in config_data:
            raise ValueError(f"Required field '{field}' missing from config file")
    
    return {
        'api_type': config_data['api_type'],
        'dataset_path': Path(config_data['dataset_path']),
        'input_column_name': config_data.get('input_column_name', 'prompt'),  # Keep reasonable defaults for column names
        'id_column_name': config_data.get('id_column_name', 'id'),
        'use_load_from_disk': config_data['use_load_from_disk'],
        'load_dataset_kwargs': config_data.get('load_dataset_kwargs') or {}
    } 