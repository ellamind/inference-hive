import argparse
import sys
from pathlib import Path

from loguru import logger

from config import load_job_config, load_inference_config


def main():
    parser = argparse.ArgumentParser(description="Validate configuration file")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML configuration file")
    args = parser.parse_args()
    
    config_path = Path(args.config)
    
    if not config_path.exists():
        logger.error(f"Configuration file '{config_path}' does not exist.")
        sys.exit(1)
    
    logger.info(f"Validating: {config_path}")
    
    # Test JobConfig validation
    job_valid = False
    try:
        load_job_config(config_path)
        logger.info("✓ Valid for create_slurm_script.py")
        job_valid = True
    except Exception as e:
        logger.error(f"✗ Invalid for create_slurm_script.py: {e}")
    
    # Test InferenceConfig validation  
    inference_valid = False
    try:
        load_inference_config(config_path)
        logger.info("✓ Valid for run_inference.py")
        inference_valid = True
    except Exception as e:
        logger.error(f"✗ Invalid for run_inference.py: {e}")
    
    if job_valid and inference_valid:
        logger.info("Configuration is valid!")
        sys.exit(0)
    else:
        logger.error("Configuration validation failed!")
        sys.exit(1)


if __name__ == "__main__":
    main() 