<div align="center">

<img src="res/logo.png" alt="inference-hive Logo" width="128" height="128" style="vertical-align: middle;">

# inference-hive

*Run offline LLM inference at scale using SLURM*

</div>

**inference-hive** is a toolkit to run distributed LLM inference on SLURM clusters. Configure a few cluster, inference server and data settings, and scale your inference workload across thousands of GPUs.

---

*Disclaimer: This is a research project and not an official ellamind product.*

---

## Features
- **⚡ Distributed Inference**: Runs inference at scale by deploying multiple inference servers.
- **🚀 Linear Scaling**: Throughput scales linearly with number of nodes.
- **🔧 SLURM Native**: Automatic SLURM job script generation and resource management.
- **🔄 OpenAI API Compatible**: Supports both completion and chat-completion API formats.
- **📊 Monitoring**: Built-in progress tracking, throughput stats visualization.
- **🛡️ Robust Error Handling**: Health checks, automatic retries, failure tracking, graceful shutdown, resumption and job requeueing.
- **📁 Easy Data Loading**: Works with any HuggingFace datasets that contains prompts or conversations.
- **✅ Data Validation**: Pre-flight validation to catch data configuration issues.
- **⚙️ Highly Configurable**: YAML-based configuration for inference parameters and cluster setup.


## Environment
We use [Pixi](https://pixi.sh/dev/) for environment management. Refer to [Pixi Installation](https://pixi.sh/dev/installation/) for the installation guide.

We provide multiple environment definitions for different GPU types and different inference servers.
|Env Name|Description|Tested on|
|---|---|---|
|cuda-vllm|CUDA + vLLM|Leonardo|
|cuda-sglang|CUDA + SGLang||
|TBD|CUDA + vLLM on aarch64 (GH200)||
|TBD|CUDA + SGLang on aarch64 (GH200)||
|TBD|rocm + vLLM||
|TBD|rocm + SGLang||
|cpu|no inference server, debug env|-|

### Pre-Installed Environments
We have pre-installed environments on multiple clusters which you can re-use if you are part of the same project. This allows you to skip environment installation, saves storage and inodes.
|Cluster|Project|Env Name|Manifest|
|---|---|---|---|
|Leonardo Booster|MultiSynt (AIFAC_L01_028)|cuda-vllm|`/leonardo_scratch/fast/AIFAC_L01_028/midahl00/pixi_manifests/inference-hive/pixi.toml`|

To use a pre-installed environment, edit `pixi_manifest` in the config accordingly.

### Installing Enviroments
To install an environment, you can run
```
pixi install -e cuda-vllm
```
This will install the `cuda-vllm` environment.

### Testing Environments
You can test an environment by running
```
pixi shell -e cuda-vllm
```
This will launch a shell in the `cuda-vllm` environment.


## Usage

### 1. Dataset Preparation
This toolkit expects a huggingface dataset as input. The dataset must contain:

**Required columns:**
- **ID column**: A column of dtype string containing unique identifiers for each row (configurable via `id_column_name`)
- **Input column**: Content depends on API type (configurable via `input_column_name`):
  - For Completions: a column of dtype string containing prompts
  - For ChatCompletions: a column of conversations, in openai API compatible format:
    ```
    [
        {"role": "system", "content": "You are a pirate."},
        {"role": "user", "content": "How is the weather today?"},

    ]
    ```
    Each row contains a list of messages, each with "role" and "content".

Example Datasets:
|Type|Link|
|---|---|
|Completion|[maxidl/lmsys-prompt-1m](https://huggingface.co/datasets/maxidl/lmsys-prompt-1m)
|ChatCompletion|[maxidl/lmsys-chat-1m](https://huggingface.co/datasets/maxidl/lmsys-chat-1m)


### 2. Config File
Create a copy of `config_template.yaml` and edit it according to your needs. We recommend that you download the model you want to run in advance.

<details><summary>Example for downloading a model from huggingface</summary>

For downloading Qwen3-4B:
```bash
pixi run huggingface-cli download "Qwen/Qwen3-4B"
```
The default config template contains
```
env_vars:
  HF_HUB_OFFLINE: "1" # you should download data and model beforehand
```
and we do not recommend changing that.
</details>

### 3. Validating Config and Data (Recommended)

#### Config Validation
Before using your configuration file, validate it to catch errors early:

```bash
pixi run python validate_config.py --config my_config.yaml
```

The config validation script will check:
- YAML syntax is valid
- Required fields are present for each script
- Field values pass validation (e.g., time limits, API types)

#### Data Validation  
Before creating a slurm script, it's highly recommended to validate your input data configuration:

```bash
pixi run python validate_data.py --config my_config.yaml
```

The data validation script will check:
- Dataset can be loaded successfully with your config settings
- Required columns (`id_column_name` and `input_column_name`) exist
- ID column contains unique string identifiers
- Input data format matches the specified `api_type` (completion vs chat-completion)
- For completion: input data are strings
- For chat-completion: messages conform to OpenAI's ChatCompletionMessageParam spec


### 4. Create a slurm script
To create a slurm script from a config file, run
```
pixi run python create_slurm_script.py --config my_config.yaml --output run1
```
The directory specified with `--output` will be used to write log files for the run. The slurm script and a copy of the config will also be saved in this directory.


### 5. Submit the slurm script
```
sbatch run1/my_job.slurm
```
All logs will be written to `run1/`. The following logs will be written for each inference server instance:
- `*.log` for slurm job info, health checks, inference client logs and progress updates.
- `*-inference-server.log` for the logs of the inference server.
- `*-inference-stats.jsonl` for inference progress and performance statistics in json format.
The log filenames are prefixed following the pattern `{job-id}-{inference-server-id}-{short-hostname}`.

Additionally, there are
- `completed_shards.log`, tracking the id's of completed shards.
- `failed_shards.log`, tracking the id's of shards that ran but did not complete.
The purpose of these is to make debugging easier and to prevent re-running already completed shards.

### 6. Monitoring the run
We provide two utility scripts to monitor progress and performance statistics for a run.
- `monitor.py` can be used to print global progress and throughput statistics for a run.
  ```
  pixi run python monitor.py run1
  ```

  See `examples/fineweb2_annotations/example_logs/monitor_example_output.txt` for an example output of this script.
- `monitor_single.py` can be used to print detailed progress and throughput statistics for individual inference servers.
  ```
  pixi run python monitor_single.py run1/...-inference-stats.jsonl
  ```
  See `examples/fineweb2_annotations/example_logs/monitor_single_example_output.txt` for an example output of this script.

### 7. Outputs
The responses will be written as parquet files to the `output_path` specified in the config. Each parquet file has two columns:
1. `id`: The unique identifier from your input dataset's ID column
2. `response`: The response object, according to OpenAI spec

#### Joining outputs back to input
Since the output preserves the original IDs from your dataset, you can easily join the results back to your input data using the ID column.
- ToDo: Example using polars
- ToDo: Example using hf datasets map


## Examples
We provice some end-to-end examples in `examples/`. To request a specific example to be added, please file an issue.
|Example|Directory|Description|
|---|---|---|
|FineWeb-2 Annotations|`examples/fineweb2_annotations`|Annotate a sample of 1M FineWeb-2 documents with edu-scores using `Qwen/Qwen3-4B`.|
|Machine Translation with Gemma-3|`examples/fineweb-edu_machine_translation`|Translate 100k FineWeb-edu documents to German with Gemma-3-27b. Includes assistant message prefill to control the response format.|
|Machine Translation into 5 languages with Unbabel Tower+ 72B|`examples/fineweb-edu_mt_tower_5langs`|Translate 100k FineWeb-edu documents into 5 languages with Unbabel Tower+ 72B|

## Roadmap
- add pre-installed environments
- improve docs on how to use/post-process outputs. including how to merge outputs back to input dataset.
- support and examples for multi-node inference servers to serve large models like DSv3.
- provide some benchmark plots that demonstrate scaling (to used for compute applications)
- additional examples. one example for using SGLang instead of vLLM.
- test run on juwels booster
- create env for jupiter
- test run on jupiter
- create env for lumi
- test run on lumi

Please file an issue to suggest additional items for the roadmap


## Acknowledgements
- This project is supported by the OpenEuroLLM project, co-funded by the Digital Europe Programme under GA no. 101195233.
For more information see [openeurollm.eu](https://openeurollm.eu/).
- We acknowledge the EuroHPC Joint Undertaking for awarding this project access to the EuroHPC supercomputer LEONARDO, hosted by CINECA (Italy) and the LEONARDO consortium through an EuroHPC AI Factory Large Scale Access call.

<img src="res/eu_cofunding.png" alt="inference-hive Logo" width="300" style="vertical-align: middle;">
