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
- **📁 Easy Data Loading**: Works with any HuggingFace datasets that contains prompts or conversations. Scales to large parquet datasets.
- **✅ Data Validation**: Pre-flight validation to catch data configuration issues.
- **⚙️ Highly Configurable**: YAML-based configuration for inference parameters and cluster setup.


## Environment
We use [Pixi](https://pixi.sh/dev/) for environment management. Refer to the [Pixi Installation Guide](https://pixi.sh/dev/installation/).
<details><summary>Oh no, another env manager</summary>

`pixi` is great. You should try it and get rid of `conda`, `peotry`, etc.

</details>
<br>

We provide multiple environment definitions for different GPU types and different inference servers.
|Env Name|Description|Tested on|
|---|---|---|
|cuda-vllm|CUDA + vLLM|Leonardo|
|cuda-sglang|CUDA + SGLang|Leonardo|
|cpu|no inference server, debug env|Leonardo|

### Pre-Installed Environments
On some clusters we have pre-installed environments which you can re-use if you are part of the same project. This allows you to skip environment installation, saves storage and inodes.
|Cluster|Project|Env Name|Manifest|
|---|---|---|---|
|Leonardo Booster|MultiSynt (AIFAC_L01_028)|cuda-vllm|`/leonardo_scratch/fast/AIFAC_L01_028/midahl00/pixi_manifests/inference-hive/pixi.toml`|


### Installing Enviroments
To install an environments, you can run
```bash
# to install the cuda-vllm environment
pixi install -e cuda-vllm
# to install all environments run:
pixi install --all
```

### Using an Environment
To use a pre-installed environment for your inference-hive runs, set `pixi_manifest` to the path of the corresponding `pixi.toml` file in your config.

To activate an environment in your shell, you can run
```bash
pixi shell -e cuda-vllm
# or to use a pre-installed env:
pixi shell -e cuda-vllm --manifest-path /path/to/existing/pixi.toml
```
This will launch a shell in the `cuda-vllm` environment.



## Usage

### 0. Prerequisites
All commands below assume that you are in the `inference-hive` root directory and that your shell has the default env activated. You can activate it by running `pixi shell`, or `pixi shell --manifest-path /path/to/existing/pixi.toml` to use a pre-installed env.

### 1. Dataset Preparation
This toolkit ingests data through huggingface datasets or parquet files. The dataset must contain:

#### Required columns:
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

#### On-the-fly Input Transformation
We support on-the-fly input transformations via user-defined-functions (udf) that are applied to each row before processing.
You can use a udf to transform your dataset into the right format on-the-fly.
This is meant for lightweight operations like applying a prompt or converting conversations into the right format by creating the required columns from other columns.
In general, we recommend converting your input data into the right format as a separate pre-processing step.
See `udf.py` `format_prompt()` for an example of a udf.

#### Example Datasets in correct format:
|Type|Link|
|---|---|
|Completion|[maxidl/lmsys-prompt-1m](https://huggingface.co/datasets/maxidl/lmsys-prompt-1m)
|ChatCompletion|[maxidl/lmsys-chat-1m](https://huggingface.co/datasets/maxidl/lmsys-chat-1m)


### 2. Configuration
Runs in inference-hive are configured via a `yaml`-based config file. We provide a template for you to start from:
1. Create a copy of `config_template.yaml` and give it a proper name.
2. Edit it according to your needs. If you have questions or face any problems, please file an issue.


### 3. Working offline
We recommend that you download the dataset and model you want to run in advance. Most clusters require this anyway, as compute nodes are often not connected to the public internet. Downloading data/model from multiple nodes at the same time can lead to issues.
The default config template therefore contains
```
env_vars:
  HF_HUB_OFFLINE: "1" # you should download data and model beforehand
```
and we do not recommend changing that.

<details><summary>Example for downloading a model from huggingface</summary>

For downloading Qwen3-4B:
```bash
hf download "Qwen/Qwen3-4B"
# or to download to a local directory instead of hf cache:
hf download "Qwen/Qwen3-4B" --local-dir /path/to/some/dir
```
For large scale runs with many nodes we recommend storing the model weights on a fast filesystem as they will be read many times.
</details>

### 4. Validating Config and Data (Optional)
We recommend that you follow the steps below to validate both your config file and input data setup.

#### Config Validation
Before using your configuration file, validate it to catch errors early:

```bash
python validate_config.py --config my_config.yaml
```

The config validation script will check whether:
- the YAML syntax is valid
- all required fields are present
- field values pass validation (e.g., time limits, API types)

#### Data Validation  
Before creating a slurm script, it's highly recommended to validate your input data configuration:

```bash
python validate_data.py --config my_config.yaml
```

The data validation script will check whether:
- the dataset can be loaded successfully with your config settings
- required columns (`id_column_name` and `input_column_name`) exist (after applying `udf` if configured)
- the id column contains string identifiers (we do not check uniqueness)
- the input data format matches the specified `api_type` (completion vs chat-completion)
  - For completion: input data are strings
  - For chat-completion: messages conform to OpenAI's ChatCompletionMessageParam spec

It also prints a couple of samples, so you can **take a look at the input data and ensure it is correct**.

### 5. Create a Run
A run in inference-hive consists of the following parts:
- a run directory (`run-dir`), that contains everything belonging to a run:
  - an automatically created slurm script used to submit jobs for the run (`ih_job.slurm`).
  - a `logs/` directory, where slurm job logs are written to. Each job writes:
    - a `{shard}-{jobid}-{node}.log` for slurm job info, health checks, inference client logs and progress updates.
    - a `{shard}-{jobid}-{node}-inference-server.log` containing the logs of the inference server.
  - a `progress/` directory, containing:
    - a `{shard}-progress.jsonl` file for each shard, where jobs log per-shard progress and throughput metrics.
    - `shards_completed.log`, keeping track of already completed shards.
    - `shards_failed.log`, containing info on shards for which jobs failed.
  - for reproducibility:
    - a copy of the config file used to create the slurm script (`ih_config.yaml`).
    - a copy of `udf.py` in case there are any udf implemented.

To create a run from a config file, use
```
python create_run.py --config my_config.yaml --output run1
```
where the directory specified with `--output` will be used as the `run-dir`.


### 6. Submitting Jobs
To submit jobs for a run to slurm, use
```bash
python submit.py --run-dir run1
# or to submit at most 4 jobs:
python submit.py --run-dir run1 --limit 2
```

> You should not submit a `ih_job.slurm` file via sbatch manually. `submit.py` contains a couple of additional checks, e.g., to prevent submitting multiple jobs for the same shard or submitting jobs for already completed shards.

If things went right, you should now see corresponding jobs in the slurm queue (`squeue --me`).
For large-scale workloads, we recommend to start with a small number of jobs (e.g., by setting `--limit 2`), and once you are confident that they run as intended (check the corresponding log files), scale the workload by increasing or omitting the limit.


### 7. Monitoring Runs
We provide a utility script to monitor progress and performance statistics for a run.
`status.py` can be used to print job status, as well as progress and throughput statistics for a run.
```bash
python status.py --run-dir run1
# use --detailed to display progress and throughput stats
python status.py --run-dir run1 --detailed
# use --shards to only display specific shards
python status.py --run-dir run1 --detailed --shards 0 1 2 3
```

### 8. Canceling Jobs
You can use `cancel.py` to cancel all jobs belonging to a run.

```bash
python cancel.py --run-dir run1
```

### 9. Outputs
The responses will be written as parquet files (one per shard) to the `output_path` specified in the config. inference-hive takes care of regular checkpointing and automatically compacts multiple small parquet files into larger ones.

Each output parquet file has two columns:
1. `id`: The unique identifier from your input dataset's ID column.
2. `response`: The response object, according to OpenAI spec.

#### Joining outputs back to input
You can join responses and your input dataset using the `id` column.
- ToDo: Provide an example.


## Examples
We provice some end-to-end examples in `examples/`. To request a specific example to be added, please file an issue.
|Example|Directory|Description|
|---|---|---|
|FineWeb-2 Annotations|`examples/fineweb2_annotations`|Annotate a sample of 1M FineWeb-2 documents with edu-scores a la fineweb-edu using `Qwen/Qwen3-4B`.|
|Machine Translation with Gemma-3 (outdated)|`examples/fineweb-edu_machine_translation`|Translate 100k FineWeb-edu documents to German with Gemma-3-27b. Includes assistant message prefill to control the response format.|

## Roadmap
- add pre-installed environments on multiple clusters.
- improve docs on how to use/post-process outputs, including how to merge outputs. back to input dataset.
- provide some benchmark plots that demonstrate scaling (to used for compute applications).
- add additional examples:
  - for using SGLang instead of vLLM.
  - for multi-node inference servers to serve models that do not fit on one node.

Please file an issue to suggest additional items for the roadmap.


## Acknowledgements
- This project is supported by the OpenEuroLLM project, co-funded by the Digital Europe Programme under GA no. 101195233.
For more information see [openeurollm.eu](https://openeurollm.eu/).
- This project is supported by the LLMs4EU project, co-funded by the Digital Europe Programme under GA no. 101198470.
For more information see [LLMs4EU website](https://www.alt-edic.eu/projects/llms4eu/)
- We acknowledge the EuroHPC Joint Undertaking for supporting this project through access to the EuroHPC supercomputer LEONARDO, hosted by CINECA (Italy) and the LEONARDO consortium, through an EuroHPC AI Factory Large Scale Access call.

<img src="res/eu_cofunding.png" alt="inference-hive Logo" width="300" style="vertical-align: middle;">
