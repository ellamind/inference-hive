# FineWeb-edu Machine Translation
This example demonstrates how to translate a sample of FineWeb-edu documents with Gemma-3 using `inference-hive`.


## Cloning `inference-hive`
```bash
git clone git@github.com:ellamind/inference-hive.git
cd inference-hive # make sure to be in the projects root dir.
```

## Preparing a sample from FineWeb-edu
`examples/fineweb-edu_machine_translation/prepare_dataset.py` contains example code for preparing a sample of FineWeb-edu.
Specifically, it performs the following steps:
1. Obtain a sample of FineWeb-edu documents by instantiating a huggingface dataset object via `load_dataset`.
Here we select one file of the 10BT sample of FineWeb-edu, of which we take the first 100k documents.

2. Define the prompt.
We define system and user prompt. Additionally, we prefill the assistant message to control how the response starts. For this to work, we need to set `continue_final_message` to true in the sampling parameters (when filling the config template).

3. Create a column of conversations.
We apply the prompt to each sample in the dataset. First, we truncate each document to 30k characters. Then we create message objects (conversations) in the format 
```
[
    {"role": "system", "content": system_prompt}
    {"role": "user", "content": user_prompt}
    {"role": "assistent", "content": assistant_prefill}
]
```

4. Save the dataset
We save the dataset as `fineweb-edu-mt-chat-completion` using `save_to_disk`.

To prepare the dataset, run:

```bash
pixi run python examples/fineweb-edu_machine_translation/prepare_dataset.py
```

The out is is a new directory `fineweb-edu-mt-chat-completion` that contains the prepared dataset.

## Create a Config File
Make a copy of `config_template.yaml`:
```bash
cp config_template.yaml config_fw-edu_mt.yaml
```
Then fill the config file. We provide an example here, `examples/fineweb-edu_machine_translation/config_fw-edu_mt.yaml`, however, the `SLURM Configuration` section will likely differ for you. Also make sure to edit the dataset and output paths. In this example config, we set the number of inference servers to 2. Note that we set

```yaml
  extra_body: {continue_final_message: true, "add_generation_prompt": false} # we need to set these two for assistant prefill to work.
```
to enable assistant prefill.

Also, download the model by running `huggingface-cli download "google/gemma-3-27b-it"`

## Validating
We validate the config and data loading:

```bash
pixi run python validate_config.py --config config_fw-edu_mt.yaml
# 2025-07-02 15:41:43.600 | INFO     | __main__:main:21 - Validating: examples/fineweb-edu_machine_translation/config_fw-edu_mt.yaml
# 2025-07-02 15:41:43.605 | INFO     | config:validate_and_set_num_data_shards:218 - num_data_shards not specified, defaulting to num_inference_servers (2)
# 2025-07-02 15:41:43.606 | INFO     | __main__:main:27 - ✓ Valid for create_slurm_script.py
# 2025-07-02 15:41:43.609 | INFO     | __main__:main:36 - ✓ Valid for run_inference.py
# 2025-07-02 15:41:43.609 | INFO     | __main__:main:42 - Configuration is valid!
```

```bash
pixi run python validate_data.py --config config_fw-edu_mt.yaml
# 2025-07-02 15:42:17.171 | INFO     | __main__:validate_dataset_from_config:98 - Loading configuration from: examples/fineweb-edu_machine_translation/config_fw-edu_mt.yaml
# 2025-07-02 15:42:17.176 | INFO     | __main__:validate_dataset_from_config:101 - Loading dataset for validation...
# 2025-07-02 15:42:17.176 | INFO     | __main__:validate_dataset_from_config:103 - Loading dataset with load_from_disk
# 2025-07-02 15:42:17.651 | INFO     | __main__:validate_dataset_from_config:115 - Dataset loaded: 100000 rows
# 2025-07-02 15:42:17.651 | INFO     | __main__:validate_dataset_from_config:118 - Starting data validation...
# 2025-07-02 15:42:17.723 | INFO     | __main__:validate_input_data_format:86 - Input data format validation passed for api_type='chat-completion' with string ID column 'id' using OpenAI's pydantic models
# 2025-07-02 15:42:17.724 | INFO     | __main__:validate_dataset_from_config:126 - ✓ Data validation completed successfully!
# 2025-07-02 15:42:17.724 | INFO     | __main__:main:165 - Data validation passed! Dataset is ready for inference.
```

## Create a slurm script
Next, use `create_slurm_script.py` to create a slurm script for the `config_fw-edu_mt.yaml` config and with `fw2_annotations_run1` as the job's logging directory.
```bash
pixi run python create_slurm_script.py --config config_fw-edu_mt.yaml --output fw-edu_mt_run1
# 2025-07-02 15:43:03.393 | INFO     | config:validate_and_set_num_data_shards:218 - num_data_shards not specified, defaulting to num_inference_servers (2)
# 2025-07-02 15:43:03.470 | INFO     | __main__:main:329 - Output directory: fw-edu_mt_run1
# 2025-07-02 15:43:03.472 | INFO     | __main__:main:359 - Config copied to: fw-edu_mt_run1/config_fw-edu_mt.yaml
# 2025-07-02 15:43:03.474 | INFO     | __main__:main:370 - SLURM job script generated successfully: fw-edu_mt_run1/fw-edu-mt.slurm
# 2025-07-02 15:43:03.474 | INFO     | __main__:main:375 - To submit the job: sbatch fw-edu_mt_run1/fw-edu-mt.slurm
# 2025-07-02 15:43:03.474 | INFO     | __main__:main:376 - To cancel all jobs: scancel --name=fw-edu-mt
# 2025-07-02 15:43:03.474 | INFO     | __main__:main:377 - To check job status: squeue -u $USER --name=fw-edu-mt
```

## Submit the job
To submit the job, run
```bash
sbatch fw-edu_mt_run1/fw-edu-mt.slurm
```
to submit the job. Log files will be written to `fw-edu_mt_run1` once it starts running.
You can check the job status using 
```bash
squeue -u $USER --name=fw-edu-mt
```

<details><summary>Some logs for running this example</summary>

SLURM queue:
```
JOBID     USER      PARTITION      ACCOUNT        NODES  STATE     TIME      NAME                          NODELIST(REASON)
17254305  midahl00  boost_usr_prod aifac_l01_028  1      RUNNING   9:18      fw-edu-mt                     lrdn0027
17254306  midahl00  boost_usr_prod aifac_l01_028  1      RUNNING   9:18      fw-edu-mt                     lrdn0203
```

Log files:
```bash
ls -1 ./fw-edu_mt_run1
17253723-1-lrdn3250-inference-server.log
17253723-1-lrdn3250.log
17253723-2-lrdn0027-inference-server.log
17253723-2-lrdn0027.log
config_fw-edu_mt.yaml
fw-edu-mt.slurm
```
</details>

You can find example logs in `examples/fineweb-edu_machine_translation/example_logs`.

Lets cancel the job after some time, to not waste compute:
```
scancel --name fw-edu-mt
```

## Monitoring Progress and Throughput

To print global progress and throughput statistics for the run, execute
```
pixi run python monitor.py fw-edu_mt_run1
```

To take a detailed look at the progress & performance for an individual shard, you can select one of the `.jsonl` logs (here `fw-edu_mt_run1/17254305-1-lrdn0203-inference-stats.jsonl`) and run
```
pixi run python monitor_single.py fw-edu_mt_run1/17254305-1-lrdn0203-inference-stats.jsonl
```
This will print shard & progress status and throughput statistics (over time)

You can find example outputs for these commands in `examples/fineweb-edu_machine_translation/example_logs`.


## Using the Outputs
The outputs were saved to `example_outputs/fw-edu-100k-responses-gemma-3-27b`.

<details><summary>Output files</summary>

```bash
ls -1 example_outputs/fw-edu-100k-responses-gemma-3-27b
shard000000_part000000.zstd.parquet
shard000001_part000000.zstd.parquet

```
</details>

The responses are written to parquet files. Each has two columns:
1. "id", to uniquely identify the input row from your dataset
2. "response", containing the response object according to OpenAI spec.

Lets print the first response:
First, start a python shell using `pixi run python`. Then run
```python
import polars as pl

# Load the parquet outputs and get the first row
df = pl.scan_parquet("example_outputs/fw-edu-100k-responses-gemma-3-27b").head(1).collect()

# Print the response from the first row
print(df.to_dicts()[0])
```

<details><summary>Output</summary>

```python
{
    "id": "<urn:uuid:f51e4a20-9e4e-460d-8a25-cfaccf052f8e>",
    "response": {
        "id": "chatcmpl-b5ac7cd5bfba43a1929ba030a3f665cf",
        "choices": [
            {
                "finish_reason": "stop",
                "index": 0,
                "logprobs": None,
                "message": {
                    "content": "\nDiese LEDs sind mit einem Thermistor verbunden, der etwas heißer läuft als die Raumtemperatur. Indem man auf den Thermistor bläst, kühlt das Geburtstagskind ihn ab und erhöht somit den Widerstand. Der Mikrocontroller erkennt dies und schaltet daraufhin einige der LEDs aus. Bauen Sie so ein Gerät und Sie müssen sich nie wieder Sorgen um geschmolzenes Wachs auf Ihrer Torte machen. Detaillierte Anleitungen finden Sie auf Instructables.\n</translated_document>",
                    "refusal": None,
                    "role": "assistant",
                    "annotations": None,
                    "audio": None,
                    "function_call": None,
                    "tool_calls": [],
                },
            }
        ],
        "created": 1751464989,
        "model": "google/gemma-3-27b-it",
        "object": "chat.completion",
        "service_tier": None,
        "system_fingerprint": None,
        "usage": {
            "completion_tokens": 105,
            "prompt_tokens": 192,
            "total_tokens": 297,
            "completion_tokens_details": None,
            "prompt_tokens_details": None,
        },
    },
}

```

Lets print the content of the first response
```python
print(
    df.select(
        pl.col("response")
        .struct.field("choices")
        .list.get(0)
        .struct.field("message")
        .struct.field("content")
    ).item()
)
# Diese LEDs sind mit einem Thermistor verbunden, der etwas heißer läuft als die Raumtemperatur. Indem man auf den Thermistor bläst, kühlt das Geburtstagskind ihn ab und erhöht somit den Widerstand. Der Mikrocontroller erkennt dies und schaltet daraufhin einige der LEDs aus. Bauen Sie so ein Gerät und Sie müssen sich nie wieder Sorgen um geschmolzenes Wachs auf Ihrer Torte machen. Detaillierte Anleitungen finden Sie auf Instructables.
# </translated_document>
```
</details>