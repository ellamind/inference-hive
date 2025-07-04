# FineWeb-edu Machine Translation
This example demonstrates how to translate a sample of FineWeb-edu documents into 5 different languages with Unbabel/Tower-Plus-72B using `inference-hive`.


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

2. Define the target languages. We then transform the dataset into rows of `(id,doc,target_lang)`. The dataset is now 500k rows.

3. Define the prompt.
We define user prompt according to `huggingface.co/Unbabel/Tower-Plus-72B`.

4. Create a column of conversations.
We apply the prompt to each sample in the dataset. First, we truncate each document to 30k characters. Then we create message objects (conversations) in the format 
```
[
    {"role": "user", "content": user_prompt}
]
```

5. Save the dataset
We save the dataset as `fineweb-edu-mt-5langs` using `save_to_disk`.

To prepare the dataset, run:

```bash
pixi run python examples/fineweb-edu_mt_tower_5langs/prepare_dataset.py
```

The out is is a new directory `fineweb-edu-mt-5langs` that contains the prepared dataset.

## Create a Config File
Make a copy of `config_template.yaml`:
```bash
cp config_template.yaml config_fw-edu_mt_tower_5langs.yaml
```
Then fill the config file. We provide an example here, `examples/fineweb-edu_mt_tower_5langs/config_fw-edu_mt_tower_5langs.yaml`, however, the `SLURM Configuration` section will likely differ for you. Also make sure to edit the dataset and output paths. In this example config, we set the number of inference servers to 20.

Also, download the model by running `pixi run huggingface-cli download "Unbabel/Tower-Plus-72B"`

## Validating
We validate the config and data loading:

```bash
pixi run python validate_config.py --config config_fw-edu_mt_tower_5langs.yaml
# 2025-07-04 09:26:28.441 | INFO     | __main__:main:21 - Validating: examples/fineweb-edu_mt_tower_5langs/config_fw-edu_mt_tower_5langs.yaml
# 2025-07-04 09:26:28.446 | INFO     | config:validate_and_set_num_data_shards:218 - num_data_shards not specified, defaulting to num_inference_servers (20)
# 2025-07-04 09:26:28.446 | INFO     | __main__:main:27 - ✓ Valid for create_slurm_script.py
# 2025-07-04 09:26:28.449 | INFO     | __main__:main:36 - ✓ Valid for run_inference.py
# 2025-07-04 09:26:28.449 | INFO     | __main__:main:42 - Configuration is valid!
```

```bash
pixi run python validate_data.py --config config_fw-edu_mt_tower_5langs.yaml
# 2025-07-04 09:27:01.752 | INFO     | __main__:validate_dataset_from_config:98 - Loading configuration from: examples/fineweb-edu_mt_tower_5langs/config_fw-edu_mt_tower_5langs.yaml
# 2025-07-04 09:27:01.757 | INFO     | __main__:validate_dataset_from_config:101 - Loading dataset for validation...
# 2025-07-04 09:27:01.757 | INFO     | __main__:validate_dataset_from_config:103 - Loading dataset with load_from_disk
# 2025-07-04 09:27:02.681 | INFO     | __main__:validate_dataset_from_config:115 - Dataset loaded: 500000 rows
# 2025-07-04 09:27:02.681 | INFO     | __main__:validate_dataset_from_config:118 - Starting data validation...
# 2025-07-04 09:27:02.793 | INFO     | __main__:validate_input_data_format:86 - Input data format validation passed for api_type='chat-completion' with string ID column 'id' using OpenAI's pydantic models
# 2025-07-04 09:27:02.793 | INFO     | __main__:validate_dataset_from_config:126 - ✓ Data validation completed successfully!
# 2025-07-04 09:27:02.796 | INFO     | __main__:main:165 - Data validation passed! Dataset is ready for inference.
```

## Create a slurm script
Next, use `create_slurm_script.py` to create a slurm script for the `config_fw-edu_mt_tower_5langs.yaml` config and with `fw-edu_mt_tower_5langs_run1` as the job's logging directory.
```bash
pixi run python create_slurm_script.py --config config_fw-edu_mt_tower_5langs.yaml --output fw-edu_mt_tower_5langs_run1
# 2025-07-04 09:28:02.933 | INFO     | config:validate_and_set_num_data_shards:218 - num_data_shards not specified, defaulting to num_inference_servers (20)
# 2025-07-04 09:28:02.950 | INFO     | __main__:main:329 - Output directory: fw-edu_mt_tower_5langs_run1
# 2025-07-04 09:28:02.954 | INFO     | __main__:main:359 - Config copied to: fw-edu_mt_tower_5langs_run1/config_fw-edu_mt_tower_5langs.yaml
# 2025-07-04 09:28:02.956 | INFO     | __main__:main:370 - SLURM job script generated successfully: fw-edu_mt_tower_5langs_run1/fw-edu-mt-tower.slurm
# 2025-07-04 09:28:02.956 | INFO     | __main__:main:375 - To submit the job: sbatch fw-edu_mt_tower_5langs_run1/fw-edu-mt-tower.slurm
# 2025-07-04 09:28:02.956 | INFO     | __main__:main:376 - To cancel all jobs: scancel --name=fw-edu-mt-tower
# 2025-07-04 09:28:02.956 | INFO     | __main__:main:377 - To check job status: squeue -u $USER --name=fw-edu-mt-tower
```

## Submit the job
To submit the job, run
```bash
sbatch fw-edu_mt_tower_5langs_run1/fw-edu-mt-tower.slurm
```
to submit the job. Log files will be written to `fw-edu_mt_tower_5langs_run1` once it starts running.
You can check the job status using 
```bash
squeue -u $USER --name=fw-edu-mt-tower
```

<details><summary>Some logs for running this example</summary>

SLURM queue:
```
JOBID     USER      PARTITION      ACCOUNT        NODES  STATE     TIME      NAME                          NODELIST(REASON)
17328678  midahl00  boost_usr_prod aifac_l01_028  1      RUNNING   14:47     fw-edu-mt-tower               lrdn2279
17328710  midahl00  boost_usr_prod aifac_l01_028  1      RUNNING   16:55     fw-edu-mt-tower               lrdn1616
17328711  midahl00  boost_usr_prod aifac_l01_028  1      RUNNING   16:55     fw-edu-mt-tower               lrdn2246
17328712  midahl00  boost_usr_prod aifac_l01_028  1      RUNNING   16:55     fw-edu-mt-tower               lrdn2581
17328706  midahl00  boost_usr_prod aifac_l01_028  1      RUNNING   16:55     fw-edu-mt-tower               lrdn2616
17328713  midahl00  boost_usr_prod aifac_l01_028  1      RUNNING   16:55     fw-edu-mt-tower               lrdn2624
17328714  midahl00  boost_usr_prod aifac_l01_028  1      RUNNING   16:55     fw-edu-mt-tower               lrdn2639
17328715  midahl00  boost_usr_prod aifac_l01_028  1      RUNNING   16:55     fw-edu-mt-tower               lrdn2647
17328717  midahl00  boost_usr_prod aifac_l01_028  1      RUNNING   16:55     fw-edu-mt-tower               lrdn2651
17328716  midahl00  boost_usr_prod aifac_l01_028  1      RUNNING   16:55     fw-edu-mt-tower               lrdn2666
17328718  midahl00  boost_usr_prod aifac_l01_028  1      RUNNING   16:55     fw-edu-mt-tower               lrdn2671
17328720  midahl00  boost_usr_prod aifac_l01_028  1      RUNNING   16:55     fw-edu-mt-tower               lrdn2679
17328719  midahl00  boost_usr_prod aifac_l01_028  1      RUNNING   16:55     fw-edu-mt-tower               lrdn2691
17328721  midahl00  boost_usr_prod aifac_l01_028  1      RUNNING   16:55     fw-edu-mt-tower               lrdn2703
17328722  midahl00  boost_usr_prod aifac_l01_028  1      RUNNING   16:55     fw-edu-mt-tower               lrdn2709
17328723  midahl00  boost_usr_prod aifac_l01_028  1      RUNNING   16:55     fw-edu-mt-tower               lrdn2736
17328707  midahl00  boost_usr_prod aifac_l01_028  1      RUNNING   16:55     fw-edu-mt-tower               lrdn2824
17328708  midahl00  boost_usr_prod aifac_l01_028  1      RUNNING   16:55     fw-edu-mt-tower               lrdn3163
17328705  midahl00  boost_usr_prod aifac_l01_028  1      RUNNING   16:55     fw-edu-mt-tower               lrdn3262
17328709  midahl00  boost_usr_prod aifac_l01_028  1      RUNNING   16:55     fw-edu-mt-tower               lrdn3264
```

```
</details>

You can find example logs in `examples/fineweb-edu_mt_tower_5langs/example_logs`.

Lets cancel the job after some time, to not waste compute:
```
scancel --name fw-edu-mt-tower 
```

## Monitoring Progress and Throughput

To print global progress and throughput statistics for the run, execute
```
pixi run python monitor.py fw-edu_mt_tower_5langs_run1
```

To take a detailed look at the progress & performance for an individual shard, you can select one of the `.jsonl` logs (here `fw-edu_mt_tower_5langs_run1/17328678-1-lrdn3262-inference-stats.jsonl`) and run
```
pixi run python monitor_single.py fw-edu_mt_tower_5langs_run1/17328678-1-lrdn3262-inference-stats.jsonl
```
This will print shard & progress status and throughput statistics (over time)

You can find example outputs for these commands in `examples/fineweb-edu_machine_translation/example_logs`.


## Using the Outputs
The outputs were saved to `example_outputs/fineweb-edu-mt-5langs-responses`.

<details><summary>Output files</summary>

```bash
ls -1 example_outputs/fineweb-edu-mt-5langs-responses
shard000000_part000000.zstd.parquet
shard000001_part000000.zstd.parquet
shard000002_part000000.zstd.parquet
shard000003_part000000.zstd.parquet
shard000004_part000000.zstd.parquet
shard000005_part000000.zstd.parquet
shard000006_part000000.zstd.parquet
shard000007_part000000.zstd.parquet
shard000008_part000000.zstd.parquet
shard000009_part000000.zstd.parquet
shard000010_part000000.zstd.parquet
shard000011_part000000.zstd.parquet
shard000012_part000000.zstd.parquet
shard000013_part000000.zstd.parquet
shard000014_part000000.zstd.parquet
shard000015_part000000.zstd.parquet
shard000016_part000000.zstd.parquet
shard000017_part000000.zstd.parquet
shard000018_part000000.zstd.parquet
shard000019_part000000.zstd.parquet
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
df = pl.scan_parquet("example_outputs/fineweb-edu-mt-5langs-responses").head(1).collect()

# Print the response from the first row
print(df.to_dicts()[0])
```

<details><summary>Output</summary>

```python
{
    "id": "<urn:uuid:f51e4a20-9e4e-460d-8a25-cfaccf052f8e>_German",
    "response": {
        "id": "chatcmpl-94194dce220448c2a5ae3599da99bffb",
        "choices": [
            {
                "finish_reason": "stop",
                "index": 0,
                "logprobs": None,
                "message": {
                    "content": "Diese LEDs sind mit einem Thermistor verbunden, der etwas heißer als die Umgebungstemperatur läuft. Wenn der Geburtstagskind also auf den Thermistor bläst, kühlt es ihn ab und erhöht dadurch den Widerstand. Der Mikrocontroller erkennt dies und schaltet einige der LEDs aus. Mit einem solchen Gerät müssen Sie sich nie wieder Sorgen um geschmolzenes Wachs auf Ihrem Kuchen machen. Für detaillierte Anweisungen besuchen Sie Instructables.",
                    "refusal": None,
                    "role": "assistant",
                    "annotations": None,
                    "audio": None,
                    "function_call": None,
                    "tool_calls": [],
                },
            }
        ],
        "created": 1751615116,
        "model": "Unbabel/Tower-Plus-72B",
        "object": "chat.completion",
        "service_tier": None,
        "system_fingerprint": None,
        "usage": {
            "completion_tokens": 114,
            "prompt_tokens": 118,
            "total_tokens": 232,
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
# Diese LEDs sind mit einem Thermistor verbunden, der etwas heißer als die Umgebungstemperatur läuft. Wenn der Geburtstagskind also auf den Thermistor bläst, kühlt es ihn ab und erhöht dadurch den Widerstand. Der Mikrocontroller erkennt dies und schaltet einige der LEDs aus. Mit einem solchen Gerät müssen Sie sich nie wieder Sorgen um geschmolzenes Wachs auf Ihrem Kuchen machen. Für detaillierte Anweisungen besuchen Sie Instructables.
```
</details>