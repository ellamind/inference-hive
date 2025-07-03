# FineWeb-2 Annotations
This example demonstrates how to annotate a sample of FineWeb-2 documents with edu-scores using `inference-hive`.


## Cloning `inference-hive`
```bash
git clone git@github.com:ellamind/inference-hive.git
cd inference-hive # make sure to be in the projects root dir.
```

## Preparing a sample from FineWeb-2
`examples/fineweb2_annotations/prepare_dataset.py` contains example code for preparing a sample of FineWeb-2.
Specifically, it performs the following steps:
1. Obtain a sample of FineWeb-2 documents by instantiating a huggingface dataset object via `load_dataset`.
Here we select one file of the German (deu_Latn) subset of FineWeb-2, of which we take the first 1M documents.

2. Define the prompt.
We use the prompt from the original [fineweb-edu-classifier](https://huggingface.co/HuggingFaceFW/fineweb-edu-classifier/blob/main/utils/prompt.txt).

3. Create a column of conversations.
We apply the prompt to each sample in the dataset. First, we truncate each document to 30k characters. Then we create message objects (conversations) in the format `[{"role": "user", "content": prompt}]` where `prompt` is the prompt formatted with the truncated document.

4. Save the dataset
We save the dataset as `fineweb2-deu_Latn-1M-chat-completion` using `save_to_disk`.

To prepare the dataset, run:
```bash
pixi run python examples/fineweb2_annotations/prepare_dataset.py
```
The out is is a new directory `fineweb2-deu_Latn-1M-chat-completion` that contains the prepared dataset.

## Create a Config File
Make a copy of `config_template.yaml`:
```bash
cp config_template.yaml config_fw2_annotations.yaml
```
Then fill the config file. We provide an example here, `examples/fineweb2_annotations/config_fw2_annotations.yaml`, however, the `SLURM Configuration` section will likely differ for you. Also make sure to edit the dataset and output paths. In this example config, we set the number of inference servers to 8.

Also, download the model by running `huggingface-cli download "Qwen/Qwen3-4B"`

## Validating
We validate the config and data loading:

```bash
pixi run python validate_config.py --config config_fw2_annotations.yaml
# 2025-06-30 11:54:13.203 | INFO     | __main__:main:21 - Validating: config_fw2_annotations.yaml
# 2025-06-30 11:54:13.208 | INFO     | __main__:main:27 - ✓ Valid for create_slurm_script.py
# 2025-06-30 11:54:13.211 | INFO     | __main__:main:36 - ✓ Valid for run_inference.py
# 2025-06-30 11:54:13.211 | INFO     | __main__:main:42 - Configuration is valid!
```

```bash
pixi run python validate_data.py --config config_fw2_annotations.yaml
# 2025-06-30 11:51:52.391 | INFO     | __main__:validate_dataset_from_config:98 - Loading configuration from: config_fw2_annotations.yaml
# 2025-06-30 11:51:52.395 | INFO     | __main__:validate_dataset_from_config:101 - Loading dataset for validation...
# 2025-06-30 11:51:52.395 | INFO     | __main__:validate_dataset_from_config:103 - Loading dataset with load_from_disk
# Loading dataset from disk: 100%|███████████████████████████████████████████████████████████████████████████████| 17/17 [00:00<00:00, 1899.09it/s]
# 2025-06-30 11:51:52.433 | INFO     | __main__:validate_dataset_from_config:115 - Dataset loaded: 1000000 rows
# 2025-06-30 11:51:52.433 | INFO     | __main__:validate_dataset_from_config:118 - Starting data validation...
# 2025-06-30 11:51:52.481 | INFO     | __main__:validate_input_data_format:86 - Input data format validation passed for api_type='chat-completion' with string ID column 'id' using OpenAI's pydantic models
# 2025-06-30 11:51:52.481 | INFO     | __main__:validate_dataset_from_config:126 - ✓ Data validation completed successfully!
# 2025-06-30 11:51:52.486 | INFO     | __main__:main:165 - Data validation passed! Dataset is ready for inference.
```

## Create a slurm script
Next, use `create_slurm_script.py` to create a job for the `config_fw2_annotations.yaml` config and with `fw2_annotations_run1` as the job's logging directory.
```bash
pixi run python create_slurm_script.py --config config_fw2_annotations.yaml --output fw2_annotations_run1
# 2025-06-30 12:02:12.496 | INFO     | __main__:main:323 - Output directory: fw2_annotations_run1
# 2025-06-30 12:02:12.499 | INFO     | __main__:main:349 - Config copied to: fw2_annotations_run1/config_fw2_annotations.yaml
# 2025-06-30 12:02:12.500 | INFO     | __main__:main:360 - SLURM job script generated successfully: fw2_annotations_run1/fw2-annotations-deu_Latn-1M.slurm
# 2025-06-30 12:02:12.500 | INFO     | __main__:main:365 - To submit the job: sbatch fw2_annotations_run1/fw2-annotations-deu_Latn-1M.slurm
# 2025-06-30 12:02:12.500 | INFO     | __main__:main:366 - To cancel all jobs: scancel --name=fw2-annotations-deu_Latn-1M
# 2025-06-30 12:02:12.500 | INFO     | __main__:main:367 - To check job status: squeue -u $USER --name=fw2-annotations-deu_Latn-1M
```

## Submit the job
To submit the job, run
```bash
sbatch fw2_annotations_run1/fw2-annotations-deu_Latn-1M.slurm
```
to submit the job. Log files will be written to `fw2_annotations_run1` once it starts running.
You can check the job status using 
```bash
squeue -u $USER --name=fw2-annotations-deu_Latn-1M
```

<details><summary>Some logs for running this example</summary>

SLURM queue:
```
JOBID     USER      PARTITION      ACCOUNT        NODES  STATE     TIME      NAME                          NODELIST(REASON)
17011973  midahl00  boost_usr_prod aifac_l01_028  1      RUNNING   12:36     fw2-annotations-deu_Latn-1M   lrdn1355
17011974  midahl00  boost_usr_prod aifac_l01_028  1      RUNNING   12:36     fw2-annotations-deu_Latn-1M   lrdn1440
...
17011971  midahl00  boost_usr_prod aifac_l01_028  1      RUNNING   12:36     fw2-annotations-deu_Latn-1M   lrdn2927
17011972  midahl00  boost_usr_prod aifac_l01_028  1      RUNNING   12:36     fw2-annotations-deu_Latn-1M   lrdn3211
```

Log files:
```bash
ls -1 ./fw2_annotations_run1
17208675-1-lrdn2163-inference-server.log
17208675-1-lrdn2163-inference-stats.jsonl
17208675-1-lrdn2163.log
...
17208675-8-lrdn2292-inference-server.log
17208675-8-lrdn2292-inference-stats.jsonl
17208675-8-lrdn2292.log
config_fw2_annotations.yaml
fw2-annotations-deu_Latn-1M.slurm
```
</details>

You can find example logs in `examples/fineweb2_annotations/example_logs`.

Lets cancel the job after some time, to not waste compute:
```
scancel --name fw2-annotations-deu_Latn-1M
```

## Monitoring Progress and Throughput

To print global progress and throughput statistics for the run, execute
```
pixi run python monitor.py fw2_annotations_run1
```

To take a detailed look at the progress & performance for an individual shard, you can select one of the `.jsonl` logs (here `fw2_annotations_run1/17208675-1-lrdn2163-inference-stats.jsonl`) and run
```
pixi run python monitor_single.py fw2_annotations_run1/17208675-1-lrdn2163-inference-stats.jsonl
```
This will print shard & progress status and throughput statistics (over time)

You can find example outputs for these commands in `examples/fineweb2_annotations/example_logs`.

## Using the Outputs
The outputs were saved to `example_outputs/fineweb2-deu_Latn-1M-responses-qwen3-4b`.

<details><summary>Output files</summary>

```bash
ls -1 example_outputs/fineweb2-deu_Latn-1M-responses-qwen3-4b
shard000000_part000000.zstd.parquet
shard000001_part000000.zstd.parquet
shard000002_part000000.zstd.parquet
shard000003_part000000.zstd.parquet
shard000004_part000000.zstd.parquet
shard000005_part000000.zstd.parquet
shard000006_part000000.zstd.parquet
shard000007_part000000.zstd.parquet
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
df = pl.scan_parquet("example_outputs/fineweb2-deu_Latn-1M-responses-qwen3-4b").head(1).collect()

# Print the response from the first row
print(df.to_dicts()[0])
```

<details><summary>Output</summary>

```python
{
    "id": "<urn:uuid:8a4af1a1-462b-4a61-9e67-64d3d3a4770c>",
    "response": {
        "id": "chatcmpl-b87186a7f0404cb29748844550b0ef62",
        "choices": [
            {
                "finish_reason": "stop",
                "index": 0,
                "logprobs": None,
                "message": {
                    "content": "<think>\nOkay, let's evaluate this extract. The user wants to know if it's educational for primary to grade school levels using a 5-point system.\n\nFirst, the content is in German and talks about Italian politics, corruption, and Berlusconi. There's a lot of opinion and personal views, not factual information. It mentions political figures and events but doesn't explain concepts in an educational way. The language is informal and emotional, with no structured lessons or explanations. There's no mention of curricula or educational standards. The text is more of a personal commentary than an educational resource. It doesn't provide any structured information or key concepts relevant to school subjects. The user might be looking for something more factual and organized. So, it doesn't meet the criteria for even the first point. The score should be low.\n</think>\n\nThe extract is a personal commentary on Italian politics, filled with opinions, emotional language, and informal tone. It lacks structured educational content, coherent explanations, or alignment with school curricula. While it touches on political themes, it does not provide factual information, organized concepts, or pedagogical value suitable for primary/grade school education. No points are earned for relevance, coherence, or educational utility.  \n\nEducational score: 0",
                    "refusal": None,
                    "role": "assistant",
                    "annotations": None,
                    "audio": None,
                    "function_call": None,
                    "tool_calls": [],
                },
            }
        ],
        "created": 1751384431,
        "model": "Qwen/Qwen3-4B",
        "object": "chat.completion",
        "service_tier": None,
        "system_fingerprint": None,
        "usage": {
            "completion_tokens": 257,
            "prompt_tokens": 1138,
            "total_tokens": 1395,
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
# "<think>\nOkay, let's evaluate this extract. The user wants to know if it's educational for primary to grade school levels using a 5-point system.\n\nFirst, the content is in German and talks about Italian politics, corruption, and Berlusconi. There's a lot of opinion and personal views, not factual information. It mentions political figures and events but doesn't explain concepts in an educational way. The language is informal and emotional, with no structured lessons or explanations. There's no mention of curricula or educational standards. The text is more of a personal commentary than an educational resource. It doesn't provide any structured information or key concepts relevant to school subjects. The user might be looking for something more factual and organized. So, it doesn't meet the criteria for even the first point. The score should be low.\n</think>\n\nThe extract is a personal commentary on Italian politics, filled with opinions, emotional language, and informal tone. It lacks structured educational content, coherent explanations, or alignment with school curricula. While it touches on political themes, it does not provide factual information, organized concepts, or pedagogical value suitable for primary/grade school education. No points are earned for relevance, coherence, or educational utility.  \n\nEducational score: 0"
```
</details>