# FineWeb-2 Annotations
This example demonstrates how to annotate a sample of FineWeb-2 documents with edu-scores using `inference-hive`.


## Cloning `inference-hive`
```bash
git clone git@github.com:ellamind/inference-hive.git
cd inference-hive # make sure to be in the inference-hive root dir.
```

## Preparing a sample from FineWeb-2
`examples/fineweb2_annotations/prepare_dataset.py` contains example code for preparing a sample of FineWeb-2.
Specifically, it performs the following steps:
1. Obtain a sample of FineWeb-2 documents by instantiating a huggingface dataset object via `load_dataset`.
Here we select one file of the German (deu_Latn) subset of FineWeb-2, of which we take the first 1M documents, truncated to the first 20k characters.

2. Define the prompt.
We use the prompt from the original [fineweb-edu-classifier](https://huggingface.co/HuggingFaceFW/fineweb-edu-classifier/blob/main/utils/prompt.txt).

3. Create a column of conversations.
We apply the prompt to each sample in the dataset. First, we truncate each document to 30k characters. Then we create message objects (conversations) in the format `[{"role": "user", "content": prompt}]` where `prompt` is the original fineweb-edu prompt formatted with the document.

4. Save the dataset
We save the dataset as `fineweb2-deu_Latn-1M-chat-completion` using `save_to_disk`.

To prepare the dataset, run:
```bash
python examples/fineweb2_annotations/prepare_dataset.py
```
The out is is a new directory `fineweb2-deu_Latn-1M-chat-completion` that contains the prepared dataset.

## Create a Config File
Make a copy of `ih_config_template.yaml`:
```bash
cp ih_config_template.yaml ih_config_fw2_annotations.yaml
```
Then fill the config file. We provide an example here, `examples/fineweb2_annotations/ih_config_fw2_annotations.yaml`, however, the *SLURM Configuration* section will likely differ for you. Also make sure to edit the dataset and output paths. In this example config, we set the number of inference servers to 2.

## Download the model
Download the model by running `hf download "Qwen/Qwen3-4B"`

## Validating
We validate the config and data loading:

```bash
python validate_config.py --config config_fw2_annotations.yaml
```
<details><summary>Show output</summary>

```
2025-09-17 14:49:33.963 | INFO     | __main__:main:21 - Validating: examples/fineweb2_annotations/config_fw2_annotations.yaml
2025-09-17 14:49:33.969 | INFO     | __main__:main:27 - ✓ Valid for create_slurm_script.py
2025-09-17 14:49:33.972 | INFO     | __main__:main:36 - ✓ Valid for run_inference.py
2025-09-17 14:49:33.972 | INFO     | __main__:main:42 - Configuration is valid!
```

</details>

```bash
python validate_data.py --config config_fw2_annotations.yaml
```

<details><summary>Show output</summary>

```
2025-09-17 14:49:45.133 | INFO     | __main__:validate_dataset_from_config:120 - Loading configuration from: examples/fineweb2_annotations/config_fw2_annotations.yaml
2025-09-17 14:49:45.143 | INFO     | __main__:validate_dataset_from_config:123 - Loading dataset for validation...
2025-09-17 14:49:45.143 | INFO     | __main__:validate_dataset_from_config:124 - Config: api_base_url='http://localhost:64776/v1' api_type='chat-completion' model='Qwen/Qwen3-4B' dataset_type='hf-disk' dataset_path=PosixPath('/leonardo_work/AIFAC_L01_028/midahl00/inference-hive/fineweb2-deu_
Latn-1M-chat-completion') input_column_name='conversation' id_column_name='id' output_path=PosixPath('/leonardo_work/AIFAC_L01_028/midahl00/inference-hive/example_outputs/fineweb2-deu_Latn-1M-responses-qwen3-4b') dataset_kwargs={} completions_kwargs=None apply_udf=None apply_udf_kwargs={} ma
x_connections=100 max_retries=3
2025-09-17 14:49:45.143 | INFO     | data_utils:load_data:327 - Loading dataset with with kwargs: {}
2025-09-17 14:49:45.143 | INFO     | data_utils:load_data:333 - Loading dataset with load_from_disk
Loading dataset from disk: 100%|███████████████████████████████████████████████████████████████████████████████████| 17/17 [00:00<00:00, 18.67it/s]
2025-09-17 14:49:46.118 | INFO     | __main__:validate_dataset_from_config:126 - Dataset shard=None loaded: 1000000 rows
2025-09-17 14:49:46.118 | INFO     | __main__:validate_dataset_from_config:127 - Dataset({
    features: ['id', 'text', 'conversation'],
    num_rows: 1000000
})
2025-09-17 14:49:46.118 | INFO     | __main__:validate_dataset_from_config:130 - Starting data validation...
2025-09-17 14:49:46.224 | INFO     | __main__:validate_input_data_format:82 - Input data format validation passed for api_type='chat-completion' with string ID column 'id' using OpenAI's pydantic models
2025-09-17 14:49:46.224 | INFO     | __main__:validate_input_data_format:85 - Sample rows:
2025-09-17 14:49:46.224 | INFO     | __main__:validate_input_data_format:86 - ================================================================================
2025-09-17 14:49:46.225 | INFO     | __main__:validate_input_data_format:92 - ----------------------------------------
2025-09-17 14:49:46.225 | INFO     | __main__:validate_input_data_format:93 - Sample 1/10
2025-09-17 14:49:46.225 | INFO     | __main__:validate_input_data_format:94 - row_id='<urn:uuid:9ac40fde-7881-4472-b755-7eb74eb47ddd>'
2025-09-17 14:49:46.225 | INFO     | __main__:validate_input_data_format:102 - Messages:
2025-09-17 14:49:46.225 | INFO     | __main__:validate_input_data_format:106 - [1]
role='user'
content='Below is an extract from a web page. Evaluate whether the page has a high educational value and could be useful in an educational setting for teaching from primary school to grade school levels using the additive 5-point scoring system described below. Points are accumulated based o
n the satisfaction of each criterion:\n\n- Add 1 point if the extract provides some basic information relevant to educational topics, even if it includes some irrelevant or non-academic content like advertisements and promotional material.\n- Add another point if the extract addresses certai
n elements pertinent to education but does not align closely with educational standards. It might mix educational content with non-educational material, offering a superficial overview of potentially useful topics, or presenting information in a disorganized manner and incoherent writing sty
le.\n- Award a third point if the extract is appropriate for educational use and introduces key concepts relevant to school curricula. It is coherent though it may not be comprehensive or could include some extraneous information. It may resemble an introductory section of a textbook or a ba
sic tutorial that is suitable for learning but has notable limitations like treating concepts that are too complex for grade school students. \n- Grant a fourth point if the extract highly relevant and beneficial for educational purposes for a level not higher than grade school, exhibiting a
 clear and consistent writing style. It could be similar to a chapter from a textbook or a tutorial, offering substantial educational content, including exercises and solutions, with minimal irrelevant information, and the concepts aren\'t too advanced for grade school students. The content
is coherent, focused, and valuable for structured learning.\n- Bestow a fifth point if the extract is outstanding in its educational value, perfectly suited for teaching either at primary school or grade school. It follows detailed reasoning, the writing style is easy to follow and offers pr
ofound and thorough insights into the subject matter, devoid of any non-educational or complex content.\n\nThe extract:\n\n\n23.06.02\n| Out of the Blue - Explodierende Träume\n\n\nCa 1980 (16mm, 93min)\n| Übersicht\n\n\n(zurück zur Übersicht)\nIMDb :\n(mehr externe Filminfos)\n|R: Dennis Ho
pper|\nB: Gary Jules Jouvenat, Brenda Niel\nD: Linda Manz, Dennis Hopper, Sharon Farrell, Don Gordon, Raymond Burr\nMy my, hey hey / Rock and roll is here to stay ... CeBe ist 14, als ihr alter Herr wieder aus dem Gefängnis rauskommt. Er hat vor 5 Jahren angesoffen und mit CeBe auf dem Beifa
hrersitz seinen Laster in einen Schulbus gefahren; dabei sind ein paar Kinder umgekommen. Irgendwie freut CeBe sich auf ihren Vater, irgendwie auch nicht. Ihre Mutter geht gar nicht, Hippie und Junkie, arbeitet als Bedienung und macht mit zwei Typen gleichzeitig rum, labert Scheiße in einer
Tour: wenn Frauwerden so aussieht, dann lieber nicht. CeBe zieht sich solche Klamotten gar nicht erst an: Haare schön kurz und Elvis auf dem Kassettenrekorder hören. Elvis ist geil, Elvis ist the King. Aber jetzt ist Elvis tot. It\'s better to burn out / Than to fade away / My my, hey hey. /
 Out of the blue / and into the black / They give you this, but you pay for that / And once you\'re gone, / you can never come back / When you\'re out of the blue / and into the black. "Wenn Dein Vater wieder zuhause ist, werden wir wieder eine glückliche kleine Familie", meint die Alte. Abe
r kaum ist er da, wird es nur noch schlimmer; seine Vergangenheit sucht ihn heim, nichts glückt, er säuft immer mehr, ständig ist Krach. CeBe hat nur noch drei Rückzugsgebiete: ihr Zimmer, ihre Elviskassetten und einen Punkschuppen in Vancouver. Die Alte hat Angst, daß CeBe eine Lesbe werden
 könnte; Vattern nimmt das in die Hand, besoffen zusammen mit ein paar ebenfalls besoffenen Kumpanen. Und jetzt ist selbst CeBes Zimmer keine sichere Burg mehr... Aber CeBe ist kein "Mädchen", sie kann sich wehren und ihr Motto ist schon lange: "Kill all hippies"....\n\nAfter examining the e
xtract: \n- Briefly justify your total score, up to 100 words.\n- Conclude with the score using the format: "Educational score:  <total points>"\n'
2025-09-17 14:49:46.225 | INFO     | __main__:validate_input_data_format:92 - ----------------------------------------
2025-09-17 14:49:46.225 | INFO     | __main__:validate_input_data_format:93 - Sample 2/10
...(additional samples omitted here)
2025-09-17 14:49:46.226 | INFO     | __main__:validate_input_data_format:108 - ================================================================================
2025-09-17 14:49:46.226 | INFO     | __main__:validate_dataset_from_config:150 - ✓ Data validation completed successfully!
2025-09-17 14:49:46.232 | INFO     | __main__:main:194 - Data validation passed! Dataset is ready for inference.
```

</details>


## Create a run
Next, use `create_run.py` to create a run for the `ih_config_fw2_annotations.yaml` config and with `fw2_annotations_run1` as the run dir.
```bash
python create_run.py --config ih_config_fw2_annotations.yaml --output fw2_annotations_run1
```
<details><summary>Show output</summary>

```
2025-09-17 14:56:48.946 | INFO     | __main__:main:353 - Output directory: fw2_annotations_run1
2025-09-17 14:56:48.950 | INFO     | __main__:main:391 - Config copied to: fw2_annotations_run1/ih_config.yaml
2025-09-17 14:56:48.951 | INFO     | __main__:main:412 - SLURM job script generated successfully: fw2_annotations_run1/ih_job.slurm
2025-09-17 14:56:48.951 | INFO     | __main__:main:417 - Run created at: fw2_annotations_run1
```

</details>

## Submitting jobs
To submit jobs for the run, use

```bash
python submit.py --run-dir fw2_annotations_run1
```
to submit the jobs for the run.

<details><summary>Show output</summary>

```
2025-09-17 15:12:52.611 | INFO     | __main__:main:45 - No existing jobs found for fw2-annotations-deu-1M
2025-09-17 15:12:52.611 | INFO     | __main__:main:50 - Jobs to submit: 2
2025-09-17 15:12:52.611 | INFO     | __main__:main:53 - Submitting 2 jobs...
2025-09-17 15:12:52.611 | INFO     | __main__:main:64 -   1/2: Submitting job for shard 0...
2025-09-17 15:12:52.902 | INFO     | __main__:main:76 -     Submitted batch job 20254160 for shard 0
2025-09-17 15:12:52.903 | INFO     | __main__:main:64 -   2/2: Submitting job for shard 1...
2025-09-17 15:12:53.167 | INFO     | __main__:main:76 -     Submitted batch job 20254161 for shard 1
2025-09-17 15:12:53.167 | INFO     | __main__:main:77 - Done.
```

</details>
<br>

Log files will be written to `fw2_annotations_run1/logs` once they start running.

You can check the status of the run using 
```bash
python status.py --run-dir fw2_annotations_run1
# or for more details, including progress and throughput stats
python status.py --run-dir fw2_annotations_run1 --detailed
```

<details><summary>Show output</summary>

```
2025-09-17 15:31:13.484 | INFO     | __main__:main:347 - Found 2 existing jobs for fw2-annotations-deu-1M
2025-09-17 15:31:13.488 | INFO     | __main__:main:349 -     2/2 RUNNING

|   Shard | Completed   | State       |    JobID |
|---------|-------------|-------------|----------|
|       0 | False       | ['RUNNING'] | 20254160 |
|       1 | False       | ['RUNNING'] | 20254161 |

2025-09-17 15:31:13.492 | INFO     | __main__:main:379 - Using cutoff based on system time: cutoff=2025-09-17 15:26:13, reference=2025-09-17 15:31:13, window=5.0 min
2025-09-17 15:31:13.495 | INFO     | __main__:main:387 - Found 2 progress files
2025-09-17 15:31:13.495 | INFO     | __main__:main:392 - Loading 2 progress files
2025-09-17 15:31:13.692 | INFO     | __main__:main:394 - Total data points: 30

PER-SHARD STATISTICS (2/2 shards active)
==========================================================================================
+---------+-------------+--------------+-----------+----------+-------+-------------+--------------+-------------+---------------------+-------+
| Shard   | Completed   | Progress %   |   Entries | Status   |   RPS |   Total TPS |   Prompt TPS |   Compl TPS | Last Update         | ETA   |
+=========+=============+==============+===========+==========+=======+=============+==============+=============+=====================+=======+
| Shard 0 | False       | 0.6%         |        15 | active   |   3.4 |      6769.4 |       3993.3 |      2776.1 | 2025-09-17 13:30:55 | 47.9h |
+---------+-------------+--------------+-----------+----------+-------+-------------+--------------+-------------+---------------------+-------+
| Shard 1 | False       | 0.6%         |        15 | active   |   3.2 |      6582.6 |       3778.6 |      2803.9 | 2025-09-17 13:30:56 | 41.0h |
+---------+-------------+--------------+-----------+----------+-------+-------------+--------------+-------------+---------------------+-------+

CURRENT TOTAL THROUGHPUT (2 active shards, last 5.0 min)
============================================================
+----------------+----------+
| Metric         | Rate     |
+================+==========+
| RPS            | 6.3      |
+----------------+----------+
| Total TPS      | 13,116.6 |
+----------------+----------+
| Prompt TPS     | 7,390.7  |
+----------------+----------+
| Completion TPS | 5,725.9  |
+----------------+----------+

SUMMARY
==============================
----------------  ------------
Total shards      2
Shards completed  0
Shards w/ data    2
Active shards     2
Duration          14.0 minutes
Total completed   5,603
Total requests    1,000,000
Overall progress  0.6%
Longest ETA       47.9h
----------------  ------------
```

</details>
<br>

You can find example logs in `examples/fineweb2_annotations/example_logs`.

Lets cancel the job after some time, to not waste compute:
```bash
python cancel.py --run-dir fw2_annotations_run1
```

<details><summary>Show output</summary>

```
2025-09-17 15:35:08.459 | INFO     | __main__:main:39 - Cancelled 2 job(s): 20254160 20254161
2025-09-17 15:35:08.463 | INFO     | __main__:main:44 - Done.
```

</details>
<br>

## Using the Outputs
The outputs were saved to `example_outputs/fineweb2-deu_Latn-1M-responses-qwen3-4b`.

<details><summary>Output files</summary>

```bash
ls -1 example_outputs/fineweb2-deu_Latn-1M-responses-qwen3-4b
# shard000000_checkpoint000000.parquet
# shard000001_checkpoint000000.parquet
```
</details>

The responses are written to parquet files. Each has two columns:
1. "id", to uniquely identify the input row from your dataset
2. "response", containing the response object according to OpenAI spec.

Lets print the first response:
First, start a python shell using `python`. Then run
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
    "id": "<urn:uuid:3278a728-d95e-427b-85ea-da1fb7caa798>",
    "response": {
        "id": "chatcmpl-a2c8cf1941b745eb8839462d01e2ba18",
        "choices": [
            {
                "finish_reason": "stop",
                "index": 0,
                "logprobs": None,
                "message": {
                    "content": '<think>\nOkay, let\'s evaluate this extract. The user is asking if the web page has high educational value for teaching primary to grade school. The extract is a technical discussion about Debian package management, specifically about a problem with "texdoctk" and "tetex-base" packages.\n\nFirst, the content is technical and related to software packaging, which is way beyond primary or grade school level. The language is complex, using terms like "conffiles," "dpkg," and "maintainer-scripts." There\'s no mention of basic educational concepts or anything suitable for younger students. The text is a discussion between users, not a structured tutorial or lesson plan. It includes a bug report, which isn\'t educational for kids. The writing is coherent but not relevant to educational standards for that age group. There\'s no mention of any educational materials, exercises, or explanations tailored for students. So, it doesn\'t meet any of the criteria for points 1 to 5. The only possible point is 1 for some basic info, but even that is too advanced. Therefore, the score should be 0.\n</think>\n\nThe extract is a technical discussion about Debian package management, using complex terminology and addressing a software-related issue. It contains no basic educational content, no structured lessons, and is irrelevant to primary or grade school curricula. The writing is coherent but non-educational for the target age group. No points are earned for its relevance or suitability.  \n\nEducational score: 0',
                    "refusal": None,
                    "role": "assistant",
                    "annotations": None,
                    "audio": None,
                    "function_call": None,
                    "tool_calls": [],
                },
            }
        ],
        "created": 1758114955,
        "model": "Qwen/Qwen3-4B",
        "object": "chat.completion",
        "service_tier": None,
        "system_fingerprint": None,
        "usage": {
            "completion_tokens": 304,
            "prompt_tokens": 812,
            "total_tokens": 1116,
            "completion_tokens_details": None,
            "prompt_tokens_details": None,
        },
    },
}
```
</details>
<br>

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
```

<details><summary>Output</summary>

```
<think>
Okay, let's evaluate this extract. The user is asking if the web page has high educational value for teaching primary to grade school. The extract is a technical discussion about Debian package management, specifically about a problem with "texdoctk" and "tetex-base" packages.

First, the content is technical and related to software packaging, which is way beyond primary or grade school level. The language is complex, using terms like "conffiles," "dpkg," and "maintainer-scripts." There's no mention of basic educational concepts or anything suitable for younger students. The text is a discussion between users, not a structured tutorial or lesson plan. It includes a bug report, which isn't educational for kids. The writing is coherent but not relevant to educational standards for that age group. There's no mention of any educational materials, exercises, or explanations tailored for students. So, it doesn't meet any of the criteria for points 1 to 5. The only possible point is 1 for some basic info, but even that is too advanced. Therefore, the score should be 0.
</think>

The extract is a technical discussion about Debian package management, using complex terminology and addressing a software-related issue. It contains no basic educational content, no structured lessons, and is irrelevant to primary or grade school curricula. The writing is coherent but non-educational for the target age group. No points are earned for its relevance or suitability.

Educational score: 0
```

</details>