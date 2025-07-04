import datasets as hfds


# load a sample of 100k examples from the first file of the 10BT sample
limit = 100_000
ds = hfds.load_dataset("HuggingFaceFW/fineweb-edu", split="train", data_files="sample/10BT/000_00000.parquet")
ds = ds.select(range(limit))

print(ds)
# Dataset({
#     features: ['text', 'id', 'dump', 'url', 'date', 'file_path', 'language', 'language_score', 'token_count', 'score', 'int_score'],
#     num_rows: 100000
# })

# We only need the id and text columns
ds = ds.select_columns(["id", "text"])


# add a column with the target language
ds_list = []
target_langs = ["German", "Spanish", "French", "Czech", "Finnish"]
for target_lang in target_langs:
    ds_target_lang = ds.add_column("target_lang", [target_lang] * len(ds))
    ds_target_lang = ds_target_lang.map(lambda row: {'id': row['id'] + f"_{target_lang}"}, keep_in_memory=True, desc=f"Adding target language {target_lang}")
    ds_list.append(ds_target_lang)
ds = hfds.concatenate_datasets(ds_list)
print(ds)
# Dataset({
#     features: ['id', 'text', 'target_lang'],
#     num_rows: 500000
# })

# define the prompts
user_prompt = """Translate the following English source text to {target_lang}:\nEnglish: {document}\n{target_lang}: """


# create a column of conversations for chat-completion
def create_conversation(example):
    truncated_text = example["text"][:30000] # truncate to 30000 characters for this example
    return {
        "conversation": [
            {"role": "user", "content": user_prompt.format(document=truncated_text, target_lang=example["target_lang"])},
        ]
    }
ds = ds.map(create_conversation, keep_in_memory=True, desc="Creating conversations")

print(f"------ {ds[0]['conversation'][0]['role']} ------")
print(ds[0]['conversation'][0]['content'])

# save the dataset
ds.save_to_disk("fineweb-edu-mt-5langs")






