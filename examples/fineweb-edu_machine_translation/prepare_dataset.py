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

# define the prompts
system_prompt = """You are an expert translator.
You can translate user provided documents into any language.
Documents are provided to you inside of <document></document> tags.
You respond with the translated document wrapped in <translated_document></translated_document> tags.
You must only respond with the translated document, nothing else.
"""

user_prompt = """Translate the following document to German.

<document>
{document}
</document>
"""

# we can provide an assistant prefill to control the output of the model.
# for this we need to set continue_final_message to True in the client requests.
assistant_prefill= "<translated_document>"


# create a column of conversations for chat-completion
def create_conversation(example):
    truncated_text = example["text"][:30000] # truncate to 30000 characters
    return {
        "conversation": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt.format(document=truncated_text)},
            {"role": "assistant", "content": assistant_prefill}
        ]
    }
ds = ds.map(create_conversation, keep_in_memory=True, desc="Creating conversations")

for i in range(3):
    print(f"------ {ds[0]['conversation'][i]['role']} ------")
    print(ds[0]['conversation'][i]['content'])

# save the dataset
ds.save_to_disk("fineweb-edu-mt-chat-completion")






