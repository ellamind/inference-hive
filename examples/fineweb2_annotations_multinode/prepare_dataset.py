import datasets as hfds


# load a sample of 1M examples from the first file of the German subset
limit = 1_000_000
ds = hfds.load_dataset("HuggingFaceFW/fineweb-2", split="train", data_files="data/deu_Latn/train/000_00000.parquet")
ds = ds.select(range(limit))

print(ds)
# Dataset({
#     features: ['text', 'id', 'dump', 'url', 'date', 'file_path', 'language', 'language_score', 'language_script', 'minhash_cluster_size', 'top_langs'],
#     num_rows: 1000000
# })

# We only need the id and text columns
ds = ds.select_columns(["id", "text"])

# define the prompt
prompt = """Below is an extract from a web page. Evaluate whether the page has a high educational value and could be useful in an educational setting for teaching from primary school to grade school levels using the additive 5-point scoring system described below. Points are accumulated based on the satisfaction of each criterion:

- Add 1 point if the extract provides some basic information relevant to educational topics, even if it includes some irrelevant or non-academic content like advertisements and promotional material.
- Add another point if the extract addresses certain elements pertinent to education but does not align closely with educational standards. It might mix educational content with non-educational material, offering a superficial overview of potentially useful topics, or presenting information in a disorganized manner and incoherent writing style.
- Award a third point if the extract is appropriate for educational use and introduces key concepts relevant to school curricula. It is coherent though it may not be comprehensive or could include some extraneous information. It may resemble an introductory section of a textbook or a basic tutorial that is suitable for learning but has notable limitations like treating concepts that are too complex for grade school students. 
- Grant a fourth point if the extract highly relevant and beneficial for educational purposes for a level not higher than grade school, exhibiting a clear and consistent writing style. It could be similar to a chapter from a textbook or a tutorial, offering substantial educational content, including exercises and solutions, with minimal irrelevant information, and the concepts aren't too advanced for grade school students. The content is coherent, focused, and valuable for structured learning.
- Bestow a fifth point if the extract is outstanding in its educational value, perfectly suited for teaching either at primary school or grade school. It follows detailed reasoning, the writing style is easy to follow and offers profound and thorough insights into the subject matter, devoid of any non-educational or complex content.

The extract:
{document}.

After examining the extract: 
- Briefly justify your total score, up to 100 words.
- Conclude with the score using the format: "Educational score:  <total points>"
"""
# original prompt from fineweb-edu (https://huggingface.co/HuggingFaceFW/fineweb-edu-classifier/blob/main/utils/prompt.txt)


# create a column of conversations for chat-completion
def create_conversation(example):
    truncated_text = example["text"][:30000] # truncate to 30000 characters
    return {
        "conversation": [{"role": "user", "content": prompt.format(document=truncated_text)}]
    }
ds = ds.map(create_conversation, keep_in_memory=True, desc="Creating conversations")

# save the dataset
ds.save_to_disk("fineweb2-deu_Latn-1M-chat-completion")






