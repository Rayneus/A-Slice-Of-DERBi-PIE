from datasets import load_dataset
from transformers import GPT2Tokenizer

#dataset tokenization
def load_process():
    #load dataset
    dataset = load_dataset("wikitext", "wikitext-103-v1")
    n = dataset["test"].shape[0]

    # Load the tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    def tokenize(examples):
        input_ids = tokenizer(examples["text"], truncation=True, padding="max_length", max_length=512)

        input_ids["labels"] = input_ids["input_ids"].copy()

        return input_ids
    

    tokenizedDatasets = dataset.map(tokenize, batched=True, remove_columns=["text"])
    for i in range(10):
        print(tokenizedDatasets["test"][i]["input_ids"])

    return tokenizedDatasets

