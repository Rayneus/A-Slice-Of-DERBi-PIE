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

def count(dataset):
    # Flatten the list of input_ids
    all_input_ids = [word for example in dataset for word in example['input_ids']]
    
    # Get unique words
    unique_words = set(all_input_ids)
    
    # Count the occurrences of each unique word
    word_count = {word: 0 for word in unique_words}
    for word in all_input_ids:
        word_count[word] += 1
    
    return word_count

tokenized_datasets = load_process()
word_count = count(tokenized_datasets["test"])
print(word_count)




