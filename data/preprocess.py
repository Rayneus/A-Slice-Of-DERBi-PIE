from datasets import load_dataset
from transformers import GPT2Tokenizer
import numpy as np

#dataset tokenization
def load_process():
    #load dataset
    # dataset = load_dataset("wikitext", "wikitext-103-v1")
    dataset = load_dataset("imdb")
    # dataset = dataset['train'].shuffle(seed=42).select([i for i in range(100)])

    # Load the tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    def tokenize(examples):
        input_ids = tokenizer(examples["text"], truncation=True, padding="max_length", max_length=512)

        input_ids["labels"] = input_ids["input_ids"].copy()

        return input_ids
    

    tokenizedDatasets = dataset.map(tokenize, batched=True, remove_columns=["text"])
    
    word_count = count(tokenizedDatasets["train"])
    # print(word_count)

    # Apply reduction to test set
    reduced_test_set = reduce_corpus(tokenizedDatasets["train"], word_count)
    tokenizedDatasets["train"] = reduced_test_set
    print(tokenizedDatasets.keys())

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

# Corpus reduction function to remove X% of occurrences
def reduce_corpus(dataset, word_count, reduction_percentage=0.25):
    target_reduction = {word: int(count * reduction_percentage) for word, count in word_count.items()}
    
    # Iterate through each example in the dataset
    reduced_dataset = []
    for example in dataset:
        new_input_ids = []
        for word in example['input_ids']:
            # Check if we still need to remove more occurrences of this word
            if target_reduction[word] > 0:
                # With probability reduction_percentage, skip adding the word
                if np.random.random() < reduction_percentage:
                    target_reduction[word] -= 1
                    continue
            new_input_ids.append(word)
        
        # Replace the input_ids in the example with the reduced version
        reduced_example = example.copy()
        reduced_example['input_ids'] = new_input_ids
        reduced_dataset.append(reduced_example)
    
    return reduced_dataset

# tokenized_datasets = load_process()
# word_count = count(tokenized_datasets)
# print(word_count)

# # Apply reduction to test set
# reduced_test_set = reduce_corpus(tokenized_datasets, word_count)
# word_count = count(reduced_test_set)
# print(word_count)



