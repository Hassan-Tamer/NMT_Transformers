from datasets import load_from_disk
from transformers import AutoTokenizer
from typing import Dict, List
from functools import partial


def map_example(example: Dict[str, str], source_tokenizer, target_tokenizer) -> Dict[str, List[int]]:
    text_en = example['text_en']
    text_fr = example['text_fr']

    token_eng = target_tokenizer(text_en)
    token_fr = source_tokenizer(text_fr)

    return {'encoder_input_ids': token_fr['input_ids'], 'decoder_input_ids': token_eng['input_ids']}

def get_tokenizers():
    source_tokenizer = AutoTokenizer.from_pretrained("resources/tokenizer_fr")
    target_tokenizer = AutoTokenizer.from_pretrained("resources/tokenizer_en")
    return source_tokenizer, target_tokenizer

def get_tokenized_dataset():
    raw_text_datasets = load_from_disk("resources/parallel_en_fr_corpus")
    source_tokenizer, target_tokenizer = get_tokenizers()
    # tokenized_datasets = raw_text_datasets.map(map_example, batched=False)

    # tokenized_datasets = raw_text_datasets.map(
    #     lambda example: map_example(example, source_tokenizer, target_tokenizer), 
    #     batched=False
    # )

    tokenized_datasets = raw_text_datasets.map(
    partial(map_example, source_tokenizer=source_tokenizer, target_tokenizer=target_tokenizer), 
    batched=False
    )

    tokenized_datasets = tokenized_datasets.remove_columns(raw_text_datasets.column_names["train"])
    return tokenized_datasets


if __name__ == "__main__":
    source_tokenizer, target_tokenizer = get_tokenizers()
    tokenized_datasets = get_tokenized_dataset()

    raw_text_datasets = load_from_disk("resources/parallel_en_fr_corpus")
    # print("Summary of splits:", raw_text_datasets)
    # print("First training example:", raw_text_datasets["train"][0])

    # source_tokenizer, target_tokenizer = get_tokenizers()

    print("Vocab size of source language:", source_tokenizer.vocab_size)
    print("Vocab size of target language:", target_tokenizer.vocab_size)

    # As a demonstration, we will the following English sentence to tokens.
    example_sentence = "we have an example"
    tokenizer_output = target_tokenizer(example_sentence)
    print("\n*** Example ***")
    print("Example sentence:", example_sentence)
    print("Tokenizer output:", tokenizer_output)

    decoded_sequence = [target_tokenizer.decode(token) for token in tokenizer_output["input_ids"]]
    print("Tokens:", decoded_sequence)

    # By replacing the special character ▁ with whitespace, we can reconstruct a legibile sentence,
    # which differs from the original example by special tokens, includings <unk> tokens, and minor whitespace differences.
    reconstructed = "".join(decoded_sequence).replace("▁", " ")
    print("Reconstructed sentence", reconstructed)

    # When mapped is applied to the DatasetDict object, it will apply `map` separately to each split.
    # tokenized_datasets = raw_text_datasets.map(map_example, batched=False)
    # The `remove_columns` removes the existing text features from the new dataset, as they are no longer needed.
    # tokenized_datasets = tokenized_datasets.remove_columns(raw_text_datasets.column_names["train"])
    # Sanity checks on the new dataset
    assert set(tokenized_datasets.column_names["train"]) == {"decoder_input_ids", "encoder_input_ids"}
    assert len(tokenized_datasets["train"]) == len(raw_text_datasets["train"])
    print("First training example:", tokenized_datasets['train'][0])