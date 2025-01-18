import os


# token counting
def count_tokens_batch(batch, tokenizer):
    return [len(tokenizer.encode(text, add_special_tokens=False)) for text in batch]


if __name__ == "__main__":
    # Example usage
    print(os.getcwd())
    batch = [
        "This is a test sentence.",
        "This is another test sentence.",
    ]
    print(count_tokens_batch(batch))
