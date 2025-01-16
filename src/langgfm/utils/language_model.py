from transformers import AutoTokenizer
import os


# Multiprocessing-based token counting
def count_tokens_batch(batch, model_name="./models/Llama-3.1-8B-Instruct"):
    # Load tokenizer (replace with your model path)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return [len(tokenizer.encode(text, add_special_tokens=False)) for text in batch]


if __name__ == "__main__":
    # Example usage
    print(os.getcwd())
    batch = [
        "This is a test sentence.",
        "This is another test sentence.",
    ]
    print(count_tokens_batch(batch))
