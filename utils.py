from transformers import GPT2Tokenizer


def load_tokenizer(tokenizer_data_path):
    tokenizer = GPT2Tokenizer.from_pretrained(tokenizer_data_path)
    tokenizer.add_special_tokens({
        "eos_token": "</s>",
        "bos_token": "<s>",
        "unk_token": "<unk>",
        "pad_token": "<pad>",
        "mask_token": "<mask>"
    })
    return tokenizer
