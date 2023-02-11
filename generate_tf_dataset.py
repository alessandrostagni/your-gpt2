import os
import time

import tensorflow as tf
from decouple import config

from utils import load_tokenizer


CLEAN_DATASET_PATH = config('CLEAN_DATASET_PATH')
TRAIN_DATASET_PATH = config('TRAIN_DATASET_PATH')
TOKENIZER_DATA_PATH = config('TOKENIZER_DATA_PATH')

BLOCK_SIZE = config('BLOCK_SIZE', cast=int)  # 100
BATCH_SIZE = config('BATCH_SIZE', cast=int)  # 12
BUFFER_SIZE = config('BUFFER_SIZE', cast=int)  # 1000

# loading tokenizer from the saved model path
tokenizer = load_tokenizer(TOKENIZER_DATA_PATH)

# TODO: Avoid loading the whole text in memory
single_string = ''
for filename in os.listdir(CLEAN_DATASET_PATH):
    with open(f'{CLEAN_DATASET_PATH}/{filename}', "r", encoding='utf-8') as f:
        x = f.read()
    single_string += x + tokenizer.eos_token

print('Tokenizing dataset...')
start_time = time.time()
string_tokenized = tokenizer.encode(single_string)
tokenization_time = execution_time = time.time() - start_time
print("Finished in:", execution_time)

examples = []

for i in range(0, len(string_tokenized) - BLOCK_SIZE + 1, BLOCK_SIZE):
    examples.append(string_tokenized[i:i + BLOCK_SIZE])
inputs, labels = [], []

for ex in examples:
    inputs.append(ex[:-1])
    labels.append(ex[1:])

dataset = tf.data.Dataset.from_tensor_slices((inputs, labels))
dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)
dataset.save(TRAIN_DATASET_PATH)
