import os

import tensorflow as tf
from decouple import config
from transformers import GPT2Tokenizer, GPT2Config, TFGPT2LMHeadModel, \
  WEIGHTS_NAME, CONFIG_NAME

from utils import load_tokenizer


TOKENIZER_DATA_PATH = config('TOKENIZER_DATA_PATH')
TRAIN_DATASET_PATH = config('TRAIN_DATASET_PATH')
TRAINED_TOKENIZER_PATH = config('TRAINED_TOKENIZER_PATH')
MODEL_PATH = config('MODEL_PATH')
INPUT_MODEL_PATH = config('INPUT_MODEL_PATH')
OUTPUT_MODEL_PATH = config('OUTPUT_MODEL_PATH')
EPOCHS = config('EPOCHS', cast=int)  # 1


retrain = False
if retrain:
    tokenizer = GPT2Tokenizer.from_pretrained(TRAINED_TOKENIZER_PATH)
else:
    tokenizer = load_tokenizer(TOKENIZER_DATA_PATH)


dataset = tf.data.Dataset.load(TRAIN_DATASET_PATH)

# creating the configurations from which the model can be made
gpt2_config = GPT2Config(
  vocab_size=tokenizer.vocab_size,
  bos_token_id=tokenizer.bos_token_id,
  eos_token_id=tokenizer.eos_token_id
)

# creating/loading the model

if retrain:
    model = TFGPT2LMHeadModel.from_pretrained(INPUT_MODEL_PATH)
else:
    model = TFGPT2LMHeadModel(gpt2_config)
    model.save(MODEL_PATH)

# defining our optimizer
optimizer = tf.keras.optimizers.Adam(
    learning_rate=3e-5,
    epsilon=1e-08,
    clipnorm=1.0
)

# definining our loss function
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
# defining our metric which we want to observe

metric = tf.keras.metrics.SparseCategoricalAccuracy('accuracy')

# compiling the model
model.compile(
    optimizer=optimizer,
    loss=[loss, *[None] * model.config.n_layer],
    metrics=[metric]
)

# model.compile(
#     optimizer=optimizer,
#     loss=[loss, *[None] * model.config.n_layer],
#     metrics=[metric],
#     run_eagerly=True
# )

history = model.fit(dataset, epochs=EPOCHS)

model_to_save = model.module if hasattr(model, 'module') else model
output_model_file = os.path.join(OUTPUT_MODEL_PATH, WEIGHTS_NAME)
output_config_file = os.path.join(OUTPUT_MODEL_PATH, CONFIG_NAME)

# save model and model configs
model.save_pretrained(OUTPUT_MODEL_PATH)
model_to_save.config.to_json_file("trained_config")

# save tokenizer
tokenizer.save_pretrained(TRAINED_TOKENIZER_PATH)
