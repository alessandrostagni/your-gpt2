import os

import tensorflow as tf
from transformers import GPT2Config, TFGPT2LMHeadModel, GPT2Tokenizer, WEIGHTS_NAME, CONFIG_NAME


PATH = 'dataset'

# loading tokenizer from the saved model path
tokenizer = GPT2Tokenizer.from_pretrained('tokenizer')
tokenizer.add_special_tokens({
  "eos_token": "</s>",
  "bos_token": "<s>",
  "unk_token": "<unk>",
  "pad_token": "<pad>",
  "mask_token": "<mask>"
})
# creating the configurations from which the model can be made
config = GPT2Config(
  vocab_size=tokenizer.vocab_size,
  bos_token_id=tokenizer.bos_token_id,
  eos_token_id=tokenizer.eos_token_id
)
# creating the model
model = TFGPT2LMHeadModel(config)
model.save("model")

single_string = ''
for filename in os.listdir(PATH):
    with open(f'{PATH}/{filename}', "r", encoding='utf-8') as f:
        x = f.read()
    single_string += x + tokenizer.eos_token
string_tokenized = tokenizer.encode(single_string)
print(string_tokenized)

examples = []
# block_size = 100
block_size = 10

BATCH_SIZE = 12
BUFFER_SIZE = 1000
for i in range(0, len(string_tokenized) - block_size + 1, block_size):
    examples.append(string_tokenized[i:i + block_size])
inputs, labels = [], []

for ex in examples:
    inputs.append(ex[:-1])
    labels.append(ex[1:])

dataset = tf.data.Dataset.from_tensor_slices((inputs, labels))
dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)
dataset.save('tf_dataset/tf_dataset')

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
# model.compile(
#     optimizer=optimizer,
#     loss=[loss, *[None] * model.config.n_layer],
#     metrics=[metric]
# )

model.compile(
    optimizer=optimizer,
    loss=[loss, *[None] * model.config.n_layer],
    metrics=[metric],
    run_eagerly=True
)

num_epoch = 1
history = model.fit(dataset, epochs=num_epoch)

model_to_save = model.module if hasattr(model, 'module') else model
output_model_file = os.path.join(output_dir, WEIGHTS_NAME)
output_config_file = os.path.join(output_dir, CONFIG_NAME)
# save model and model configs
model.save_pretrained("trained_model")
model_to_save.config.to_json_file("trained_config")
# save tokenizer
tokenizer.save_pretrained("trained_tokenizer")
