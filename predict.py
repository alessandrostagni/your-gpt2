import sys

from decouple import config
from transformers import TFGPT2LMHeadModel, GPT2Tokenizer

TRAINED_TOKENIZER_PATH = config('TRAINED_TOKENIZER_PATH')
OUTPUT_MODEL_PATH = config('OUTPUT_MODEL_PATH')

tokenizer = GPT2Tokenizer.from_pretrained(TRAINED_TOKENIZER_PATH)
model = TFGPT2LMHeadModel.from_pretrained(OUTPUT_MODEL_PATH)

text = sys.argv[1]
# encoding the input text
input_ids = tokenizer.encode(text, return_tensors='tf')
# getting out output
beam_output = model.generate(
  input_ids,
  max_length=500,
  num_beams=5,
  temperature=0.7,
  no_repeat_ngram_size=2,
  num_return_sequences=5
)

print(tokenizer.decode(beam_output[0]))
