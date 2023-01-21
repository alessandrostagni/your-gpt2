from transformers import GPT2Config, TFGPT2LMHeadModel, GPT2Tokenizer

tokenizer = GPT2Tokenizer.from_pretrained('trained_tokenizer')
model = TFGPT2LMHeadModel.from_pretrained('trained_model')

text = "Hi everyone "
# encoding the input text
input_ids = tokenizer.encode(text, return_tensors='tf')
# getting out output
beam_output = model.generate(
  input_ids,
  max_length=50,
  num_beams=5,
  temperature=0.7,
  no_repeat_ngram_size=2,
  num_return_sequences=5
)

print(tokenizer.decode(beam_output[0]))
