from BPE_token import BPE_token
from pathlib import Path


PATH = 'dataset'

# the folder 'text' contains all the files
paths = [str(x) for x in Path(f'./{PATH}/').glob("**/*.txt")]
tokenizer = BPE_token()
print(paths)
# train the tokenizer model
tokenizer.bpe_train(paths)

# saving the tokenized data in our specified folder
save_path = 'tokenized_data'
tokenizer.save_tokenizer(save_path)
