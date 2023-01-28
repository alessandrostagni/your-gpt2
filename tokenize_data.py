from decouple import config

from BPE_token import BPE_token
from pathlib import Path


CLEAN_DATASET_PATH = config('CLEAN_DATASET_PATH')
TOKENIZER_DATA_PATH = config('TOKENIZER_DATA_PATH')

# the folder 'text' contains all the files
# TODO: Avoid loading files in memory
paths = [str(x) for x in Path(f'./{CLEAN_DATASET_PATH}/').glob("**/*")]
tokenizer = BPE_token()

# train the tokenizer model
tokenizer.bpe_train(paths)

# saving the tokenized data in our specified folder
tokenizer.save_tokenizer(TOKENIZER_DATA_PATH)
