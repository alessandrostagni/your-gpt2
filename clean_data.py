import os
import string

from decouple import config

from pathlib import Path


DATASET_PATH = config('DATASET_PATH')
CLEAN_DATASET_PATH = config('CLEAN_DATASET_PATH')

accepted_chars = list(string.ascii_letters) + list(string.digits) + [' ']

# the folder 'text' contains all the files
# TODO: Avoid loading files in memory
paths = [str(x) for x in Path(f'./{DATASET_PATH}/').glob("**/*")]

for filename in os.listdir(DATASET_PATH):
    cleaned_data = ''
    with open(f'{DATASET_PATH}/{filename}', "r", encoding='utf-8') as f:
        x = f.read()
        for c in x:
            if c == '\n':
                cleaned_data += ' '
            if c in accepted_chars:
                cleaned_data += c.lower()
    with open(f'{CLEAN_DATASET_PATH}/{filename}', 'w') as w:
        w.write(cleaned_data)
