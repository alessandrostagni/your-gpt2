# Your GPT2
Train GPT2 on your own data.

## Requirements
Code hasn't been tested on CPU but in general models like GPT needs to be trained on GPUs.
You will need Docker installed if you want to run the in the Docker image of the code.

## Environment variables
Each of the following environment variables need to be set in in your system in order to make the code run correctly.

`DATASET_PATH`: Folder containing the raw dataset as a set of text files.
<br/>
`CLEANED_DATASET_PATH`: Folder that will contain the dataset once it's been cleaned.
<br/>
`TRAIN_DATASET_PATH`: Folder that will contain the dataset in Tensorflow format, ready to be used for training
<br/>
`TOKENIZER_DATA_PATH`: Tokenizer data once tokenization has happened.

`BLOCK_SIZE`: Size of each block (training sample).
<br/>
`BATCH_SIZE`: Batch size for the training phase.
<br/>
`BUFFER_SIZE`: How much data is buffered in memory while training.
<br/>
`EPOCHS`: How many epochs to train the model for.

`MODEL_PATH`: Path where to save the untrained, starting model.
<br/>
`INPUT_MODEL_PATH`: Needed only if you train a pretrained existing model. Path where the pretrained model data is located.
<br/>
`OUTPUT_MODEL_PATH`: Path where the trained model will be saved.


## Run in Docker image
Build the Docker image:
<br/>
`docker build -t your-gpt2 .`

Run the docker image:
<br/>
`docker run -p 8888:8888 -v /home/alessandro/Code/your-gpt2:/home/ --gpus all -it --rm your-gpt2 bash`

## Steps for running the code

1. (Optional) Use the Python notebook `analyse_data.ipynb` to analyze your dataset for noise symbols, size, etc.
2. Clean the dataset by running `python clean_data.py`
3. Tokenize the data by running `python tokenize_data.py`
4. Generate the Tensorflow dataset by running  `python generate_tf_dataset.py`
5. Train with `python train.py`
6. Chat with the model by running `python predict.py [Your message]`

## Notes
Since data quality and size were poor and based on video captionm, I removed all punctuation and capitalisation.
<br/>
You can change this in the `clean_data.py` script.
<br/>
Change the `accepted_chars` list and remove the `.lower()` function call when concatenating strings.

## References:
[GPT2 Paper](https://d4mucfpksywv.cloudfront.net/better-language-models/language-models.pdf) <br/>
[Guide](https://towardsdatascience.com/train-gpt-2-in-your-own-language-fc6ad4d60171) this code is written from.

## TODO
- [ ]Discord Bot

