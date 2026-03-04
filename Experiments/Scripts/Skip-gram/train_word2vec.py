from gensim.models import Word2Vec
from tqdm import tqdm
import logging

# use preprocessed ENCOW version where my set of compounds is replaced by underscore version
available_file_numbers = ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12", "13", "14", "15"]
file_path_start = "/path/to/data/encow/processed/processed-"
file_path_end = ".txt"

# load all sentences I want to train the model on
all_sentences = [] # training corpus must be a sequence of sentences, and each sentence must be a list of string tokens -> nested lists
print("Load sentences from splits:")
for file_num in available_file_numbers:
    print("Processing slice ",file_num,"/",str(len(available_file_numbers)),":")
    with open(file_path_start+file_num+file_path_end, "r") as infile:
        sentences = infile.readlines()
    for sentence in tqdm(sentences):
        all_sentences += [sentence.split(" ")]

# do the actual training, use params from Sinan
print("Train model:")
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO) # want some training looging -> inspired by https://stackoverflow.com/questions/77096387/how-to-get-a-progess-bar-for-gensim-models-fasttext-train
model = Word2Vec(
    sentences=all_sentences,
    vector_size=300, # dimensionality of vectors
    min_count=5, # ignore words with frequency lower than this
    window=20, # max distance between current and predicted word within a sentence
    negative=15, # negative sampling -> how many "noise words" should be drawn
    sample=1e-5, # randomly downsample higher-frequency words with this threshold
    sg=1, # 1 for skip-gram, otherwise CBOW
    workers=20) # multiple worker threads for faster training

print("Save model to file:")
model.save("/path/to/repo/Experiments/Scripts/Skip-gram/checkpoint.model")

print("Done training the model")