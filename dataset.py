import re
import os
from collections import deque
from urllib import request
import tarfile

import spacy
import numpy as np


def download_dataset(dataset_path):

    """Downloads, extracts and stores the dataset at the given path"""

    url = 'http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz'
    filename = 'dataset.tar.gz'

    request.urlretrieve(url, filename)

    archive = tarfile.open(filename, 'r:gz')
    archive.extractall()
    top_dir = archive.getnames()[0]
    archive.close()

    os.rename(top_dir, dataset_path)


def preprocess_file(input_file, output_file):

    """Cleans up file contents and only keeps relevant words (that have a vector representations in our model)"""

    with open(input_file, 'r', encoding='utf-8') as f:

        content = f.read()  # get review content

        content = content.lower().replace("<br />", " ")
        content = re.sub(_strip_special_chars, "", content)  # only keep relevant characters

        processed = [t.text for t in nlp(content) if t.has_vector]  # TODO try using lemmas instead of text?

        string = ' '.join(x for x in processed)  # save result as a string containing all relevant words

        output_file.write(string + '\n')


def preprocess_data_subset(input_directory_path, output_file_path):

    """Cleans up each review, saves results as lines in a single text file"""

    file_names = [input_directory_path + '/' + name for name in os.listdir(input_directory_path)]

    with open(output_file_path, 'w', encoding='utf-8') as outfile:
        for infile in file_names:
            preprocess_file(infile, outfile)


def embedding_vector_generator(file):

    """Generates a batch of word vectors from the dataset using lazy evaluation"""

    buffer = deque()

    while True:

        if len(buffer) is 0:

            line = file.readline()  # read the next line in the file as a string

            if len(line) is 0:
                return
            else:
                # fill the buffer with the vector representations of read words
                buffer = deque([t.vector for t in nlp(line) if t.has_vector])  # TODO compare speed with buffer.append

        yield buffer.popleft()


def data_batch_generator(positive_file, negative_file, max_seq_length, batch_size=24):

    """Generates training batches using lazy evaluation"""

    # initialize a word vector generator for both files

    positive_vectors_generator = embedding_vector_generator(positive_file)
    negative_vectors_generator = embedding_vector_generator(negative_file)

    while True:

        labels = []  # stores the label for each row
        output = np.zeros([batch_size, max_seq_length, word_vector_size])  # initialise output tensor with 0

        for i in range(batch_size):

            if i % 2 is 0:

                # generate a positive example

                for j in range(max_seq_length):

                    try:
                        v = next(positive_vectors_generator)
                    except StopIteration:
                        return

                    output[i][j] = np.asarray(v)

                labels.append([1, 0])

            else:

                # generate a negative example

                for j in range(max_seq_length):

                    try:
                        v = next(negative_vectors_generator)
                    except StopIteration:
                        return

                    output[i][j] = np.asarray(v)

                labels.append([0, 1])

        yield output, labels


def preprocess():

    # if the dataset has not been downloaded attempt to download it

    if not os.path.isdir(dataset_path):
        print("Dataset not found, attempting download...")
        try:
            download_dataset(dataset_path)
        except Exception as e:
            print(e)
            print("Failed to download dataset. Exiting.")
            return

    # if the desired folder structure is not present create it

    if not os.path.exists(processed_training_set):
        os.makedirs(processed_training_set)
    if not os.path.exists(processed_test_set):
        os.makedirs(processed_test_set)

    # preprocess the dataset

    preprocess_data_subset(training_set + '/pos', processed_training_set + 'positive.txt')
    preprocess_data_subset(training_set + '/neg', processed_training_set + 'negative.txt')
    preprocess_data_subset(test_set + '/pos', processed_test_set + 'positive.txt')
    preprocess_data_subset(test_set + '/neg', processed_test_set + 'negative.txt')


_strip_special_chars = re.compile("[^A-Za-z0-9 ]+")  # characters matching the pattern will be ignored

nlp = spacy.load('en')  # TODO try other vector representations

# get the number of dimmensions for a vector in the current model
word_vector_size = nlp.vocab.__getitem__("this").vector.shape[0]

dataset_path = 'stanford_movie_review_dataset'

training_set = dataset_path + '/train'
test_set = dataset_path + 'test'

processed_training_set = 'processed_dataset/train/'
processed_test_set = 'processed_dataset/test/'

positive_training_file = processed_training_set + 'positive.txt'
negative_training_file = processed_training_set + 'negative.txt'

positive_test_file = processed_training_set + 'positive.txt'
negative_test_file = processed_training_set + 'negative.txt'
