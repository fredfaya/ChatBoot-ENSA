import os
from collections import Counter

import numpy
from keras.preprocessing.text import Tokenizer
from keras.utils.data_utils import pad_sequences
import text_preprocessor

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def counter_word(text_col):
    count = Counter()
    for word in text_col.values:
        count[word] += 1

    return count


class DatasetPreprocessor:
    def __init__(self, dictionnary, dataset, split_ratio = 0.9, max_length=25):
        self.counter = counter_word(dictionnary.ortho)
        self.num_unique_words = len(self.counter)
        self.max_length = max_length

        dataset = dataset.sample(frac=1).reset_index()

        train_size = int(dataset.shape[0] * split_ratio)
        train_df = dataset[:train_size]
        val_df = dataset[train_size:]

        train_df_clean = train_df.text.apply(text_preprocessor.preprocess)
        val_df_clean = val_df.text.apply(text_preprocessor.preprocess)

        self.train_sentences = train_df_clean.to_numpy()
        self.train_labels = train_df.target.to_numpy()
        self.val_sentences = val_df_clean.to_numpy()
        self.val_labels = val_df.target.to_numpy()

        self.dictionnary_words = dictionnary.ortho.astype(str)

        self.tokenizer = Tokenizer(num_words=self.num_unique_words)
        self.tokenizer.fit_on_texts(self.dictionnary_words)

        self.word_index = self.tokenizer.word_index

        train_sequences = self.encode_text(self.train_sentences)
        val_sequences = self.encode_text(self.val_sentences)

        self.train_padded = pad_sequences(train_sequences, maxlen=max_length, padding="post", truncating="post")
        self.val_padded = pad_sequences(val_sequences, maxlen=max_length, padding="post", truncating="post")

        self.reverse_word_index = dict({(idx, word) for (word, idx) in self.word_index.items()})

    def decode_text(self, sequence):
        return " ".join([self.reverse_word_index.get(idx, "?") for idx in sequence])

    def encode_text(self, text):
        return self.tokenizer.texts_to_sequences(text)

    def preprocess_text_to_predict(self, text):
        text = text_preprocessor.preprocess(text)
        text = numpy.array([text])
        text_encoded = self.encode_text(text)
        output = pad_sequences(text_encoded, maxlen=self.max_length, padding="post", truncating="post")

        return output
