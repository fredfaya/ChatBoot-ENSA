from collections import Counter
from keras.preprocessing.text import Tokenizer
from keras.utils.data_utils import pad_sequences
import text_preprocessor


def counter_word(text_col):
    count = Counter()
    for word in text_col.values:
        count[word] += 1

    return count


class DatasetPreprocessor:
    def __init__(self, dictionnary, dataset, split_ratio, max_length=100):
        self.counter = counter_word(dictionnary.ortho)
        self.num_unique_words = len(self.counter)

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
