from collections import Counter
from tensorflow.keras.preprocessing.text import Tokenizer

def counter_word(text_col):
    count = Counter()
    for word in text_col.values:
        count[word] += 1

    return count


class DatasetPreprocessor:
    def __init__(self, dataset, split_ratio,dictionnary):
        self.counter = counter_word(dictionnary.ortho)
        self.num_unique_words = len(self.counter)

        self.train_size=int(dataset.shape[0]*split_ratio)
        self.train_df= dataset[:self.train_size]
        self.val_df=dataset[self.train_size:]

        self.train_sentences = self.train_df.text.to_numpy()
        self.train_labels = self.train_df.target.to_numpy()
        self.val_sentences = self.val_df.text.to_numpy()
        self.val_labels = self.val_df.target.to_numpy()

        self.dictionnary_words = dictionnary.ortho.to_numpy()

        self.tokenizer= Tokenizer(num_words=self.num_unique_words)
        self.tokenizer.fit_on_text(self.dictionnary_words)

        self.word_index=self.tokenizer.word_index

        self.train_sequences = self.tokenizer.texts_to_sequences(self.train_sentences)
        self.val_sequences = self.tokenizer.texts_to_sequences(self.val_sentences)

