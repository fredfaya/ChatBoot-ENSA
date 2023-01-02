import nltk
import re
import string
from nltk.corpus import stopwords
import spacy

stopwords_fr = set(stopwords.words('french'))
punctuations = string.punctuation
nlp_fr = spacy.load('fr_core_news_md')


def remove_things(text):
    text = str(text).lower()

    text = re.sub(r'https?:\/\/\w*\.\w*[\/\w*]*', '', text)

    text = re.sub(r'\w*[~`+@#$%*()_\^&–+={}>/|\\\[\]”‘<]+\w*', ' ', text)
    text = re.sub(r'\w*\'', '', text)
    return text


def tokenize_text(text):
    return [token.lemma_ for token in nlp_fr(text)]


def remove_stops(text_tokens):
    tokens_clean = []

    for word in text_tokens:
        if word not in stopwords_fr and word not in punctuations:
            out = ''.join([i for i in word if i not in punctuations])
            tokens_clean.append(out)

    return tokens_clean


def preprocess(text):
    output1 = remove_things(text)
    output2 = tokenize_text(output1)
    output = remove_stops(output2)

    return " ".join(output)

