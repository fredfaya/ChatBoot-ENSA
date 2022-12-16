import nltk
import re
import string
# from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

stopwords_fr = set(stopwords.words('french'))
punctuations = string.punctuation
lemmatizer = WordNetLemmatizer()


def remove_things(text):
    text = str(text).lower()

    text = re.sub(r'https?:\/\/\w*\.\w*[\/\w*]*', '', text)

    text = re.sub(r'\w*[~`+@#$%*()_\^&–+={}>/|\\\[\]”‘<]+\w*', ' ', text)
    text = re.sub(r'\w*\'', '', text)
    print(text)
    return text


def tokenize_text(text):
    text_token = nltk.tokenize.word_tokenize(text)

    return text_token


def remove_stops(text_tokens):
    tokens_clean = []

    for word in text_tokens:
        if word not in stopwords_fr and word not in punctuations:
            out = ''.join([i for i in word if i not in punctuations])
            tokens_clean.append(out)

    return tokens_clean


def lemmatizing(tokens):
    lemmatized_tokens = [lemmatizer.lemmatize(word) for word in tokens]

    return lemmatized_tokens


def preprocess(text):

    output1 = remove_things(text)
    output2 = tokenize_text(output1)
    output3 = remove_stops(output2)
    output = lemmatizing(output3)

    return " ".join(output)


