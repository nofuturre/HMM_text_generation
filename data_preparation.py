import PyPDF2
import re
import numpy as np
import pandas as pd


def read_pdf_file(name):
    data = []
    file = PyPDF2.PdfReader(name)
    for page in file.pages:
        text = page.extractText()
        text = tokenize(text)
        for word in text:
            data.append(word)

    return data


def tokenize(data):
    pattern = re.compile(r'[A-Za-z]+[\w^\']*|[\w^\']*[A-Za-z]+[\w^\']*')
    return pattern.findall(data)


def normalize(data):
    data = [word.lower() for word in data]
    return data, mapping(data)


def mapping(data):
    return dict((word, i) for i, word in enumerate(set(data)))


def tokens_ids(data):
    return dict((i, word.lower()) for i, word in enumerate(set(data)))


def first_words_dist(data, tokens):
    first_words = []
    first_words_cnt = np.zeros(len(tokens))
    for word in data:
        if word[0].isupper():
            first_words.append(word)

    for word in first_words:
        first_words_cnt[tokens[word]] += 1

    first_words_cnt = np.divide(first_words_cnt, len(first_words))

    return dict((d, i) for i, d in enumerate(first_words_cnt))


def transitions(data, tokens, k):
    matrix = np.zeros((len(tokens), len(tokens)))
    for i in range(len(data) - 1):
        for j in range(1, k + 1):
            try:
                matrix[tokens[data[i]]][tokens[data[i + j]]] += 1
            except IndexError:
                pass

    df = pd.DataFrame(matrix)
    df = df.div(df.sum(axis=1), axis=0)
    df.replace(np.NAN, 0, inplace=True)

    return df




