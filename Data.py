import PyPDF2
import re
import numpy as np
import pandas as pd


def tokenize(data):
    pattern = re.compile(r'[A-Za-z]+[\w^\']*|[\w^\']*[A-Za-z]+[\w^\']*')
    return pattern.findall(data)


def read_pdf_file(name):
    data = []
    file = PyPDF2.PdfReader(name)
    for page in file.pages:
        text = page.extractText()
        text = tokenize(text)
        for word in text:
            data.append(word)
    return data


def read_txt_file(name):
    data = []
    with open(name, 'r') as file:
        for line in file:
            line = tokenize(line)
            for word in line:
                data.append(word)
    return data


class Data:
    def __init__(self, name):
        if name.endswith('.pdf'):
            self.data = read_pdf_file(name)
        elif name.endswith('.txt'):
            self.data = read_txt_file(name)
        else:
            raise Exception("Incorrect file format. Supported file types: pdf, txt")
        self.first_words = self.first_words()
        self.normalize()
        self.tokens = self.mapping()

    def transitions(self, tokens, k):
        matrix = np.zeros((len(tokens), len(tokens)))
        for i in range(len(self.data) - 1):
            for j in range(1, k + 1):
                try:
                    matrix[tokens[self.data[i]]][tokens[self.data[i + j]]] += 1
                except IndexError:
                    pass

        df = pd.DataFrame(matrix)
        df = df.div(df.sum(axis=1), axis=0)
        df.replace(np.NAN, 0, inplace=True)

        return df.to_numpy()

    def first_words_dist(self):
        first_words_cnt = np.zeros(len(self.tokens))

        for word in self.first_words:
            first_words_cnt[self.tokens[word.lower()]] += 1

        first_words_cnt = np.divide(first_words_cnt, len(self.first_words))

        return first_words_cnt

    def tokens_ids(self):
        return dict((i, word.lower()) for i, word in enumerate(set(self.data)))

    def mapping(self):
        return dict((word, i) for i, word in enumerate(set(self.data)))

    def normalize(self):
        self.data = [word.lower() for word in self.data]

    def first_words(self):
        first_words = []
        for i in range(len(self.data)):
            if self.data[i][0].isupper():
                first_words.append(self.data[i])
        return first_words


