import numpy as np
import random


class Network:

    def __init__(self, data, window):
        self.window = window
        self.tokens = data.mapping()
        self.ids = data.tokens_ids()
        self.prior_prob = data.first_words_dist()
        self.transition = data.transitions(self.tokens, self.window)
        self.emission = self.transition
        self.no_hidden_states = len(self.tokens)

        assert self.transition.shape[0] == self.transition.shape[1]
        assert self.transition.shape[0] == self.prior_prob.shape[0]
        assert self.transition.shape[0] == self.no_hidden_states

    def print_(self):
        print(f"Number of unique words: {self.no_hidden_states} \n")
        print("Prior probabilities: ")
        for i in range(self.no_hidden_states):
            if self.prior_prob[i] != 0.0:
                print(f"{self.ids[i]} \t\t {self.prior_prob[i]}")
        # print(self.tokens)
        #
        # print(self.emission)

    def code_observed(self, observed):
        codes = []
        words = observed.split()
        for word in words:
            try:
                codes.append(self.tokens[word.lower()])
            except KeyError:
                pass
        return codes

    def viterbi_(self, observed, length):
        seq = self.viterbi(observed, length)
        tmp = seq.split()
        tmp.reverse()
        i = 1
        seq = []
        x = len(observed.split())

        for t in tmp:
            if i == 1:
                seq.append(t.title())
            elif i == x:
                seq.append(t + '.')
                i = 0
            else:
                seq.append(t)
            i += 1

        return ' '.join(str(s) for s in seq)

    def viterbi(self, observed, length):
        observed = self.code_observed(observed)
        no_rows, no_cols = self.no_hidden_states, len(observed)

        probs = np.zeros(shape=(no_rows, no_cols), dtype=float)  # probability of each state given each observation
        backtrace = np.zeros(shape=(no_rows, no_cols), dtype=int)  # pointer to the best prior state

        for i in range(no_rows):
            probs[i, 0] = self.emission[i, observed[0]] * self.prior_prob[i]

        for o in range(1, no_cols):
            for s in range(no_rows):
                tmp = [probs[k, o - 1] * self.transition[k, s] * self.emission[s, observed[o]] for k in range(no_rows)]
                k = np.argmax(tmp)
                probs[s, o] = probs[k, o - 1] * self.transition[k, s] * self.emission[s, observed[o]]
                backtrace[s, o] = k

        sequence = []
        tmp = [probs[k, no_cols - 1] for k in range(no_rows)]
        k = np.argmax(tmp)
        for o in range(no_cols - 1, -1, -1):
            sequence.append(self.ids[k])
            k = backtrace[k, o]

        tmp = len(sequence)
        sequence = ' '.join(str(s) for s in sequence)

        if length < tmp:
            return sequence
        else:
            return sequence + " " + self.viterbi(sequence, length - tmp)
