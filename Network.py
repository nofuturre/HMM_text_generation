import numpy as np


class Network:

    def __init__(self, init, transmission, first_words, tokens, no_hidden_layers):
        self.initial_states = first_words
        self.initial_states_prob = init
        self.transmission_probabilities = transmission
        self.hidden_states = tokens
        self.no_hidden_layers = no_hidden_layers
        self.emission_probability = transmission
        self.emission_states = tokens
        self.no_hidden_states = len(tokens)

    def forward(self):
        alpha = np.zeros(self.no_hidden_states, self.no_hidden_states)