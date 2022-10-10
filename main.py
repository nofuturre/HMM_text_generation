from data_preparation import *


if __name__ == '__main__':
    data = read_pdf_file("Harlan Coben - Mickey Bolitar 01 - Shelter (Schronienie)1.pdf")
    tokens = mapping(data)
    ids = tokens_ids(data)
    verbose = False

    print(f"Number of unique words: {len(tokens)}")

    initial_emission = first_words_dist(data, tokens)

    if verbose:
        for k, v in initial_emission.items():
            if k != 0.0:
                print(f"Word: {ids[v]} \t, initial emission probability: \t {k}")

    data, tokens = normalize(data)
    print(tokens)
    transition_matrix = transitions(data, tokens, 2)
