import numpy as np

from Data import *
from Network import Network


if __name__ == '__main__':
    data = Data(name="Harlan Coben - Mickey Bolitar 01 - Shelter (Schronienie)1.pdf")
    hmm = Network(data=data, window=1)

    hmm.print_()
    text = "Find an old lady in a weird white dress and demand she explain her whack-a-doodle rants"
    s = hmm.viterbi_(observed=text, length=30)

    print("-------------------")

    print(f"Input text: {text}")
    print(f"Output text: {s}")
