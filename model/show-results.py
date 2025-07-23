import json
from helper import plot_incorrect

def results(fileName):
    incorrect_solutions = json.load(open('data/output/' + fileName + '.json'))
    for incorrect in incorrect_solutions:
        plot_incorrect(incorrect)

results('incorrect19')