import json
from helper import plot_incorrect

incorrect_solutions = json.load(open('incorrect.json'))
for incorrect in incorrect_solutions:
    plot_incorrect(incorrect)