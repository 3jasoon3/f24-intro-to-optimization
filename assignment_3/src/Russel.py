from assignment_3.src.TransportationProblem import TransportationProblem
import numpy as np
class Russel:
    def __init__(self, transportation_problem : TransportationProblem):
        self.tp = transportation_problem

    def solve(self):
        if not self.tp.is_balanced():
            print("Problem is unbalanced")
            return
