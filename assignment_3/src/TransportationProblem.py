from typing import List, Union
import numpy as np

class TransportationProblem:
    
    def __init__(
        self, n: int, m: int, A: List[int], B: List[int], costs: List[List[int]]
    ):
        self.n = n
        self.m = m
        self.A = np.array(A)
        self.B = np.array(B)
        self.costs = np.array(costs)

    def is_balanced(self):
        return sum(self.A) == sum(self.B)

