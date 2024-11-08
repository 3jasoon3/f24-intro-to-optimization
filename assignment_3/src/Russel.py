from src.TransportationProblem import TransportationProblem
import numpy as np


class Russel:
    def __init__(self, transportation_problem: TransportationProblem):
        self.tp = transportation_problem
        self.costs = self.tp.costs.copy()
        self.supply = self.tp.A.copy()
        self.demand = self.tp.B.copy()
        self.row_indices = np.arange(len(self.supply))
        self.col_indices = np.arange(len(self.demand))

    def solve(self):
        if not self.tp.is_balanced():
            print("Problem is unbalanced")
            return []

        solution = []
        while len(self.supply) > 0 and len(self.demand) > 0:

            row_maxima = np.max(self.costs, axis=1)
            col_maxima = np.max(self.costs, axis=0)

            
            differences = np.zeros((len(self.supply), len(self.demand)))
            for i in range(len(self.supply)):
                for j in range(len(self.demand)):
                    differences[i, j] = self.costs[i, j] - (row_maxima[i] + col_maxima[j])

            min_i, min_j = np.unravel_index(differences.argmin(), differences.shape)
            allocation = min(self.supply[min_i], self.demand[min_j])
            orig_row = self.row_indices[min_i]
            orig_col = self.col_indices[min_j]
            solution.append((orig_row, orig_col, self.costs[min_i, min_j], allocation))
            if self.supply[min_i] < self.demand[min_j]:
                self.demand[min_j] -= self.supply[min_i]
                self.costs = np.delete(self.costs, min_i, axis=0)
                self.supply = np.delete(self.supply, min_i)
                self.row_indices = np.delete(self.row_indices, min_i)
            elif self.supply[min_i] > self.demand[min_j]:
                self.supply[min_i] -= self.demand[min_j]
                self.costs = np.delete(self.costs, min_j, axis=1)
                self.demand = np.delete(self.demand, min_j)
                self.col_indices = np.delete(self.col_indices, min_j)
            else:
                self.costs = np.delete(np.delete(self.costs, min_i, axis=0), min_j, axis=1)
                self.supply = np.delete(self.supply, min_i)
                self.demand = np.delete(self.demand, min_j)
                self.row_indices = np.delete(self.row_indices, min_i)
                self.col_indices = np.delete(self.col_indices, min_j)

        return solution