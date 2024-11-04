import numpy as np


class LeastCost:

    def __init__(self, problem):
        self.problem = problem

    def find_initial_feasible_solution(self):
        a = self.problem.A.copy()
        b = self.problem.B.copy()
        costs = self.problem.costs.copy()
        initial_feasible_solution = []

        while any(a != 0) and any(b != 0):

            m_i = np.argwhere(costs == np.min(costs))

            i, j = m_i[0][0], m_i[0][1]
            x = min(a[i], b[j])
            initial_feasible_solution.append((i, j, costs[i][j], x))
            a[i] -= x
            b[j] -= x
            if a[i] == 0:
                costs[i, :] = 100000000
            if b[j] == 0:
                costs[:, j] = 100000000
        return initial_feasible_solution
