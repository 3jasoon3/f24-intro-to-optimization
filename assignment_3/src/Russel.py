from assignment_3.src.TransportationProblem import TransportationProblem
import numpy as np
class Russel:
    def __init__(self, transportation_problem : TransportationProblem):
        self.tp = transportation_problem

    def solve(self):
        if not self.tp.is_balanced():
            print("Problem is unbalanced")
            return

        row_reduction = 0
        col_reduction = 0
        solution = []
        for i in range(int(self.tp.n * self.tp.m/2)):

            differences = np.zeros((self.tp.n, self.tp.m))
            for row in range(self.tp.n - row_reduction):
                for col in range(self.tp.m - col_reduction):
                    differences[row, col] = self.tp.costs[row][col] - (
                                max(self.tp.costs[row]) + max(self.tp.costs[:, col]))
            element_i = np.unravel_index(differences.argmin(), differences.shape)

            row_i = element_i[0]
            col_i = element_i[1]

            if self.tp.A[row_i] < self.tp.B[col_i]:

                solution.append((row_i + row_reduction, col_i+col_reduction ,
                                 self.tp.costs[row_i][col_i], self.tp.A[row_i]))
                self.tp.B[col_i] -= self.tp.A[row_i]
                self.tp.A = np.delete(self.tp.A, row_i, axis=0)
                self.tp.costs = np.delete(self.tp.costs, row_i, axis=0)

                row_reduction += 1




            elif self.tp.A[row_i] > self.tp.B[col_i]:

                solution.append((row_i+row_reduction, col_i+col_reduction,
                                 self.tp.costs[row_i][col_i], self.tp.B[col_i]))
                self.tp.A[row_i] -= self.tp.B[col_i]
                self.tp.B = np.delete(self.tp.B, col_i, axis=0)
                self.tp.costs = np.delete(self.tp.costs, col_i, axis=1)

                col_reduction += 1


            elif self.tp.A[row_i] == self.tp.B[col_i]:

                solution.append((row_i + row_reduction, col_i + col_reduction,
                                 self.tp.costs[row_i][col_i], self.tp.A[row_i]))
                self.tp.B = np.delete(self.tp.B, col_i, axis=0)
                self.tp.A = np.delete(self.tp.A, row_i, axis=0)
                self.tp.costs = np.delete(self.tp.costs, row_i, axis=0)
                self.tp.costs = np.delete(self.tp.costs, col_i, axis=1)

                row_reduction += 1
                col_reduction += 1
        return solution


