from assignment_3.src.TransportationProblem import TransportationProblem
import numpy as np
class VAM:
    def __init__(self, tp: TransportationProblem):
        self.tp = tp


    def solve(self):
        if self.tp.is_balanced():
            print("Problem is unbalanced")
            return
        solution = []
        row_difference = []
        col_difference = []
        row_reduction = 0
        col_reduction = 0
        for row in range(self.tp.n):
            x1, x2 = np.partition(self.tp.costs[row], 1)[0:2]
            row_difference.append(abs(x1 - x2))
        for col in range(self.tp.m):
            x1, x2 = np.partition(self.tp.costs[:, col], 1)[0:2]
            col_difference.append(abs(x1 - x2))
        solution = []
        for i in range(int(self.tp.n*self.tp.m *(0.5))):
            if max(row_difference) > max(col_difference):
                row_i = np.argmax(row_difference)
                element = min(self.tp.costs[row_i])
                col_i = np.argmin(self.tp.costs[row_i])
            else:
                col_i = np.argmax(col_difference)
                element = min(self.tp.costs[:,col_i])
                row_i = np.argmin(self.tp.costs[:,col_i])
            if self.tp.A[row_i] < self.tp.B[col_i]:

                solution.append((row_i + row_reduction, col_i+col_reduction ,
                                 element, self.tp.A[row_i]))
                self.tp.B[col_i] -= self.tp.A[row_i]
                self.tp.A = np.delete(self.tp.A, row_i, axis=0)
                self.tp.costs = np.delete(self.tp.costs, row_i, axis=0)
                row_difference.remove(row_difference[row_i])
                row_reduction += 1




            elif self.tp.A[row_i] > self.tp.B[col_i]:

                solution.append((row_i+row_reduction, col_i+col_reduction,
                                 element, self.tp.B[col_i]))
                self.tp.A[row_i] -= self.tp.B[col_i]
                self.tp.B = np.delete(self.tp.B, col_i, axis=0)
                self.tp.costs = np.delete(self.tp.costs, col_i, axis=1)
                col_difference.remove(col_difference[col_i])
                col_reduction += 1


            elif self.tp.A[row_i] == self.tp.B[col_i]:

                solution.append((row_i + row_reduction, col_i + col_reduction,
                                 element, self.tp.A[row_i]))
                self.tp.B = np.delete(self.tp.B, col_i, axis=0)
                self.tp.A = np.delete(self.tp.A, row_i, axis=0)
                self.tp.costs = np.delete(self.tp.costs, row_i, axis=0)
                self.tp.costs = np.delete(self.tp.costs, col_i, axis=1)
                col_difference.remove(col_difference[col_i])
                row_difference.remove(row_difference[row_i])
                row_reduction += 1
                col_reduction += 1

        return solution










