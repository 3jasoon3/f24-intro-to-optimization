from assignment_3.src.TransportationProblem import TransportationProblem
class NWCM:
    def __init__(self, tp: TransportationProblem):
        self.tp = tp


    def solve(self):
        solution = []
        row_reduction = 0
        col_reduction = 0
        for i in range(int(self.tp.n*self.tp.m *(0.5))):
            cell = self.tp.costs[0 +row_reduction, 0 +col_reduction]
            if self.tp.A[0 + row_reduction] < self.tp.B[0 + col_reduction]:

                solution.append((0+row_reduction, 0+col_reduction,
                                 self.tp.costs[0+row_reduction][0+col_reduction], self.tp.A[0 + row_reduction]))
                self.tp.B[0 + col_reduction] -= self.tp.A[0 + row_reduction]
                self.tp.A[0 + row_reduction]  = 0
                row_reduction += 1

            elif self.tp.A[0 + row_reduction] > self.tp.B[0 + col_reduction]:

                solution.append((0+row_reduction, 0+col_reduction,
                                 self.tp.costs[0+row_reduction][0+col_reduction], self.tp.B[0 +col_reduction]))
                self.tp.A[0 + row_reduction] -= self.tp.B[0 + col_reduction]
                self.tp.B[0 + col_reduction]  = 0
                col_reduction += 1

            elif self.tp.A[0 + row_reduction] == self.tp.B[0 + col_reduction]:

                solution.append((0+row_reduction, 0+col_reduction,
                                 self.tp.costs[0+row_reduction][0+col_reduction], self.tp.A[0 + row_reduction]))
                self.tp.B[0 + col_reduction] = 0
                self.tp.A[0 + row_reduction]  = 0
                row_reduction += 1
                col_reduction += 1
