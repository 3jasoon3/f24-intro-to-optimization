from src.TransportationProblem import TransportationProblem


class NWCM:
    def __init__(self, tp: TransportationProblem):
        self.tp = tp

    def solve(self):
        solution = []
        row_reduction = 0
        col_reduction = 0
        
        
        while row_reduction < self.tp.n and col_reduction < self.tp.m:
            
            if self.tp.A[row_reduction] == 0:
                row_reduction += 1
                continue
            if self.tp.B[col_reduction] == 0:
                col_reduction += 1
                continue
                
            cell = self.tp.costs[row_reduction, col_reduction]
            
            
            if self.tp.A[row_reduction] < self.tp.B[col_reduction]:
                solution.append(
                    (
                        row_reduction,
                        col_reduction,
                        cell,
                        self.tp.A[row_reduction],
                    )
                )
                self.tp.B[col_reduction] -= self.tp.A[row_reduction]
                self.tp.A[row_reduction] = 0
                row_reduction += 1

            
            elif self.tp.A[row_reduction] > self.tp.B[col_reduction]:
                solution.append(
                    (
                        row_reduction,
                        col_reduction,
                        cell,
                        self.tp.B[col_reduction],
                    )
                )
                self.tp.A[row_reduction] -= self.tp.B[col_reduction]
                self.tp.B[col_reduction] = 0
                col_reduction += 1

            else:
                solution.append(
                    (
                        row_reduction,
                        col_reduction,
                        cell,
                        self.tp.A[row_reduction],
                    )
                )
                self.tp.B[col_reduction] = 0
                self.tp.A[row_reduction] = 0
                row_reduction += 1
                col_reduction += 1

        return solution