from typing import List, Tuple
import numpy as np


class Simplex:

    def __init__(
        self, C: List[float], A: List[float], b: List[float], accuracy: float
    ) -> None:
        self.C_coef = np.array(C)
        self.A_coef = np.array(A)
        self.b_coef = np.array(b)
        self.accuracy = accuracy
        self.table = None
        self.optimised = False

    def fill_initial_table(self) -> None:
        self.table = np.hstack(
            (
                self.A_coef,
                np.eye(self.A_coef.shape[0]),
                np.reshape(self.b_coef, (-1, 1)),
            )
        )
        func = np.hstack((-self.C_coef, np.zeros(self.A_coef.shape[0] + 1)))
        self.table = np.vstack((self.table, func))

    def print_current_table(self) -> None:
        print(self.table)

    def is_optimised(self) -> bool:
        return self.optimised

    def make_iteration(self) -> None:
        if self.table is None:
            print("Table was not initialized!")
            return

        pivot_column = np.argmin(self.table[-1, :-1])
        if self.table[-1, :-1][pivot_column] >= -self.accuracy:
            self.optimised = True
            return
        ratios = np.divide(
            self.table[:-1, -1],
            self.table[:-1, pivot_column],
            out=np.full_like(self.table[:-1, -1], np.inf),
            where=self.table[:-1, pivot_column] > 0,
        )
        pivot_row = np.argmin(ratios)

        self.table[pivot_row] = (
            self.table[pivot_row] / self.table[pivot_row][pivot_column]
        )

        for row in range(self.table.shape[0]):
            if row != pivot_row:
                self.table[row] = (
                    self.table[row]
                    - self.table[row][pivot_column] * self.table[pivot_row]
                )

    def get_solution(self) -> Tuple[List[float], float]:
        while not self.optimised:
            self.make_iteration() # TODO: Add Simplex applicability check

        # Initialize solution array with the correct size: length of C_coef (number of decision variables)
        solution = np.zeros(
            self.C_coef.shape[0] + self.A_coef.shape[0]
        )  # Decision vars + slack vars
        for row in range(self.A_coef.shape[0]):
            # Find the column index in this row where the value is 1
            for col in range(self.C_coef.shape[0] + self.A_coef.shape[0]):
                # Check if this column is a basic variable
                if self.table[row, col] == 1 and np.sum(self.table[:, col]) == 1:
                    # This is a basic variable column
                    solution[col] = self.table[row, -1]
                    break  # Move to the next row

        # Extract decision variables from the solution
        decision_vars = solution[: self.C_coef.shape[0]]
        max_value = self.table[-1, -1]

        return decision_vars, max_value
