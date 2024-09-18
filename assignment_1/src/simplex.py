from typing import List, Tuple
import numpy as np


class Simplex:
    """
    The Simplex class implements the Simplex method for solving linear programming problems.
    """

    def __init__(
        self, C: List[float], A: List[float], b: List[float], accuracy: float
    ) -> None:
        """
        Initializes the Simplex method with the following inputs:
        C: Coefficients of the objective function.
        A: Coefficients of the inequality constraints.
        b: Right-hand side values of the inequality constraints.
        accuracy: Precision for detecting optimality (helps handle floating-point errors).
        """
        self.C_coef = np.array(C)  # Objective function coefficients
        self.A_coef = np.array(A)  # Coefficients of the constraints
        self.b_coef = np.array(b)  # Right-hand side values
        self.accuracy = accuracy  # Desired accuracy
        self.table = None  # Simplex table
        self.optimised = False  # Indicates whether the solution is optimized
        self.solvable = True  # Indicates whether the problem is solvable

    def check_infeasibility(self) -> bool:
        """
        If any value in the right-hand side vector b is negative
        and the corresponding row in the matrix A has no positive coefficients,
        the problem is infeasible.
        """
        for i in range(len(self.b_coef)):
            if self.b_coef[i] < 0 and all(
                self.A_coef[i][j] <= 0 for j in range(len(self.A_coef[i]))
            ):
                return True
        return False

    def check_unboudedness(self, ratios: np.ndarray) -> bool:
        """
        If the objective function can grow indefinitely in the direction
        of the feasible region, then the problem is unbounded.
        """
        if np.all(np.isinf(ratios)):
            print("The method is not applicable!")
            self.solvable = False
            return True
        return False

    def fill_initial_table(self) -> None:
        """
        Initializes the Simplex table by combining the constraint matrix A,
        the identity matrix (for slack variables), and the right-hand side vector b.
        Also appends the objective function row with negative coefficients of C.
        """
        self.table = np.hstack(
            (
                self.A_coef,  # Coefficients of the constraints
                np.eye(self.A_coef.shape[0]),  # Identity matrix for slack variables
                np.reshape(self.b_coef, (-1, 1)),  # Right-hand side vector b
            )
        )
        # Objective function row (negative coefficients of C)
        func = np.hstack((-self.C_coef, np.zeros(self.A_coef.shape[0] + 1)))
        # Add the objective function row at the bottom
        self.table = np.vstack((self.table, func))

    def make_iteration(self) -> None:
        """
        Performs one iteration of the Simplex algorithm:
        1. Finds the pivot column.
        2. Checks for unboundedness.
        3. Performs the pivot operation to transform the table.
        """

        if self.table is None:
            print("Table was not initialized!")
            return

        # Find the most negative value in the objective row
        pivot_column = np.argmin(self.table[-1, :-1])

        # Check if the solution is already optimal
        if self.table[-1, :-1][pivot_column] >= -self.accuracy:
            self.optimised = True
            return

        # Compute the ratios for the ratio test
        ratios = np.divide(
            self.table[:-1, -1],  # Right-hand side values (b)
            self.table[:-1, pivot_column],  # Pivot column values
            out=np.full_like(
                self.table[:-1, -1], np.inf
            ),  # Fill with inf where division is not valid
            where=self.table[:-1, pivot_column]
            > 0,  # Only consider positive entries in the pivot column
        )

        if self.check_unboudedness(ratios):
            return

        # Select the pivot row
        pivot_row = np.argmin(ratios)
        # Normalize the pivot row
        self.table[pivot_row] = (
            self.table[pivot_row] / self.table[pivot_row][pivot_column]
        )
        # Make all other elements in the pivot column zero
        for row in range(self.table.shape[0]):
            if row != pivot_row:
                self.table[row] = (
                    self.table[row]
                    - self.table[row][pivot_column] * self.table[pivot_row]
                )

    def get_solution(self) -> Tuple[List[float], float]:
        """
        Returns the decision variables and the optimized objective function value if the solution exists.
        """

        # Check if the problem is infeasible
        if self.check_infeasibility():
            print("The method is not applicable!")
            self.solvable = False

        # Perform iterations while the solution is not optimized
        while (not self.optimised) and self.solvable:
            self.make_iteration()

        # If the problem is unsolvable, return empty results
        if not self.solvable:
            return [], None

        # Initialize solution array (size of decision variables + slack variables)
        solution = np.zeros(self.C_coef.shape[0] + self.A_coef.shape[0])

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

        # Round to 10 decimal places
        decision_vars = [round(var, 10) for var in decision_vars]
        max_value = round(self.table[-1, -1], 10)

        return decision_vars, max_value
