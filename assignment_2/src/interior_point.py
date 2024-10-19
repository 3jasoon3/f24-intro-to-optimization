from typing import List, Tuple
import numpy as np


class InteriorPoint:
    def __init__(
        self,
        C: List[float],
        A: List[List[float]],
        b: List[float],
        accuracy: float,
        alpha: float = 0.5,
        starting_point: List[float] = None,
    ) -> None:
        """
        Initializes the problem with the following inputs:
        C: adjficients of the objective function (for maximization).
        A: adjficients of the inequality constraints.
        b: Right-hand side values of the inequality constraints.
        accuracy: Precision for detecting optimality (helps handle floating-point errors).
        """

        self.A_origin: List[List[float]] = np.array(A)  # adjficients of the constraints
        self.b: List[float] = np.array(b)  # Right-hand side values
        self.accuracy: float = accuracy  # Desired accuracy
        self.starting_point: List[float] = starting_point
        self.n: int = len(C)
        self.m: int = len(b)

        self.alpha: float = alpha  # Speed of convergence(step)
        self.is_converged: bool = False  # Indicates whether the solution is optimized
        self.solvable: bool = True  # Indicates whether the problem is solvable

        # Adjasting slack
        self.C_adj = np.hstack(
            (np.array(C), np.zeros(self.A_origin.shape[0]))
        )  # Objective function adjficients
        self.A_adj = np.hstack((self.A_origin, np.eye(self.A_origin.shape[0])))
        if starting_point is None:
            self.starting_point = self.generate_random_point()

    def make_iteration(
        self,
        c: np.ndarray,
        a: np.ndarray,
        x0: np.ndarray,
        alpha: float = 0.5,
    ) -> List[float]:
        x = x0.copy()
        D = np.diag(x)
        AA = np.dot(a, D)
        cc = np.dot(D, c)
        F = np.dot(AA, np.transpose(AA))
        FI = np.linalg.inv(F)
        H = np.dot(np.transpose(AA), FI)
        P = np.subtract(np.eye(len(c)), np.dot(H, AA))
        cp = np.dot(P, cc)
        nu = np.absolute(np.min(cp))
        y = np.add(np.ones(len(c), float), (alpha / nu) * cp)
        yy = np.dot(D, y)
        x = yy

        return x

    def generate_random_point(self) -> List[float]:
        """
        Generate an initial feasible point x0 such that A_adj @ x0 = b and x0 > 0.
        """
        n_vars: int = self.A_adj.shape[1]
        x0 = np.ones(n_vars)  # Start with all ones

        for _ in range(100):
            residual = self.A_adj @ x0 - self.b
            if np.linalg.norm(residual) < self.accuracy:
                break

            delta_x = np.linalg.lstsq(self.A_adj, self.b - self.A_adj @ x0, rcond=None)[
                0
            ]
            x0 += delta_x

            x0 = np.maximum(x0, 1e-6)

        return x0

    def solve(
        self, alpha: float = 0.5, max_iterations: int = 1000, epsilon: float = 1e-6
    ) -> Tuple[List[float], float]:
        """
        Solve the given problem and return the decision variables and optimized objective value.

        :param alpha: Convergence speed (step size)
        :param max_iterations: Maximum number of iterations
        :param epsilon: Precision for detecting optimality (helps handle floating-point errors)
        :return: A tuple containing the optimized decision variables and the objective value
        """

        x = np.array(self.starting_point)

        for iteration in range(max_iterations):
            x_new = self.make_iteration(self.C_adj, self.A_adj, x, alpha)

            if np.allclose(x, x_new, rtol=epsilon, atol=epsilon):
                self.is_converged = True
                break

            x = x_new

        # Check convergence status
        if not self.is_converged:
            print(
                "Warning: The algorithm did not converge within the maximum iterations."
            )

        # Extract decision variables (excluding slack variables)
        decision_variables = x[: self.n]

        # Calculate the optimized objective value
        optimized_objective = self.C_adj[: self.n].dot(decision_variables)

        return decision_variables, optimized_objective
