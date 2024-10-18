import numpy as np
from typing import List, Tuple
import random

class InteriorPoint:
    def __init__(self, C: List[float], A: List[List[float]],
                 b: List[float], accuracy: float, a: float = 0.5, starting_point: Tuple[float, float] = None,
                 ) -> None:
        """
        Initializes the problem with the following inputs:
        C: adjficients of the objective function (for maximization).
        A: adjficients of the inequality constraints.
        b: Right-hand side values of the inequality constraints.
        accuracy: Precision for detecting optimality (helps handle floating-point errors).
        """

        self.A_origin = np.array(A)  # adjficients of the constraints
        self.b = np.array(b)  # Right-hand side values
        self.accuracy = accuracy  # Desired accuracy
        self.starting_point = starting_point  # Initial point of algorithm
        self.n = len(C)
        self.m = len(b)

        self.a = a   # Speed of convergence(step)
        self.is_converged = False  # Indicates whether the solution is optimized
        self.solvable = True  # Indicates whether the problem is solvable


        #Adjasting slack
        self.C_adj = np.hstack((np.array(C), np.zeros(self.A_origin.shape[0])))  # Objective function adjficients
        self.A_adj = np.hstack((self.A_origin,  np.eye(self.A_origin.shape[0])))
        if starting_point is None:
            self.starting_point = self.generate_random_point()
            print(self.starting_point)



    def make_iteration(self, c: np.ndarray, a: np.ndarray, b: np.ndarray, x0, alpha=0.5, epsilon=1e-6):
        x = x0.copy()
        D = np.diag(x)
        AA = np.dot(a, D)
        cc = np.dot(D, c)
        I = np.eye(len(c))
        F = np.dot(AA, np.transpose(AA))
        FI = np.linalg.inv(F)
        H = np.dot(np.transpose(AA), FI)
        P = np.subtract(I, np.dot(H, AA))
        cp = np.dot(P, cc)
        nu = np.absolute(np.min(cp))
        y = np.add(np.ones(len(c), float), (alpha / nu) * cp)
        yy = np.dot(D, y)
        x = yy

        return x

    def generate_random_point(self):
        """
        Generate an initial feasible point x0 such that A_adj @ x0 = b and x0 > 0.
        """
        n_vars = self.A_adj.shape[1]
        x0 = np.ones(n_vars)  # Start with all ones


        for _ in range(100):
            residual = self.A_adj @ x0 - self.b
            if np.linalg.norm(residual) < self.accuracy:
                break

            delta_x = np.linalg.lstsq(self.A_adj, self.b - self.A_adj @ x0, rcond=None)[0]
            x0 += delta_x

            x0 = np.maximum(x0, 1e-6)
        return x0
    def solve(self, alpha=0.5, max_iterations=1000, epsilon=1e-6):
        """
        Solve given problem

        :param alpha: Convergence speed(step size)
        :param max_iterations: Maximum number of iterations
        :param epsilon: Precision for detecting optimality (helps handle floating-point errors).
        :return: The optimal solution
        """


        x = np.array(self.starting_point)

        print(f"Initial point: {x}")
        print(f"Initial objective value: {self.C_adj.dot(x)}")

        for iteration in range(max_iterations):
            x_new = self.make_iteration(self.C_adj, self.A_adj, self.b, x, alpha, epsilon)

            print(f"Iteration {iteration + 1}:")
            print(f"  x = {x_new}")
            print(f"  Objective value: {self.C_adj.dot(x_new)}")
            print(x, x_new, np.allclose(x, x_new, rtol=epsilon, atol=epsilon))
            if np.allclose(x, x_new, rtol=epsilon, atol=epsilon):
                self.is_converged = True
                return x_new

            x = x_new

        # no convergence :(
        self.is_converged = False
        return x