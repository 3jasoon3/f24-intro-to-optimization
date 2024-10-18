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
            self.starting_point = self.generate_random_point()[0]
            print(self.starting_point)



    def make_iteration(self, c: np.ndarray, a: np.ndarray, b: np.ndarray, x0, alpha=0.5, epsilon=1e-6):


        n = len(c)
        x = x0.copy()

        # Create diagonal matrix D - new basis
        D = np.diag(x)


        # Adjust A matrix and function adjficients
        A_new = a @ D
        c_new = D @ c

        # Calculate projection matrix P and projected gradient c_p
        A_new_T = A_new.T
        P = np.eye(n) - A_new_T @ np.linalg.inv(A_new @ A_new_T) @ A_new
        c_p = P @ c_new

        # Calculate x in new basis (note the change for maximization)
        negative_c_p = c_p[c_p < 0]
        if negative_c_p.size > 0:
            v = max(abs(negative_c_p.min()), epsilon)
        else:
            v = epsilon
        x_tilde = np.ones(n) + (alpha / v) * c_p

        # Return x in old basis
        x_new = D @ x_tilde

        return x_new

    def generate_random_point(self):
        b = self.b.reshape(-1, 1)
        a_norm = self.A_adj/ b
        random_point = np.random.random((1, self.A_adj.shape[1]))
        normalized = random_point/np.sum(random_point)



        print(np.multiply(a_norm, normalized))
        return np.multiply(a_norm, normalized)
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