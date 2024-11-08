from src.TransportationProblem import TransportationProblem
import numpy as np
from typing import List, Tuple


class VAM:
    def __init__(self, tp: TransportationProblem):
        self.tp = tp
    def solve(self):
        if not self.tp.is_balanced():
            print("Problem is unbalanced")
            return []

        
        costs = self.tp.costs.copy()
        supply = self.tp.A.copy()
        demand = self.tp.B.copy()
        solution = []

        
        row_map = np.arange(len(supply))
        col_map = np.arange(len(demand))

        while len(supply) > 0 and len(demand) > 0:
            row_diff = np.zeros(len(supply))
            for i in range(len(supply)):
                if len(costs[i]) >= 2:
                    sorted_costs = np.sort(costs[i])
                    row_diff[i] = sorted_costs[1] - sorted_costs[0]

           
            col_diff = np.zeros(len(demand))
            for j in range(len(demand)):
                column_costs = costs[:, j]
                if len(column_costs) >= 2:
                    sorted_costs = np.sort(column_costs)
                    col_diff[j] = sorted_costs[1] - sorted_costs[0]
            max_row_penalty = np.max(row_diff)
            max_col_penalty = np.max(col_diff)
            if max_row_penalty >= max_col_penalty:
                row = np.argmax(row_diff)
                col = np.argmin(costs[row])
            else:
                col = np.argmax(col_diff)
                row = np.argmin(costs[:, col])

           
            orig_row = row_map[row]
            orig_col = col_map[col]
            cost = costs[row, col]
            allocation = min(supply[row], demand[col])

            # Update supply and demand
            supply[row] -= allocation
            demand[col] -= allocation

            # Add to solution
            solution.append((orig_row, orig_col, cost, allocation))

            # Update matrices
            supply_indices = np.where(supply > 0)[0]
            demand_indices = np.where(demand > 0)[0]

            costs = costs[np.ix_(supply_indices, demand_indices)]
            supply = supply[supply_indices]
            demand = demand[demand_indices]
            row_map = row_map[supply_indices]
            col_map = col_map[demand_indices]

        return solution