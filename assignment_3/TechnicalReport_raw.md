# Assignment 3

## Team Information
- **Team Leader:** [Egor Chernobrovkin](e.chernobrovkin@innopolis.university)
- **Team Member 1:** [Dmitrii Kuznetsov](d.kuznetsov@innopolis.university)
- **Team Member 2:** [Alexandra Starikova-Nasibullina](a.nasibullina@innopolis.university)

## Link to the Product
- The product is available at: [GitHub](https://github.com/3jasoon3/f24-intro-to-optimization/tree/main/assignment_3)

## Programming Language
- **Programming Language:** Python

## Tests

### Test Case 1

- **Input**:
  - **Supply Vector (S)**: `[15, 20, 25]`
  - **Demand Vector (D)**: `[10, 10, 20, 20]`
  - **Cost Matrix (C)**:
    \[
    \begin{bmatrix}
    8 & 6 & 10 & 9 \\
    9 & 12 & 13 & 7 \\
    14 & 9 & 16 & 5 \\
    \end{bmatrix}
    \]

- **Output**:
  - **Input Parameter Table**:
    |      | D1 | D2 | D3 | D4 | Supply |
    |------|----|----|----|----|--------|
    | **S1** | 8  | 6  | 10 | 9  | 15     |
    | **S2** | 9  | 12 | 13 | 7  | 20     |
    | **S3** | 14 | 9  | 16 | 5  | 25     |
    | **Demand** | 10 | 10 | 20 | 20 |        |

  - **Initial Basic Feasible Solution Vectors \( x_0 \)**:
    - **North-West Corner Method**: `[[10, 5, 0, 0], [0, 5, 15, 0], [0, 0, 5, 20]]`
    - **Vogel’s Approximation Method**: `[[0, 5, 10, 0], [10, 0, 10, 0], [0, 5, 0, 20]]`
    - **Russell’s Approximation Method**: `[[0, 5, 10, 0], [10, 0, 10, 0], [0, 5, 0, 20]]`

---

### Test Case 2

- **Input**:
  - **Supply Vector (S)**: `[25, 15, 30]`
  - **Demand Vector (D)**: `[15, 10, 25, 20]`
  - **Cost Matrix (C)**:
    \[
    \begin{bmatrix}
    4 & 8 & 6 & 12 \\
    10 & 14 & 7 & 11 \\
    13 & 6 & 15 & 9 \\
    \end{bmatrix}
    \]

- **Output**:
  - **Input Parameter Table**:
    |      | D1 | D2 | D3 | D4 | Supply |
    |------|----|----|----|----|--------|
    | **S1** | 4  | 8  | 6  | 12 | 25     |
    | **S2** | 10 | 14 | 7  | 11 | 15     |
    | **S3** | 13 | 6  | 15 | 9  | 30     |
    | **Demand** | 15 | 10 | 25 | 20 |        |

  - **Initial Basic Feasible Solution Vectors \( x_0 \)**:
    - **North-West Corner Method**: `[[15, 10, 0, 0] [0, 0 ,15, 0], [0,0,10,20]]`
    - **Vogel’s Approximation Method**: `[[15, 0, 10, 0],[0, 0, 15, 0],[0, 10, 0, 20]]`
    - **Russell’s Approximation Method**: `[[15, 0, 10, 0], [0, 0, 15, 0], [0,10,0,20]]`

---

### Test Case 3

- **Input**:
  - **Supply Vector (S)**: `[30, 25, 15]`
  - **Demand Vector (D)**: `[20, 10, 15, 25]`
  - **Cost Matrix (C)**:
    \[
    \begin{bmatrix}
    5 & 9 & 12 & 8 \\
    6 & 11 & 14 & 10 \\
    15 & 13 & 10 & 7 \\
    \end{bmatrix}
    \]

- **Output**:
  - **Input Parameter Table**:
    |      | D1 | D2 | D3 | D4 | Supply |
    |------|----|----|----|----|--------|
    | **S1** | 5  | 9  | 12 | 8  | 30     |
    | **S2** | 6  | 11 | 14 | 10 | 25     |
    | **S3** | 15 | 13 | 10 | 7  | 15     |
    | **Demand** | 20 | 10 | 15 | 25 |        |

  - **Initial Basic Feasible Solution Vectors \( x_0 \)**:
    - **North-West Corner Method**: `[[20, 10, 0, 0] [0, 0 ,15, 10], [0, 0, 0, 15]]`
    - **Vogel’s Approximation Method**: `[[0, 10, 10, 10],[20, 0, 5, 0],[0, 0, 0, 15]]`
    - **Russell’s Approximation Method**: `[[0, 10, 0, 20], [20, 0, 0, 5], [0, 0, 15, 0]]`

## Setup and Run
```bash
cd assignment_3
python main.py
```

## Code
`NWCM.py`
```python
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
```


`Russel.py`
```python
from src.TransportationProblem import TransportationProblem
import numpy as np


class Russel:
    def __init__(self, transportation_problem: TransportationProblem):
        self.tp = transportation_problem
        self.costs = self.tp.costs.copy()
        self.supply = self.tp.A.copy()
        self.demand = self.tp.B.copy()
        self.row_indices = np.arange(len(self.supply))
        self.col_indices = np.arange(len(self.demand))

    def solve(self):
        if not self.tp.is_balanced():
            print("Problem is unbalanced")
            return []

        solution = []
        while len(self.supply) > 0 and len(self.demand) > 0:

            row_maxima = np.max(self.costs, axis=1)
            col_maxima = np.max(self.costs, axis=0)

            
            differences = np.zeros((len(self.supply), len(self.demand)))
            for i in range(len(self.supply)):
                for j in range(len(self.demand)):
                    differences[i, j] = self.costs[i, j] - (row_maxima[i] + col_maxima[j])

            min_i, min_j = np.unravel_index(differences.argmin(), differences.shape)
            allocation = min(self.supply[min_i], self.demand[min_j])
            orig_row = self.row_indices[min_i]
            orig_col = self.col_indices[min_j]
            solution.append((orig_row, orig_col, self.costs[min_i, min_j], allocation))
            if self.supply[min_i] < self.demand[min_j]:
                self.demand[min_j] -= self.supply[min_i]
                self.costs = np.delete(self.costs, min_i, axis=0)
                self.supply = np.delete(self.supply, min_i)
                self.row_indices = np.delete(self.row_indices, min_i)
            elif self.supply[min_i] > self.demand[min_j]:
                self.supply[min_i] -= self.demand[min_j]
                self.costs = np.delete(self.costs, min_j, axis=1)
                self.demand = np.delete(self.demand, min_j)
                self.col_indices = np.delete(self.col_indices, min_j)
            else:
                self.costs = np.delete(np.delete(self.costs, min_i, axis=0), min_j, axis=1)
                self.supply = np.delete(self.supply, min_i)
                self.demand = np.delete(self.demand, min_j)
                self.row_indices = np.delete(self.row_indices, min_i)
                self.col_indices = np.delete(self.col_indices, min_j)

        return solution
```


`VAM.py`
```python
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

            
            supply[row] -= allocation
            demand[col] -= allocation

           
            solution.append((orig_row, orig_col, cost, allocation))

            
            supply_indices = np.where(supply > 0)[0]
            demand_indices = np.where(demand > 0)[0]

            costs = costs[np.ix_(supply_indices, demand_indices)]
            supply = supply[supply_indices]
            demand = demand[demand_indices]
            row_map = row_map[supply_indices]
            col_map = col_map[demand_indices]

        return solution











```

`TransportationProblem.py`
```python
from typing import List
import numpy as np


class TransportationProblem:
    def __init__(self, n : int  ,m : int, A: List[int], B: List[int], costs: List[List[int]] ):
        self.n = n
        self.m = m
        self.A = np.array(A)
        self.B = np.array(B)
        self.costs = np.array(costs)

    def is_balanced(self):
        return sum(self.A) == sum(self.B)







```


`main.py`
```python
from src.TransportationProblem import TransportationProblem
from src.NWCM import NWCM
from src.VAM import VAM
from src.Russel import Russel
import numpy as np
m = 4
n = 3
costs = [
    [3,1,7,4],
    [2,6,5,9],
    [8,3,3,2]
]
a = [300, 400, 500]
b = [250,350,400,200]



command = ""

while command.lower() != "end":
    print("\nWhat a nice day to solve transportation problem!")
    print("Enter 'end' to exit the program")
    print("Enter vector of supplies: ")
    command: str = input()
    if command.lower() == "end":
        break
    try:
        supply= list(map(int, command.split()))
    except ValueError:
        print("Invalid supply. Please try again.")
        continue

    print("Enter size of costs matrix(row, columns): and costs matrix itself ")
    try:
        n,m = map(int, input().split())
        costs = []
        for _ in range(n):
            costs.append(list(map(int, input().split())))
    except ValueError:
        print("Invalid costs. Please try again.")
        continue

    print("Enter vectors of demands: ")
    try:
        demands = list(map(int, input().split()))
    except ValueError:
        print("Invalid vector of demands. Please try again.")
        continue



    try:

        problem = TransportationProblem(n, m, supply, demands, costs)
        nvcm = NWCM(problem)
        nvcm_sol = nvcm.solve()
        print("--------------------------------")
        print("Initial feasible solution by NVCM:")
        initial = np.zeros((3,4))
        for item in nvcm_sol:
            initial[item[0]][item[1]] = item[3]

        for i in range(n):
            for j in range(m):
                print(initial[i][j], end=" ")
            print("\n")
        print("--------------------------------")
        problem = TransportationProblem(n, m, supply, demands, costs)
        vam = VAM(problem)
        vam_sol = vam.solve()
        print("Initial feasible solution by VAM:")
        initial = np.zeros((3, 4))
        for item in vam_sol:
            initial[item[0]][item[1]] = item[3]

        for i in range(n):
            for j in range(m):
                print(initial[i][j], end=" ")
            print("\n")
        print("--------------------------------")
        problem = TransportationProblem(n, m, supply, demands, costs)
        russel = Russel(problem)
        russel_sol = russel.solve()
        print("Initial feasible solution by Russel:")
        initial = np.zeros((3, 4))
        for item in russel_sol:
            initial[item[0]][item[1]] = item[3]

        for i in range(n):
            for j in range(m):
                print(initial[i][j], end=" ")
            print("\n")
        print("--------------------------------")
    except Exception as e:
        print(f"An error occurred: {str(e)}")

```