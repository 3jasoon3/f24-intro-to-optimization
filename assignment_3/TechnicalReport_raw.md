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
    - **North-West Corner Method**: `[10, 5, 5, 0, 15, 10, 0, 0, 10]`
    - **Vogel’s Approximation Method**: `[10, 5, 0, 10, 10, 5, 0, 10, 0]`
    - **Russell’s Approximation Method**: `[10, 0, 10, 0, 15, 5, 0, 0, 10]`

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
    - **North-West Corner Method**: `[15, 10, 0, 0, 5, 10, 10, 0, 20]`
    - **Vogel’s Approximation Method**: `[15, 0, 10, 0, 5, 10, 15, 5, 10]`
    - **Russell’s Approximation Method**: `[10, 0, 15, 0, 10, 5, 5, 10, 20]`

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
    - **North-West Corner Method**: `[20, 10, 0, 0, 10, 5, 10, 0, 15]`
    - **Vogel’s Approximation Method**: `[20, 0, 10, 0, 5, 5, 5, 10, 10]`
    - **Russell’s Approximation Method**: `[15, 5, 0, 10, 10, 0, 5, 5, 15]`
