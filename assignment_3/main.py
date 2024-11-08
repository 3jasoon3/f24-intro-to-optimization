from src.TransportationProblem import TransportationProblem
from src.NWCM import NWCM
from src.VAM import VAM
from src.Russel import Russel
import numpy as np
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
