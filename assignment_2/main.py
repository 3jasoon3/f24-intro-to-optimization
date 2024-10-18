from src.interior_point import InteriorPoint
import numpy as np

command = ""

while command.lower() != "end":
    # print("\nWhat a nice day to solve optimization with this constrained gradient descent-like algorithm")
    # print("Enter 'end' to exit the program")
    # print("Enter function coefficients: ")
    command = input()
    # if command.lower() == "end":
    #     break
    # try:
    #     function_row = list(map(float, command.split()))
    # except ValueError:
    #     print("Invalid function coefficients. Please try again.")
    #     continue
    #
    # print("Enter number of constraints and then coefficients of constraints: ")
    # try:
    #     n = int(input())
    #     constraint_coef = []
    #     for _ in range(n):
    #         constraint_coef.append(list(map(float, input().split())))
    # except ValueError:
    #     print("Invalid constraints. Please try again.")
    #     continue
    #
    # print("Enter right hand side: ")
    # try:
    #     rhs = list(map(float, input().split()))
    # except ValueError:
    #     print("Invalid right-hand side coefficients. Please try again.")
    #     continue
    #
    # print("Enter accuracy: ")
    # try:
    #     acc = float(input())
    # except ValueError:
    #     print("Invalid accuracy value. Please try again.")
    #     continue

    # print("Enter starting point: ")
    # try:
    #     starting_point = tuple(map(float, input().split()))
    # except ValueError:
    #     print("Invalid starting point. Please try again.")
    #     continue

    try:
        function_row = [9,10,16 ]
        constraint_coef = [[18,15,12 ], [6,4,8], [5,3,3]]
        rhs = [360, 192, 180]
        acc = 0.00001

        ip = InteriorPoint(function_row, constraint_coef, rhs, acc)
        x = ip.solve()
        print("Solution: ")
        for i, value in enumerate(x):
            print(f"x{i + 1} = {value:.4f}")

        if not ip.is_converged:
            print("\nWarning: The algorithm did not converge to the specified accuracy.")
            print("The solution may not be optimal.")
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        print("You entered an invalid problem. Please try again.")