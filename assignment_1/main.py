from src.simplex import Simplex


command = ""
while command.lower() != "end":

    print("What a nice day to solve optimization with simplex! (enter end to finish)")
    print("Enter function coeficients: ")
    command = input()
    if command.lower() == "end":
        break
    try:
        function_row = list(map(float, command.split(" ")))
    except:
        print("Invalid function coefficients. Please, try again.")
        break
    print("Enter number of constraints and then coeficients of constraints: ")
    try:
        n = int(input())
        constraint_coef = []
        for _ in range(n):
            constraint_coef.append(list(map(float, input().split(" "))))
    except:
        print("Invalid constraints. Please, try again.")
        break
    print("Enter right hand side: ")
    try:
        rhs = list(map(float, input().split()))
    except:
        print("Invalid right-hand side coefficients. Please, try again.")
        break
    print("Enter accuracy: ")
    try:
        acc = float(input())
    except:
        print("Invalid accuracy value. Please, try again.")
        break
    try:
        simplex = Simplex(function_row, constraint_coef, rhs, acc)
        simplex.fill_initial_table()
        answer, max_value = simplex.get_solution()
    except:
        print("You entered invalid problem. Please, try again.")
        break
    print("Solution: ")
    for i in range(len(answer)):
        print(f"x{i + 1} = {answer[i]}")
    print("Max value: ")
    print(max_value)
