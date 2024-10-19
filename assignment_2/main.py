from src.interior_point import InteriorPoint

command = ""

while command.lower() != "end":
    print("\nWhat a nice day to solve optimization with this constrained gradient descent-like algorithm")
    print("Enter 'end' to exit the program")
    print("Enter function coefficients: ")
    command: str = input()
    if command.lower() == "end":
        break
    try:
        function_row = list(map(float, command.split()))
    except ValueError:
        print("Invalid function coefficients. Please try again.")
        continue
    
    print("Enter number of constraints and then coefficients of constraints: ")
    try:
        n = int(input())
        constraint_coef = []
        for _ in range(n):
            constraint_coef.append(list(map(float, input().split())))
    except ValueError:
        print("Invalid constraints. Please try again.")
        continue
    
    print("Enter right hand side: ")
    try:
        rhs = list(map(float, input().split()))
    except ValueError:
        print("Invalid right-hand side coefficients. Please try again.")
        continue
    
    print("Enter accuracy: ")
    try:
        acc = float(input())
    except ValueError:
        print("Invalid accuracy value. Please try again.")
        continue

    try:
        ip = InteriorPoint(function_row, constraint_coef, rhs, acc)

        # Solve the problem, getting the decision variables and optimized objective value
        answer, max_value = ip.solve()
        
        if answer is None:  # No solution found
            break

        # Print decision variables
        print("Decision variables:")
        for i in range(len(answer)):
            print(f"x{i + 1} = {answer[i]}")
        
        # Print the optimized objective value
        print(f"Optimized objective function's value: {max_value}")
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        print("You entered an invalid problem. Please try again.")
