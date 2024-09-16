from simplex import Simplex



command = ""
while command.lower() != "end":
    
    print("What a nice day to solve optimization with simplex!(enter end to finish)")
    print("Enter function coeficients: ")
    command = input()
    if command.lower() == "end":
        break
    function_row = list(map(float, command.split(" "))) 
    print("Enter number of constraints and then coeficients of constraints: ")
    n = int(input())
    constraint_coef = []
    for _ in range(n):
        constraint_coef.append(list(map(float, input().split(" "))))
    print("Enter right hand side: ")
    rhs = list(map(float, input().split()))
    print("Enter accuracy: ")
    acc = float(input())
    simplex = Simplex(function_row, constraint_coef, rhs, acc)
    simplex.fill_initial_table()
    answer, max_value = simplex.get_solution()
    print("Solution: ")
    for i in range(len(answer)):
        print(f"x{i + 1} = {answer[i]}")
    print("Max value: ")
    print(max_value)



    
