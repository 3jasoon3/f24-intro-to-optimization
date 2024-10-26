from src.TransportationProblem import TransportationProblem
from src.NWCM import NWCM
from src.VAM import VAM
from src.Russel import Russel
m = 4
n = 3
costs = [[3,1,7,4], [2,6,5,9], [8,3,3,2]]
a = [300, 400, 500]
b = [250,350,400,200]

problem = TransportationProblem(n, m, a,b, costs)
russel = Russel(problem)
sol = russel.solve()
for i in range(len(sol)):
    print(sol[i])

