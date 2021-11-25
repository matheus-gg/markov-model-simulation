import numpy as np
import pandas as pd
from collections import deque
from numpy.linalg import matrix_power
import matplotlib.pyplot as plt

deltaT = 1
l = 0.005
muC = 0.05
muP = 0.01
C = 0.6
n = 1000

A = np.array([
  [1-2*l*deltaT, muC*deltaT, muC*deltaT, muP*deltaT, muC*deltaT],
  [2*l*deltaT*C, 0.0, 0.0, 0.0, 0.0],
  [0.0, l*deltaT*C, 0.0, l*deltaT*C, 0.0],
  [l*deltaT*(1 - C), 0.0, 0.0, 0.0, 0.0],
  [l*deltaT*(1 - C), l*deltaT*(1 - C), l*deltaT, l*deltaT*(1 - C), 1 - muC*deltaT]])
print("A = \n", A)
print("================================")

initialState = np.array([
  [1.0], 
  [0.0], 
  [0.0], 
  [0.0], 
  [0.0]])

measures = deque()
for t in range(n):
  Pt = np.matmul(matrix_power(A, t), initialState)
  Rt = Pt[0][0] + Pt[1][0] + Pt[2][0] + Pt[3][0]
  if(t % (n / 100) == 0): print(f'R({t}) = ', Rt)
  measures.append([t, Rt])
print("================================")

# print("measures = \n", measures)
stateHist = np.array(measures)
dfDistrHist = pd.DataFrame(stateHist, columns=list('xy'))

# print("stateHist = \n", stateHist)
# print("dfDistrHist = \n", dfDistrHist)


mttf = np.trapz(dfDistrHist['y']) * deltaT
print("MTTF = ", mttf)

dfDistrHist.plot(x='x', y='y', label="P(t)")
plt.title('P(nΔt) Plot')
plt.xlabel('tempo(h)')
plt.ylabel('confiabilidade')
plt.show()
