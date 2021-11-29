import numpy as np
from numpy.linalg import matrix_power
import matplotlib.pyplot as plt

exercise = 'conf' # 'disp' or 'conf'

deltaT = 1
l = 0.005
muC = 0.05
muP = 0.00
cArray = [0.6, 0.7, 0.8, 0.9, 1]
n = 50000

ax = plt.gca()

initialState = np.array([
  [1.0], 
  [0.0], 
  [0.0], 
  [0.0], 
  [0.0]])

print("================================")
for c in cArray:
  print('C = ' + str(c))
  if (exercise == 'disp'):
    A = np.array([
      [1-2*l*deltaT, muC*deltaT, muC*deltaT, muP*deltaT, muC*deltaT],
      [2*l*deltaT*c, 1 - (muC + l)*deltaT, 0.0, 0.0, 0.0],
      [0.0, l*deltaT*c, 1 - (muC + l)*deltaT, l*deltaT*c, 0.0],
      [l*deltaT*(1 - c), 0.0, 0.0, 1 - (muP + l)*deltaT, 0.0],
      [l*deltaT*(1 - c), l*deltaT*(1 - c), l*deltaT, l*deltaT*(1 - c), 1 - muC*deltaT]
    ])
  elif (exercise == 'conf'):
    A = np.array([
      [1-2*l*deltaT, muC*deltaT, muC*deltaT, muP*deltaT, 0.0],
      [2*l*deltaT*c, 1 - (muC + l)*deltaT, 0.0, 0.0, 0.0],
      [0.0, l*deltaT*c, 1 - (muC + l)*deltaT, l*deltaT*c, 0.0],
      [l*deltaT*(1 - c), 0.0, 0.0, 1 - (muP + l)*deltaT, 0.0],
      [l*deltaT*(1 - c), l*deltaT*(1 - c), l*deltaT, l*deltaT*(1 - c), 1]
    ])
  print("A = \n", A)

  time = []
  pOp = []
  for t in range(n):
    Pt = np.matmul(matrix_power(A, t), initialState)
    Rfailure = Pt[4][0]
    time.append(t*deltaT)
    pOp.append(1-Rfailure)

  print('Passintotica = ' + str(1-Rfailure))
  ax.plot(time, pOp, label="C=" + str(c))
  mttf = np.trapz(pOp, dx=deltaT)
  print("Area = ", mttf)
  print("================================")

ax.set_title('Gráfico P(nΔt)')
ax.set_xlabel('tempo(h)')
ax.set_ylabel('confiabilidade')
ax.legend()
plt.show()
