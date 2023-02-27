import numpy
import random
import math
from scipy.optimize import minimize
import matplotlib.pyplot as plt

# Generate data
def generate_data():
  numpy.random.seed(100)
  classA = numpy.concatenate((numpy.random.randn(10, 2)* 0.2 + [1.5, 0.5], numpy.random.randn(10, 2) * 0.2 + [-1.5, 0.5]))
  classB = numpy.random.randn(20, 2) * 0.2 + [0.0, -0.5]

  inputs = numpy.concatenate((classA, classB))
  targets = numpy.concatenate((numpy.ones(classA.shape[0]), -numpy.ones(classB.shape[0])))
  N = inputs.shape[0] # Number of rows (samples)

  # Randomly reorders the samples.
  permute = list(range(N))
  random.shuffle(permute)
  inputs = inputs[permute, :]
  targets = targets[permute]

  return inputs, targets, N 

# A function that returns the kernel matrix. As a parameter, it takes the inputs, the targets, the dimension of the matrix and the kernel function.
def kernels(x,y, type):
  if type == 'linear':
    return numpy.dot(x, y)
  elif type == 'polynomial':
    return (1 + numpy.dot(x, y)) ** 3
  elif type == 'rbf':
    return math.exp(-numpy.linalg.norm(x-y)**2 / (2 * (5.0 ** 2)))
  else:
    return 0

# Calculates the P matrix --> 
def matrix(inputs, targets, dimension, kernel_type):
  P = numpy.zeros((dimension, dimension))
  for i in range(dimension):
    for j in range(dimension):
      P[i][j] = targets[i] * targets[j] * kernels(inputs[i], inputs[j], kernel_type)
  return P

# Calculates the objective function
def objective(alpha,P):
  return 0.5 * numpy.dot(alpha, numpy.dot(P, alpha)) - numpy.sum(alpha)

# Zerofun function
def zerofun(alpha, targets):
  return numpy.dot(alpha, targets)

def non_zeros(alpha, inputs, targets):
  alpha_notzero = []
  input_notzero= []
  target_notzero = []
  for i in range(len(alpha)):
    if alpha[i] > 10e-5:
      alpha_notzero.append(alpha[i])
      input_notzero.append(inputs[i])
      target_notzero.append(targets[i])
  return alpha_notzero, input_notzero, target_notzero


# Calculates Bias according to point (7) - Vi ändrade alpha not zero från input_notzero
def calculate_bias(alpha, inputs, input_notzero, targets, target_notzero):
  support_vector = input_notzero[0]
  support_vector_target = target_notzero[0] # Chooose any target non-zero
  sum = 0
  for i in range(len(alpha)):
    sum += alpha[i] * targets[i] * kernels(support_vector, inputs[i], 'linear')
  return sum - support_vector_target


# Make indicator function according to point (6)
def indicator(dataPoint, alpha_nonzeroes, input_nonzeros, targets_nonzeroes,bias):
  sum = 0
  for i in range(len(alpha_nonzeroes)):
    sum += alpha_nonzeroes[i] * targets_nonzeroes[i] * kernels(dataPoint, input_nonzeros[i], 'linear')
  answer = sum - bias
  return answer 


def main():
  classA = numpy.concatenate((numpy.random.randn(10, 2) * 0.2 + [1.5, 0.5], numpy.random.randn(10, 2) * 0.2 + [-1.5, 0.5]))
  classB = numpy.random.randn(20, 2) * 0.2 + [0.0, -0.5]

  # Generate data
  inputs, targets, N = generate_data()
  p_matrix = matrix(inputs, targets, N, 'linear')
  
  # Start optimization
  start = numpy.zeros(N)
  C = 1000
  B = [(0, C) for b in range(N)]
  XC = {'type':'eq', 'fun':zerofun, 'args':(targets,)}
  ret = minimize(objective, start, bounds=B, constraints=XC, args=(p_matrix))
  alpha = ret['x']

  # Get non-zero alphas
  alpha_notzero, input_notzero, target_notzero = non_zeros(alpha, inputs, targets)

  # Calculate bias
  bias = calculate_bias(alpha, inputs, input_notzero, targets, target_notzero)

  # Plot 
  plt.plot([p[0] for p in classA], [p[1] for p in classA], 'b.')
  plt.plot([p[0] for p in classB], [p[1] for p in classB], 'r.')
  plt.axis('equal')  # Force same scale on both axes
  

   # Plot the decision boundary
  xgrid = numpy.linspace(-5, 5)
  ygrid = numpy.linspace(-4, 4)
  grid = numpy.array([[indicator([x, y],alpha_notzero, input_notzero, target_notzero, bias) for x in xgrid] for y in ygrid])
  plt.contour(xgrid, ygrid, grid, (-1.0, 0.0, 1.0), colors=('red', 'black', 'blue'), linewidths=(1, 3, 1))
  plt.savefig('rakin.pdf')  # Save a copy in a file
  plt.show()  # Show the plot on the screen



# Main function
if __name__ == "__main__":
  main()


