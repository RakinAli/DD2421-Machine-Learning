import numpy
import random
import math
from scipy.optimize import minimize
import matplotlib.pyplot as plt

numpy.random.seed(100)
classA = numpy.concatenate(
    (numpy.random.randn(25, 2)*0.2+[1.5, 0.5],
     numpy.random.randn(25, 2)*0.2+[-1.5, 0.5])
)
classB = numpy.random.randn(20, 2)*0.2+[0.0, -0.5]

def generate_data(classA, classB):
    inputs = numpy.concatenate((classA, classB))
    targets = numpy.concatenate(
        (numpy.ones(classA.shape[0]),
        -numpy.ones(classB.shape[0]))
    )

    N = inputs.shape[0]  # Number of rows (samples)
    permute = list(range(N))
    random.shuffle(permute)
    inputs = inputs[permute, :]  
    targets = targets[permute]  
    return inputs, targets, N 

inputs, targets, N = generate_data(classA, classB)

# Kernel functions
def kernel_linear(x1, x2):
    return numpy.dot(x1, x2)

def kernel_poly(x1, x2, p=2):
    return (numpy.dot(x1, x2) + 1)**p

def kernel_radial(x1, x2, sigma=4):
    return math.exp(-numpy.linalg.norm(x1-x2)**2/(2*sigma**2))

# Gram matrix
def calculate_gram_matrix(kernel, inputs, targets):
    P = numpy.zeros((N, N))
    for i in range(N):
        for j in range(N):
            P[i, j] = targets[i]*targets[j]*kernel(inputs[i], inputs[j])
    return P

P = calculate_gram_matrix(kernel_linear, inputs, targets)

# Objective function
def objective(alpha):
    return 0.5*numpy.dot(alpha, numpy.dot(alpha, P)) - numpy.sum(alpha)

def zerofun(alpha):
    return numpy.dot(alpha, targets)

C = None
ret = minimize(objective, numpy.zeros(N), bounds=[
                      (0, C) for b in range(N)], constraints={'type': 'eq', 'fun': zerofun}, options={'disp': False, 'maxiter': 1000})


# add maxiter(iterations) to minimize function to avoid warning message for maximum iterations reached

# check if the optimization was successful
if (ret['success'] == False):
    print("Error in optimization")
    print(ret['message'])
    exit()
else:
    print("Optimization success")
    print(ret['message'])

alpha = ret['x']



def non_zeros(alpha, inputs, targets):
  alpha_notzero = []
  input_notzero = []
  target_notzero = []
  for i in range(len(alpha)):
    if alpha[i] > 10e-5:
      alpha_notzero.append(alpha[i])
      input_notzero.append(inputs[i])
      target_notzero.append(targets[i])
  return alpha_notzero, input_notzero, target_notzero

# Get non-zero alphas
alpha_notzero, input_notzero, target_notzero = non_zeros(alpha, inputs, targets)

# Calculate b
def calculate_bias(alpha, inputs, targets,  input_notzero, target_notzero):
    sv = input_notzero[0]
    t = target_notzero[0]
    b = 0
    for i in range(len(alpha)):
        b += alpha[i]*targets[i]*kernel_linear(sv, inputs[i])
    b = b - t
    return b

bias = calculate_bias(alpha, inputs, targets, input_notzero, target_notzero)



def indicator(x):
    ind = 0
    for i in range(len(alpha_notzero)):
        ind += alpha_notzero[i]*target_notzero[i]*kernel_linear(x, input_notzero[i])
    ind -= bias
    return ind


# Plot the data
plt.plot([p[0] for p in classA], [p[1] for p in classA], 'b.')
plt.plot([p[0] for p in classB], [p[1] for p in classB], 'r.')
plt.axis('equal')  # Force same scale on both axes


# Plot the decision boundary
xgrid = numpy.linspace(-5, 5)
ygrid = numpy.linspace(-4, 4)
grid = numpy.array([[indicator([x, y]) for x in xgrid] for y in ygrid])
plt.contour(xgrid, ygrid, grid, (-1.0, 0.0, 1.0),
            colors=('red', 'black', 'blue'), linewidths=(1, 3, 1))
plt.savefig('./bilder/cluster2.png')  # Save a copy in a file
plt.show()  # Show the plot on the screen
