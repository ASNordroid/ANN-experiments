import numpy as np


# Sigmoid activation function
def nonlin(x, deriv=False):
    if deriv:
        return x * (1 - x)

    return 1 / (1 + np.exp(-x))


X = np.array([[0, 0, 1],
              [0, 1, 1],
              [1, 0, 1],
              [1, 1, 1]])

y = np.array([[0],
              [1],
              [1],
              [0]])

np.random.seed(1)

weights0 = 2 * np.random.random((3, 4)) - 1
weights1 = 2 * np.random.random((4, 1)) - 1

# Train the network
for j in range(100000):
    layer0 = X
    layer1 = nonlin(np.dot(layer0, weights0))
    layer2 = nonlin(np.dot(layer1, weights1))

    # Calculate the error
    layer2_error = y - layer2
    if j % 10000 == 0:
        print('Error:' + str(np.mean(np.abs(layer2_error))))

    # Back propagation of errors using chain rule
    layer2_delta = layer2_error * nonlin(layer2, deriv=True)
    layer1_error = layer2_delta.dot(weights1.T)
    layer1_delta = layer1_error * nonlin(layer1, deriv=True)

    # Using the deltas, we can update the weights to reduce
    # the error rate with every iteration (gradient decent)
    weights1 += layer1.T.dot(layer2_delta) * .1 # .1 is our learning rate
    weights0 += layer0.T.dot(layer1_delta) * .1

print(layer2)
