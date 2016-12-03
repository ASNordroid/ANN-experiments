import numpy as np


def sign(i):
    if i >= 0:
        return 1
    else:
        return -1


def train(matrix):
    y2 = np.multiply(W, y2)
    W += matrix

    return W


def recognize(vector, resvector):
    for i in range(1, 2):
        for j in range(1, 2):
            resvector[i] += W * vector[j]

image1 = np.matrix('-1, 1, -1, 1')
timage1 = image1.transpose()
image2 = np.matrix('1, -1, 1, 1')
timage2 = image2.transpose()
image3 = np.matrix('-1, 1, -1, -1')
timage3 = image3.transpose()

y = np.matrix('1, -1, 1, -1')

b1 = np.multiply(image1, timage1)
b2 = np.multiply(image2, timage3)
b3 = np.multiply(image3, timage3)

#print(b1)
#print(b2)
#print(b3)

W = b1 + b2 + b3

for i in range(1, 4):
    for j in range(1, 4):
        if i == j:
            W[i, j] = 0

y1 = np.multiply(W, y.transpose())

print(W)
print(y1)
#train(y1)
