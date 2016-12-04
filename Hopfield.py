import numpy as np


def sign(i):
    if i >= 0:
        return 1
    else:
        return -1

v_sign = np.vectorize(sign)


def recognize(y1):
    y1 = np.dot(W, y1)
    y1 = v_sign(y1)
    return y1

# Training patterns
pattern_1 = np.array([[-1, 1, -1, 1, 1]])
pattern_2 = np.array([[1, -1, 1, 1, -1]])
pattern_3 = np.array([[-1, 1, -1, -1, 1]])
patterns = [pattern_1, pattern_2, pattern_3]

# To recognize
y = np.array([[-1, 1, -1, -1, 1]])

# Weight matrix)
c_1 = np.dot(pattern_1.T, pattern_1)
c_2 = np.dot(pattern_2.T, pattern_2)
c_3 = np.dot(pattern_3.T, pattern_3)
W = c_1 + c_2 + c_3
np.fill_diagonal(W, 0)

print(recognize(y.transpose()))