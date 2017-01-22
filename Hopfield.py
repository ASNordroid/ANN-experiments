import numpy as np

# Learning patterns
A = np.array([-1, 1, 1, 1,-1,
               1,-1,-1,-1, 1,
               1, 1, 1, 1, 1,
               1,-1,-1,-1, 1,
               1,-1,-1,-1, 1])
B = np.array([1, 1, 1, 1, 1,
              1,-1,-1,-1, 1,
              1, 1, 1, 1, 1,
              1,-1,-1,-1, 1,
              1, 1, 1, 1, 1])
C = np.array([1, 1, 1, 1, 1,
              1,-1,-1,-1,-1,
              1,-1,-1,-1,-1,
              1,-1,-1,-1,-1,
              1, 1, 1, 1, 1])


# Print pattern in easy readable form
def print_pattern(pattern):
    for i in range(0, len(pattern), 5):
        for j in range(i, i+5):
            if pattern[j] == 1:
                print("X", end="")
            else:
                print("_", end="")
        print()


def sign(i):
    if i >= 0:
        return 1
    else:
        return -1


def train(pts):
    r, c = pts.shape
    W = np.zeros((c, c))

    for p in pts:
        W = W + np.outer(p, p) # Weight matrix
    np.fill_diagonal(W, 0)

    return W / r


def remember(W, pts, steps=5):
    for _ in range(steps):
        pts = np.vectorize(sign)(np.dot(pts, W))
    
    return pts


patterns = np.array([A, B, C])

# To recognize
y = np.array([-1, 1, 1, 1,-1,
               1,-1, 1, 1,-1,
               1,-1,-1,-1, 1,
               1,-1, 1,-1, 1,
               1,-1,-1,-1, 1])

print_pattern(y)
print_pattern(remember(train(patterns), y))