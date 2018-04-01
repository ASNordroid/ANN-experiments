import copy, numpy as np

np.random.seed(0)


def sigmoid(x, deriv=False):
    if deriv:
        return x*(1-x)

    return 1/(1+np.exp(-x))

# training dataset generation
int2bin = {}
bin_dim = 8

largest_num = pow(2, bin_dim)
binary = np.unpackbits(
    np.array([range(largest_num)], dtype=np.uint8).T, axis=1)
for i in range(largest_num):
    int2bin[i] = binary[i]

# input variables
alpha = 0.1
input_dim = 2
hidden_dim = 16
output_dim = 1

# initialize nn weights
synapse_0 = 2 * np.random.random((input_dim, hidden_dim))  - 1
synapse_1 = 2 * np.random.random((hidden_dim, output_dim)) - 1
synapse_h = 2 * np.random.random((hidden_dim, hidden_dim)) - 1

synapse_0_update = np.zeros_like(synapse_0)
synapse_1_update = np.zeros_like(synapse_1)
synapse_h_update = np.zeros_like(synapse_h)

# training logic
for j in range(10000):
    # generate simple addition problem: a + b = c
    a_int = np.random.randint(int(largest_num / 2))
    a = int2bin[a_int]
    b_int = np.random.randint(int(largest_num / 2))
    b = int2bin[b_int]
    c_int = a_int + b_int
    c = int2bin[c_int]

    # best guess in binary
    d = np.zeros_like(c)

    overallError = 0

    layer_2_deltas = []
    layer_1_values = [np.zeros(hidden_dim)]

    # moving along the positions in the binary encoding
    for pos in range(bin_dim):
        # gen input and output
        X = np.array([[a[bin_dim - pos - 1], b[bin_dim - pos - 1]]])
        y = np.array([[c[bin_dim - pos - 1]]]).T

        # hidden layer (input ~+ prev hidden)
        layer_1 = sigmoid(np.dot(X, synapse_0) + np.dot(layer_1_values[-1], synapse_h))

        # output layer (new binary representation)
        layer_2 = sigmoid(np.dot(layer_1, synapse_1))

        # error
        layer_2_error = y - layer_2
        layer_2_deltas.append(layer_2_error * sigmoid(layer_2, True))
        overallError += np.abs(layer_2_error[0])

        # decode estimate
        d[bin_dim - pos - 1] = np.round(layer_2[0][0])

        # store hidden layer, so we can use it the next time
        layer_1_values.append(copy.deepcopy(layer_1))

    future_layer_1_delta = np.zeros(hidden_dim)

    for pos in range(bin_dim):
        X = np.array([[a[pos], b[pos]]])
        layer_1 = layer_1_values[-pos - 1]
        prev_layer_1 = layer_1_values[-pos - 2]

        # error at output layer
        layer_2_delta = layer_2_deltas[-pos - 1]

        # error at hidden layer
        layer_1_delta = (future_layer_1_delta.dot(synapse_h.T) + layer_2_delta.dot(synapse_1.T)) * sigmoid(layer_1, True)

        # update all weights so we can try again
        synapse_1_update += np.atleast_2d(layer_1).T.dot(layer_2_delta)
        synapse_h_update += np.atleast_2d(prev_layer_1).T.dot(layer_1_delta)
        synapse_0_update += X.T.dot(layer_1_delta)

        future_layer_1_delta = layer_1_delta

    synapse_0 += synapse_0_update * alpha
    synapse_1 += synapse_1_update * alpha
    synapse_h += synapse_h_update * alpha

    synapse_0_update *= 0
    synapse_1_update *= 0
    synapse_h_update *= 0

    # print
    if j % 1000 == 0:
        print('Error: ' + str(overallError))
        print('Predicted: ' + str(d))
        print('True: ' + str(c))

        print(synapse_0)

        out = 0
        for index, x in enumerate(reversed(d)):
            out += x * pow(2, index)
        print(str(a_int) + ' + ' + str(b_int) + ' = ' + str(out))
        print('-' * 10)
