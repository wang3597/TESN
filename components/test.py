import numpy as np
from components.reduceDim import reduceDimension


def test(args, test_data, Win, W, Wout, x):
    # run the trained ESN in a generative mode. no need to initialize here,
    # because x is initialized with training data and we continue from there.

    X_test = np.zeros((1 + args.in_size + args.reservoir_size, args.test_length))

    Y = np.zeros((args.out_size, args.test_length))

    for t in range(args.test_length):
        u = test_data[:, t]
        x = (1 - args.leaking_rate) * x + args.leaking_rate * np.tanh(np.dot(Win, np.vstack((1, u))) + np.dot(W, x))

        if args.useRD:
            X_test[:, t] = np.vstack((1, u, x))[:, 0]
        else:
            y = np.dot(Wout, np.vstack((1, u, x)))
            Y[:, t] = y

    if args.useRD:
        X_test = reduceDimension(args=args, X=X_test)
        Y = np.dot(Wout, X_test)

    return Y
