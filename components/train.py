import numpy as np
from scipy import linalg, sparse
from components.reduceDim import reduceDimension


def train(args, train_data, train_labels):

    """generate the ESN reservoir"""
    Win = (np.random.rand(args.reservoir_size, 1 + args.in_size) - 0.5) * 1
    Win = Win * args.input_scale

    W = sparse.rand(m=args.reservoir_size, n=args.reservoir_size, density=1 - args.sparse_degree, format='csc',
                    dtype=None, random_state=args.seed).toarray()
    W[np.where(W != 0)] -= 0.5

    """normalizing and setting spectral radius (correct, slow)"""
    maxEigenvalue = max(abs(np.linalg.eig(W)[0]))
    W = W * args.spectral_radius / maxEigenvalue

    """allocated memory for the design (collected states) matrix"""
    X = np.zeros((1 + args.in_size + args.reservoir_size, args.train_length - args.init_length))

    """set the corresponding target matrix directly"""
    Yt = train_labels[:, args.init_length:args.train_length]

    """run the reservoir with the data and collect X"""
    x = np.zeros((args.reservoir_size, 1))  # (N=300, 1)
    for t in range(args.train_length):
        u = train_data[:, t]
        x = (1 - args.leaking_rate) * x + args.leaking_rate * np.tanh(np.dot(Win, np.vstack((1, u))) + np.dot(W, x))
        if t >= args.init_length:
            X[:, t - args.init_length] = np.vstack((1, u, x))[:, 0]

    reg = 1e-8  # regularization coefficient

    """reduce dimension of X"""
    if args.useRD:
        X = reduceDimension(args=args, X=X)
        Wout = linalg.solve(np.dot(X, X.T) + reg * np.eye(1 + args.in_size + args.target_size), np.dot(X, Yt.T)).T

    else:
        Wout = linalg.solve(np.dot(X, X.T) + reg * np.eye(1 + args.in_size + args.reservoir_size), np.dot(X, Yt.T)).T

    return Win, W, Wout, x