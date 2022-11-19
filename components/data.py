import os
import numpy as np


def loadData(args):

    if args.dataset.lower() == 'lorenz':
        data_file = os.path.join(args.data_path, "lorenz.txt")
        data = np.loadtxt(data_file)
        data = data[np.newaxis, :]

        args.train_length = 6999
        args.test_length = 3000
        args.init_length = 300

    elif args.dataset.lower() == 'sunspots':
        data_file = os.path.join(args.data_path, "Sunspots.txt")
        data = np.loadtxt(data_file)
        data = data[np.newaxis, :]

        args.train_length = 2264
        args.test_length = 1000
        args.init_length = 100

    else:
        data = None

    return args, data


def splitData(args, data):
    train_data = data[:, 0:args.train_length]
    train_labels = data[:, 1:1 + args.train_length]
    test_data = data[:, args.train_length:args.train_length + args.test_length]
    test_labels = data[:, 1 + args.train_length:1 + args.train_length + args.test_length]

    return train_data, train_labels, test_data, test_labels