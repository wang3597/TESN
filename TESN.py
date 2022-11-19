import numpy as np
from components.data import loadData, splitData
from components.train import train
from components.test import test
from components.metrics import getMetircs
from components.visualization import visualize

from arguments import esn_argparse

args = esn_argparse().parse_args()

if __name__ == "__main__":
    np.random.seed(args.seed)

    args, data = loadData(args)  # load data
    train_data, train_labels, test_data, test_labels = splitData(args=args, data=data)

    Win, W, Wout, x = train(args=args, train_data=train_data, train_labels=train_labels)
    Y = test(args=args, test_data=test_data, Win=Win, W=W, Wout=Wout, x=x)

    metrics = getMetircs(preds=Y, test_labels=test_labels)
    mse = metrics[0]
    rmse = metrics[1]
    nmse = metrics[2]
    nrmse = metrics[3]
    mape = metrics[4]
    smape = metrics[5]

    visualize(preds=Y, test_labels=test_labels)

