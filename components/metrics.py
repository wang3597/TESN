import numpy as np


def getMetircs(preds, test_labels):
    np.set_printoptions(suppress=True)
    n = len(test_labels[0])
    sigma = np.var(test_labels[0])

    mse = np.sum(np.square(test_labels[0] - preds[0])) / n
    rmse = np.sqrt(np.sum(np.square(test_labels[0] - preds[0])) / n)
    nmse = np.sum(np.square(test_labels[0] - preds[0])) / (n * sigma)
    nrmse = np.sqrt(np.sum(np.square(test_labels[0] - preds[0])) / (n * sigma))
    mape = np.mean(np.abs((preds[0] - test_labels[0]) / test_labels[0]))
    smape = (np.sum(np.abs(preds[0] - test_labels[0]) / ((np.abs(preds[0]) + np.abs(test_labels[0])) / 2))) / n

    return [mse, rmse, nmse, nrmse, mape, smape]