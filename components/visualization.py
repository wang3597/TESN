import matplotlib.pyplot as plt


def visualize(preds, test_labels):
    plt.figure(0).clear()
    plt.plot(test_labels.T, 'b')
    plt.plot(preds.T, 'r')
    plt.legend(['Target', 'Predict'])
    plt.show()
