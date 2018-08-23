import matplotlib.pyplot as plt
import numpy as np
import itertools


def plot_confusion_matrix(cm, x_names, y_names,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues,
                          saving_filename=None):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.figure()

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    plt.xticks(np.arange(cm.shape[1]), x_names, rotation=45)
    plt.yticks(np.arange(cm.shape[1]), y_names)
    # fmt = '{value:.2f}'
    # for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    #     plt.text(j, i, fmt.format(value=cm[i, j]),
    #              horizontalalignment="center",)

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    if saving_filename:
        plt.savefig(saving_filename)
    else:
        plt.show()
    plt.close()