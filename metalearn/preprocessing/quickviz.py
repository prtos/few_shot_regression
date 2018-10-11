import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from itertools import combinations


def heatmap(cv_results, title="", prefix_filename="heatmap"):

    cv_params_keys = [el for el in cv_results.columns if el.startswith("param_") and len(set(cv_results[el].tolist())) > 1]
    # print cv_params_keys
    for (a, b) in combinations(cv_params_keys, 2):
        if a > b:
            a, b = b, a
        df = pd.pivot_table(cv_results, index=[a], columns=[b],
                            values=["mean_test_score"], aggfunc=np.mean)
        plt.figure(figsize=(8, 6))
        plt.subplots_adjust(left=.2, right=0.95, bottom=0.15, top=0.95)
        plt.imshow(df, interpolation='nearest')
        x_name = str(df.columns.levels[1].name)[6:]
        y_name = str(df.index.name)[6:]
        x_ticks = df.columns.levels[1].values.tolist()
        if type(x_ticks[0]) == float:
            x_ticks = np.round(x_ticks, 3)
        y_ticks = df.index.values.tolist()
        if type(y_ticks[0]) == float:
            y_ticks = np.round(y_ticks, 3)
        plt.xlabel(x_name)
        plt.ylabel(y_name)
        plt.colorbar()
        plt.xticks(np.arange(len(x_ticks)), x_ticks, rotation=45)
        plt.yticks(np.arange(len(y_ticks)), y_ticks)
        plt.title(title)

        plt.savefig("{}_{}_{}.png".format(prefix_filename, x_name, y_name))
        plt.close()