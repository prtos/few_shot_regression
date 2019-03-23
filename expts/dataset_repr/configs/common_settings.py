import os
import copy
from sklearn.model_selection import ParameterGrid

test = True

n_epochs, steps_per_epoch, max_tasks = 100, 500, None
if test:
    n_epochs, steps_per_epoch, max_tasks = 1, 5, 10
