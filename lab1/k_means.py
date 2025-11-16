import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = np.loadtxt("../spiralka.csv", delimiter=',')

samples = data[np.random.choice(data.shape[0], 3, replace=False)]
print(f"{data}\n")
print(data[0])