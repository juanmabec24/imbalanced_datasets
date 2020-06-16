import sklearn
import sklearn.datasets as datasets
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# define dataset
X, y = datasets.make_classification(n_samples=100000, n_features=4, n_redundant=0,
                           n_clusters_per_class=2, weights=[0.99], flip_y=0, random_state=42)
                           
                           
                           

# summarize class distribution: how many samples do I have for each class?
from collections import Counter
counter = Counter(y)
print(counter)


# Visualize the dataset
# scatter plot of examples by class label
plt.figure(figsize=(8, 8))

for label, _ in counter.items():
    row_ix = np.where(y == label)[0]
    plt.scatter(X[row_ix, 0], X[row_ix, 1], label=str(label))
    plt.legend()
    print('Type {} : {} points'.format(label, len(X[row_ix, 0])))
plt.savefig('Scatter_points.png', dpi=100)
