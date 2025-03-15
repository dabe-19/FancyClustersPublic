# FancyClusters

A Python package for enhanced clustering with pandas and NumPy integration using scikit-learn's AgglomerativeClustering.

## Description

`FancyClusters` simplifies the process of clustering data using scikit-learn's `AgglomerativeClustering` algorithm while preserving your original data structure. It handles both pandas DataFrames and NumPy ndarrays, automatically selects numerical columns, and adds cluster labels to your data.

## Installation

```bash
pip install fancyclusters
```

```python
from fancyclusters import FancyClusters
import pandas as pd
import numpy as np

# Example with pandas DataFrame
data_pd = pd.DataFrame({
    'feature1': [1, 2, 3, 4, 5],
    'feature2': [5, 4, 3, 2, 1],
    'category': ['A', 'B', 'A', 'C', 'B']
})

fancy_clusters_pd = FancyClusters(n_clusters=2)
result_pd = fancy_clusters_pd.fit_predict(data_pd)
print("Clustered DataFrame:\n", result_pd)

cluster_groups_pd = fancy_clusters_pd.get_cluster_groups()
print("\nCluster Groups (DataFrame):\n", cluster_groups_pd)

# Example with NumPy ndarray
data_np = np.array([[1, 5, 'A'], [2, 4, 'B'], [3, 3, 'A'], [4, 2, 'C'], [5, 1, 'B']])

fancy_clusters_np = FancyClusters(n_clusters=2)
result_np = fancy_clusters_np.fit_predict(data_np)
print("\nClustered ndarray:\n", result_np)

cluster_groups_np = fancy_clusters_np.get_cluster_groups()
print("\nCluster Groups (ndarray):\n", cluster_groups_np)

#Using the fit() method
fancy_clusters_np_fit = FancyClusters(n_clusters=2)
model, result_np_fit = fancy_clusters_np_fit.fit(data_np)
print("\nClustered ndarray using fit():\n", result_np_fit)
print("\nAgglomerativeClustering Model:\n", model)
```
## Methods
__init__(n_clusters=2, **kwargs): Initializes the FancyClusters object.
n_clusters: The number of clusters.
**kwargs: Additional keyword arguments passed to AgglomerativeClustering.
fit_predict(data): Fits the model and predicts cluster labels.
data: pandas DataFrame or NumPy ndarray.
Returns: Clustered data with a 'cluster' column.
fit(data): Fits the model and returns the fitted model and clustered data.
data: pandas DataFrame or NumPy ndarray.
Returns: The fitted AgglomerativeClustering model and the clustered data with a 'cluster' column.
get_cluster_groups(): Returns a list of DataFrames or arrays, one for each cluster.
Returns: List of grouped data.

## Error Handling
Raises ValueError if the input data is not a pandas DataFrame or NumPy ndarray.
Raises ValueError if the input data contains no numerical columns.
Raises ValueError if get_cluster_groups() is called before fit() or fit_predict().

## Dependencies
pandas
NumPy
scikit-learn

## License
MIT License

# Author
dabe-19

## Contributing
... (Add contribution guidelines here)

