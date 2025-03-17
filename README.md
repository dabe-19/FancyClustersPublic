# FancyClusters

A Python package for enhanced clustering with pandas and NumPy integration using scikit-learn's AgglomerativeClustering.

## Description

`FancyClusters` simplifies the process of clustering data using scikit-learn's `AgglomerativeClustering` algorithm while preserving your original data structure. It handles both pandas DataFrames and NumPy ndarrays, automatically selects numerical columns, and adds cluster labels to your data. Will detect if ndarray contains mixed data types and will convert ndarray to Pandas DataFrame in this case. If all data types are numerics, ndarray type will be preserved. Optional "convert" argument to attempt and coerce all columns to numerics if in DataFrame format or if ndarray contains mixed data.

## Installation

```bash
pip install git+https://github.com/dabe-19/FancyClustersPublic.git#egg=fancyclusters
```

```python
import pandas as pd
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from fancyclusters import FancyClusters
```
```python
# Example with pandas DataFrame
data_pd = pd.DataFrame({
    'feature1': [1, 2, 3, 4, 5],
    'feature2': [5, 4, 3, 2, 1],
    'category': ['A', 'B', 'A', 'C', 'B']
})

# Create an instance of FancyClusters
fancy_clusters_pd = FancyClusters(n_clusters=2, linkage = "ward")
result_pd = fancy_clusters_pd.fit_predict(data_pd)
# Test fit_predict with NumPy array
cluster_groups_pd = fancy_clusters_pd.get_cluster_groups()
for i in range(fancy_clusters_pd.n_clusters):
    print(cluster_groups_pd[i].value_counts())
```

```python
# Example with NumPy ndarray mixed types
data_np = np.array([[1, 5, 'A'], [2, 4, 'B'], [3, 3, 'A'], [4, 2, 'C'], [5, 1, 'B']])
# Create a new instance of FancyClusters to test fit.
fancy_clusters_np = FancyClusters(n_clusters=3, linkage = "ward")
result_np = fancy_clusters_np.fit_predict(data_np, convert = True)
# Test fit_predict with NumPy array
cluster_groups_np = fancy_clusters_np.get_cluster_groups()
for i in range(fancy_clusters_np.n_clusters):
    print(cluster_groups_np[i].value_counts())
```

```python
# Example with NumPy ndarray numerical types
data_np2 = np.array([[1, 5, 3], [2, 4, 6], [3, 3, 13], [4, 2, 2], [5, 1, 5]])
fancy_clusters_np2 = FancyClusters(n_clusters=3, linkage = "ward")
result_np2 = fancy_clusters_np2.fit_predict(data_np2)
# Test fit_predict with NumPy array
cluster_groups_np2 = fancy_clusters_np2.get_cluster_groups()
cluster_groups_np2
```

```python
# Example with pandas DataFrame
data_pd2 = pd.DataFrame({
    'feature1': [1, 2, 3, 4, 5],
    'feature2': [5, 4, 3, 2, 1],
    'category': ['V', 'W', 'X', 'Y', 'Z']
})

# Create an instance of FancyClusters
fancy_clusters_pd2 = FancyClusters(n_clusters=2, linkage = "ward")
result_pd2 = fancy_clusters_pd2.fit(data_pd2)
# Test fit_predict with NumPy array
cluster_groups_pd2 = fancy_clusters_pd2.get_cluster_groups()
for i in range(fancy_clusters_pd2.n_clusters):
    print(cluster_groups_pd2[i].value_counts())
```
```python
# Example with NumPy ndarray mixed types
data_np3 = np.array([[1, 5, 'A'], [2, 4, 'B'], [3, 3, 'A'], [4, 2, 'C'], [5, 1, 'B']])
# Create a new instance of FancyClusters to test fit.
fancy_clusters_np3 = FancyClusters(n_clusters=3, linkage = "ward")
result_np3 = fancy_clusters_np3.fit(data_np3, convert = True)
# Test fit with NumPy array
cluster_groups_np3 = fancy_clusters_np3.get_cluster_groups()
for i in range(fancy_clusters_np3.n_clusters):
    print(cluster_groups_np3[i].value_counts())
```
```python
data_np4 = np.array([[1, 5, 3], [2, 4, 6], [3, 3, 13], [4, 2, 2], [5, 1, 5]])
fancy_clusters_np4 = FancyClusters(n_clusters=3, linkage = "ward")
result_np4 = fancy_clusters_np4.fit(data_np4)
# Test fit with NumPy array
cluster_groups_np4 = fancy_clusters_np4.get_cluster_groups()
cluster_groups_np4
```
## Methods
`__init__(n_clusters=2, **kwargs)`: Initializes the FancyClusters object.  
n_clusters: The number of clusters.  
**kwargs: Additional keyword arguments passed to AgglomerativeClustering.  
  
`fit_predict(data, convert=False)`: Fits the model and predicts cluster labels.  
data: pandas DataFrame or NumPy ndarray.
convert: default=False, will attempt to convert columns to numeric types if not already. Preserves columns not able to be converted  
Returns: Clustered data with a 'cluster' column.  
  
`fit(data, convert=False)`: Fits the model and returns the fitted model and clustered data.  
data: pandas DataFrame or NumPy ndarray.  
convert: default=False, will attempt to convert columns to numeric types if not already. Preserves columns not able to be converted  
Returns: The fitted AgglomerativeClustering model and the clustered data with a 'cluster' column.  
  
`get_cluster_groups()`: Returns a list of DataFrames or arrays, one for each cluster. Must run either fit() or fit_predict() first
Returns: List of grouped cluster data.  

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


