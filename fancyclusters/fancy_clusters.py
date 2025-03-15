# pip install --upgrade git+https://github.com/dabe-19/FancyClusters.git#egg=fancyclusters
import pandas as pd
import numpy as np
from sklearn.cluster import AgglomerativeClustering

class FancyClusters:
    def __init__(self, n_clusters=2, **kwargs):
        self.n_clusters = n_clusters
        self.clustering_model = AgglomerativeClustering(n_clusters=n_clusters, **kwargs) # create agglomerative clustering model
        self.original_data = None # instantiate original data parameter for storing original dataframe or array
        self.cluster_labels = None # instantiate cluster_labels parameter  
        self.cluster_groups = [] # instantiate cluster_groups list
        self.clustered_data = None
        self.unconverted = []
    def fit_predict(self, data, convert=False):
        
        self.original_data = data # Capture original data
        #checks if data is input is pandas DataFrame or numpy ndarray, returns an error if data is neither type.
        if isinstance(data, pd.DataFrame):
            if convert: # checks if convert argument is true and will try to convert all columns to numeric
                        # raise a warning if a column could not be converted and append that column to a list for troubleshooting
                for col in data.columns:
                    try:
                        data[col] = pd.to_numeric(data[col])
                    except ValueError:
                        self.unconverted.append(col)
                        print(f'Warning: Failed to convert column {col} to numeric')
                        pass
            numerical_cols = data.select_dtypes(include = np.number).columns # finds which columns in DataFrame are numeric
            if len(numerical_cols) == 0: # check for existence of numeric columns
                raise ValueError("Pandas Dataframe contains no numeric columns")
            numerical_data = data[numerical_cols].values # creates array out of numerical columns
        elif isinstance(data, np.ndarray): # checks if data is supplied as ndarray
            data = pd.DataFrame(data) # convert ndarray to DataFrame to check for numeric columns
            if convert:
                for col in data.columns:
                    try:
                        data[col] = pd.to_numeric(data[col])
                    except ValueError:
                        self.unconverted.append(col)
                        print(f'Warning: Failed to convert column {col} to numeric')
                        pass
            numerical_cols = data.select_dtypes(include=np.number).columns
            if len(numerical_cols) == 0: # Checks for existence of numeric columns
                raise ValueError("ndarray contains no numeric columns.")
            numerical_data = data 
        else:
            raise ValueError("Input data must be either Pandas DataFrame or Numpy ndarray.")
        
        self.cluster_labels = self.clustering_model.fit_predict(numerical_data)
        
        if isinstance(self.original_data, pd.DataFrame): # if original data is DataFrame, re-apply column names and add "cluster" column
            result = self.original_data.copy()
            result['cluster'] = self.cluster_labels
        else: # if original data is ndarray, keep as ndarray but add cluster labels column to end of array
            result = np.concatenate([self.original_data, self.cluster_labels.reshape(-1,1)], axis = 1)
        self.clustered_data = result
        return(result)
    
    def fit(self, data, convert=False):
        self.original_data = data # Capture original data
        #checks if data is input is pandas DataFrame or numpy ndarray, returns an error if data is neither type.
        if isinstance(data, pd.DataFrame):
            if convert: # checks if convert argument is true and will try to convert all columns to numeric
                        # raise a warning if a column could not be converted and append that column to a list for troubleshooting
                for col in data.columns:
                    try:
                        data[col] = pd.to_numeric(data[col])
                    except ValueError:
                        self.unconverted.append(col)
                        print(f'Warning: Failed to convert column {col} to numeric')
                        pass            
            numerical_cols = data.select_dtypes(include = np.number).columns # finds which columns in DataFrame are numeric
            if len(numerical_cols) == 0: # check for existence of numeric columns
                raise ValueError("Pandas Dataframe contains no numeric columns")
            numerical_data = data[numerical_cols].values # creates array out of numerical columns
        elif isinstance(data, np.ndarray): # checks if data is supplied as ndarray
            data = pd.DataFrame(data) # convert ndarray to DataFrame to check for numeric columns
            if convert:
                for col in data.columns:
                    try:
                        data[col] = pd.to_numeric(data[col])
                    except ValueError:
                        self.unconverted.append(col)
                        print(f'Warning: Failed to convert column {col} to numeric')
                        pass
            numerical_cols = data.select_dtypes(include=np.number).columns
            if len(numerical_cols) == 0: # Checks for existence of numeric columns
                raise ValueError("ndarray contains no numeric columns.")
            numerical_data = data 
        else:
            raise ValueError("Input data must be either Pandas DataFrame or Numpy ndarray.")
        clusMdl = self.clustering_model.fit(numerical_data)
        self.cluster_labels = clusMdl.labels_        
        if isinstance(self.original_data, pd.DataFrame): # if original data is DataFrame, re-apply column names and add "cluster" column
            result = self.original_data.copy()
            result['cluster'] = self.cluster_labels
        else: # if original data is ndarray, keep as ndarray but add cluster labels column to end of array
            result = np.concatenate([self.original_data, self.cluster_labels.reshape(-1,1)], axis = 1)
        self.clustered_data = result
        return(clusMdl, result)
    
    def get_cluster_groups(self):
        if self.clustered_data is None:
            raise ValueError("Must call fit() or fit_predict() methods first.")
        if isinstance(self.clustered_data, pd.DataFrame):
            for i in range(self.n_clusters):
                self.cluster_groups.append(self.clustered_data[self.clustered_data['cluster'] == i])
        else: 
            for i in range(self.n_clusters):
                self.cluster_groups.append(self.clustered_data[self.clustered_data[:,-1] == i])
        return(self.cluster_groups) 