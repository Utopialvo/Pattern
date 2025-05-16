import pandas as pd
import numpy as np

def transform_edgelist(df):
    
    if not isinstance(df, pd.DataFrame):
        return df
        
    if df.shape[1] != 3:
        return df
        
    col1, col2, metric_col = df.columns
    if ('_' not in col1 or '_' not in col2) and df[metric_col].dtype != float:
        return df
    
    nodes = pd.unique(pd.concat([df[col1], df[col2]]))
    nodes = sorted(nodes)
    node_to_idx = {node: i for i, node in enumerate(nodes)}
    
    a_indices = df[col1].map(node_to_idx).values
    b_indices = df[col2].map(node_to_idx).values
    values = df[metric_col].values
    
    n = len(nodes)
    matrix = np.zeros((n, n))
    matrix[a_indices, b_indices] = values
    matrix[b_indices, a_indices] = values
    np.fill_diagonal(matrix, 0)
    
    return pd.DataFrame(matrix, index=nodes, columns=nodes)