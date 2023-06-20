"""
-------------------------------
| Dartmouth College           |
| RL 4 Wildfire Containment   |
| 2023                        |
| Spencer Bertsch             |
-------------------------------

This script contains utility functions used through out this project. 
"""

# imports 
import numpy as np

def numpy_element_counter(arr: np.array) -> dict:
    """
    Utility function to return a dict containing the counts for elements in np array
    """
    unique, counts = np.unique(arr, return_counts=True)
    counts_dict: dict = dict(zip(unique, counts))
    return counts_dict