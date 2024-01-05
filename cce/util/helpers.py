
"""
General Helper Functions
"""

##################
### Imports
##################

## Standard Library
import hashlib

## Extrenal Libraries
import pandas as pd

##################
### Functions
##################

def flatten(l):
    """
    
    """
    lflat = [x for s in l for x in s] if l is not None else None
    return lflat

def chunks(l,
           n):
    """

    """
    for i in range(0, len(l), n):
        yield l[i:i + n]
    
def apply_hash(data):
    """
    
    """
    if data is None or pd.isnull(data):
        return None
    hashed_data = hashlib.sha256(str(data).encode("utf-8")).hexdigest()
    return hashed_data