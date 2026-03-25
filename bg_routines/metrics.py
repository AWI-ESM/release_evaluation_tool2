"""
Centralized metric functions for model evaluation.
Provides NaN-aware and weighted statistical metrics.
"""

import numpy as np


def rmsd(predictions, targets):
    """Calculate Root Mean Square Deviation (RMSD).
    
    Parameters
    ----------
    predictions : np.ndarray
        Model predictions
    targets : np.ndarray
        Target/reference values
        
    Returns
    -------
    float
        Root mean square deviation
    """
    return np.sqrt(((predictions - targets) ** 2).mean())


def md(predictions, targets, wgts=None):
    """Mean Deviation (bias), optionally weighted, NaN-aware.
    
    Parameters
    ----------
    predictions : np.ndarray
        Model predictions
    targets : np.ndarray
        Target/reference values
    wgts : np.ndarray, optional
        Weights for averaging. If None, uses simple mean.
        
    Returns
    -------
    float
        Mean deviation (bias)
    """
    diff = predictions - targets
    
    if wgts is None:
        # Simple unweighted mean
        return diff.mean()
    
    # Weighted version with NaN handling
    valid = ~np.isnan(diff)
    if not np.any(valid):
        return np.nan
    return np.average(diff[valid], weights=wgts[valid])


def rmsd_weighted(predictions, targets, wgts):
    """Root Mean Square Deviation with weights, NaN-aware.
    
    Parameters
    ----------
    predictions : np.ndarray
        Model predictions
    targets : np.ndarray
        Target/reference values
    wgts : np.ndarray
        Weights for averaging
        
    Returns
    -------
    float
        Weighted root mean square deviation
    """
    diff = predictions - targets
    valid = ~np.isnan(diff)
    if not np.any(valid):
        return np.nan
    squared_diff = diff[valid] ** 2
    return np.sqrt(np.average(squared_diff, weights=wgts[valid]))
