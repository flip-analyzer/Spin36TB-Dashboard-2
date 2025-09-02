import numpy as np
import pandas as pd
from typing import Optional, Union
from numba import jit
from scipy.special import gamma


class FractionalDifferentiator:
    """
    Fractional Differentiation implementation following López de Prado's methodology.
    
    This addresses the trade-off between stationarity and memory preservation
    in financial time series.
    """
    
    def __init__(self, d: float = 0.5, threshold: float = 1e-5):
        """
        Parameters:
        -----------
        d : float
            Fractional differentiation parameter (0 < d < 1)
        threshold : float
            Threshold for weight truncation
        """
        self.d = d
        self.threshold = threshold
        self.weights = None
        
    def get_weights(self, d: float, size: int) -> np.ndarray:
        """Calculate fractional differentiation weights"""
        w = [1.0]
        for k in range(1, size):
            w_ = -w[-1] * (d - k + 1) / k
            w.append(w_)
        w = np.array(w[::-1]).reshape(-1, 1)
        return w
    
    def get_weights_ffd(self, d: float, threshold: float = 1e-5) -> np.ndarray:
        """
        Calculate weights for Fixed-Width Window Fractional Differentiation (FFD)
        
        This method maintains a fixed window size by applying a threshold
        to truncate negligible weights.
        """
        w, k = [1.], 1
        cVal = 1
        
        while True:
            w_ = -w[-1] * (d - k + 1) / k
            if abs(w_) < threshold:
                break
            w.append(w_)
            cVal += w_
            k += 1
            
        # Standardize weights
        w = np.array(w[::-1]).reshape(-1, 1)
        w /= cVal
        
        return w
    
    def frac_diff(self, 
                  series: pd.Series, 
                  d: float, 
                  threshold: float = 0.01) -> pd.Series:
        """
        Apply fractional differentiation to a time series
        
        Parameters:
        -----------
        series : pd.Series
            Input time series
        d : float
            Fractional differentiation parameter
        threshold : float
            Weight threshold for truncation
            
        Returns:
        --------
        pd.Series
            Fractionally differentiated series
        """
        # Calculate weights
        w = self.get_weights_ffd(d, threshold)
        self.weights = w
        
        # Apply fractional differentiation
        width = len(w) - 1
        df = {}
        
        for name in series.index[width:]:
            loc0, loc1 = series.index.get_loc(name) - width, series.index.get_loc(name) + 1
            if not np.isfinite(series.iloc[loc0:loc1]).all():
                continue
                
            df[name] = np.dot(w.T, series.iloc[loc0:loc1].values)[0, 0]
            
        df = pd.Series(df)
        return df
    
    def frac_diff_ffd(self, 
                      series: pd.Series, 
                      d: float,
                      threshold: float = 1e-5) -> pd.Series:
        """
        Fixed-Width Window Fractional Differentiation
        
        More efficient implementation for large datasets
        """
        # Calculate weights
        w = self.get_weights_ffd(d, threshold)
        width = len(w) - 1
        
        # Prepare output
        output = pd.Series(index=series.index, dtype=float)
        
        # Apply convolution
        for iloc1 in range(width, len(series.values)):
            loc0, loc1 = iloc1 - width, iloc1 + 1
            if not np.isfinite(series.values[loc0:loc1]).all():
                continue
            output.iloc[iloc1] = np.dot(w.T, series.values[loc0:loc1])[0]
            
        return output.dropna()
    
    def find_optimal_d(self,
                       series: pd.Series,
                       d_range: tuple = (0.0, 1.0),
                       step: float = 0.1,
                       method: str = 'adf') -> float:
        """
        Find optimal fractional differentiation parameter
        
        Parameters:
        -----------
        series : pd.Series
            Input time series
        d_range : tuple
            Range to search for optimal d
        step : float
            Step size for grid search
        method : str
            Stationarity test method ('adf' or 'kpss')
            
        Returns:
        --------
        float
            Optimal d parameter
        """
        from statsmodels.tsa.stattools import adfuller, kpss
        
        d_values = np.arange(d_range[0], d_range[1] + step, step)
        results = {}
        
        for d in d_values:
            try:
                frac_diff_series = self.frac_diff_ffd(series, d)
                
                if len(frac_diff_series) < 50:  # Minimum observations
                    continue
                    
                if method == 'adf':
                    stat, p_value = adfuller(frac_diff_series.dropna())[:2]
                    results[d] = {'stat': stat, 'p_value': p_value, 'is_stationary': p_value < 0.05}
                elif method == 'kpss':
                    stat, p_value = kpss(frac_diff_series.dropna())[:2]
                    results[d] = {'stat': stat, 'p_value': p_value, 'is_stationary': p_value > 0.05}
                    
            except Exception as e:
                continue
                
        # Find minimum d that achieves stationarity
        stationary_d = [d for d, result in results.items() if result['is_stationary']]
        
        if stationary_d:
            return min(stationary_d)
        else:
            # If no d achieves stationarity, return the one with best p-value
            best_d = min(results.keys(), key=lambda x: results[x]['p_value'])
            return best_d
    
    def validate_stationarity(self, 
                              series: pd.Series,
                              method: str = 'adf') -> dict:
        """
        Validate stationarity of a time series
        
        Parameters:
        -----------
        series : pd.Series
            Time series to test
        method : str
            Test method ('adf', 'kpss', or 'both')
            
        Returns:
        --------
        dict
            Test results
        """
        from statsmodels.tsa.stattools import adfuller, kpss
        
        results = {}
        
        if method in ['adf', 'both']:
            adf_result = adfuller(series.dropna())
            results['adf'] = {
                'statistic': adf_result[0],
                'p_value': adf_result[1],
                'critical_values': adf_result[4],
                'is_stationary': adf_result[1] < 0.05
            }
            
        if method in ['kpss', 'both']:
            kpss_result = kpss(series.dropna())
            results['kpss'] = {
                'statistic': kpss_result[0],
                'p_value': kpss_result[1],
                'critical_values': kpss_result[3],
                'is_stationary': kpss_result[1] > 0.05
            }
            
        return results
    
    def memory_preservation_test(self, 
                                 original: pd.Series,
                                 transformed: pd.Series) -> float:
        """
        Test how much memory is preserved after transformation
        
        Returns correlation between original and transformed series
        """
        # Align series
        common_index = original.index.intersection(transformed.index)
        orig_aligned = original.loc[common_index]
        trans_aligned = transformed.loc[common_index]
        
        # Calculate correlation
        correlation = np.corrcoef(orig_aligned.values, trans_aligned.values)[0, 1]
        
        return correlation