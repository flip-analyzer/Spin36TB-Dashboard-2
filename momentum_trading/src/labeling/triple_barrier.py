import numpy as np
import pandas as pd
from typing import Optional, Tuple, Dict
from numba import jit
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor


class TripleBarrierLabeler:
    """
    Triple Barrier Labeling implementation following López de Prado's methodology.
    
    This method creates labels based on:
    1. Profit taking barrier (upper)
    2. Stop loss barrier (lower) 
    3. Time-based barrier (vertical)
    
    The first barrier to be touched determines the label.
    """
    
    def __init__(self, 
                 profit_taking_multiple: float = 1.0,
                 stop_loss_multiple: float = 1.0,
                 min_return: float = 0.01):
        """
        Parameters:
        -----------
        profit_taking_multiple : float
            Multiplier for profit taking threshold relative to volatility
        stop_loss_multiple : float  
            Multiplier for stop loss threshold relative to volatility
        min_return : float
            Minimum return threshold for labeling
        """
        self.pt_sl = [profit_taking_multiple, stop_loss_multiple]
        self.min_return = min_return
        
    def get_events(self,
                   close: pd.Series,
                   t_events: pd.DatetimeIndex,
                   pt_sl: list,
                   target: pd.Series,
                   min_ret: float = 0.0,
                   num_threads: int = 1,
                   vertical_barrier_times: Optional[pd.Series] = None) -> pd.DataFrame:
        """
        Get triple barrier events
        
        Parameters:
        -----------
        close : pd.Series
            Close prices
        t_events : pd.DatetimeIndex
            Timestamps where events occur
        pt_sl : list
            [profit_taking_multiple, stop_loss_multiple]
        target : pd.Series
            Target values (typically volatility)
        min_ret : float
            Minimum return to consider
        num_threads : int
            Number of threads for parallel processing
        vertical_barrier_times : pd.Series, optional
            Custom vertical barrier times
            
        Returns:
        --------
        pd.DataFrame
            DataFrame with barrier information
        """
        # Get vertical barrier
        if vertical_barrier_times is None:
            vertical_barriers = self._add_vertical_barrier(t_events, close, num_days=1)
        else:
            vertical_barriers = vertical_barrier_times
            
        # Get barriers
        barriers = pd.concat({
            'pt': pt_sl[0] * target,
            'sl': -pt_sl[1] * target, 
            't1': vertical_barriers
        }, axis=1).dropna()
        
        # Process events
        if num_threads == 1:
            events = self._get_events_single_thread(close, barriers, min_ret)
        else:
            events = self._get_events_multi_thread(close, barriers, min_ret, num_threads)
            
        return events
    
    def _add_vertical_barrier(self, 
                              t_events: pd.DatetimeIndex,
                              close: pd.Series,
                              num_days: int = 1) -> pd.Series:
        """Add vertical barrier (time-based exit)"""
        t1 = close.index.searchsorted(t_events + pd.Timedelta(days=num_days))
        t1 = t1[t1 < close.shape[0]]
        t1 = pd.Series(close.index[t1], index=t_events[:t1.shape[0]])
        return t1
    
    def _get_events_single_thread(self,
                                  close: pd.Series,
                                  barriers: pd.DataFrame,
                                  min_ret: float) -> pd.DataFrame:
        """Process events in single thread"""
        events = []
        
        for loc, barrier in barriers.iterrows():
            out = self._apply_pt_sl_on_t1(close, barrier, min_ret)
            if out is not None:
                events.append(out)
                
        return pd.concat(events, axis=1).T if events else pd.DataFrame()
    
    def _get_events_multi_thread(self,
                                 close: pd.Series,
                                 barriers: pd.DataFrame,
                                 min_ret: float,
                                 num_threads: int) -> pd.DataFrame:
        """Process events with multiple threads"""
        with ProcessPoolExecutor(max_workers=num_threads) as executor:
            jobs = []
            
            for loc, barrier in barriers.iterrows():
                job = executor.submit(self._apply_pt_sl_on_t1, close, barrier, min_ret)
                jobs.append(job)
                
            events = [job.result() for job in jobs if job.result() is not None]
            
        return pd.concat(events, axis=1).T if events else pd.DataFrame()
    
    def _apply_pt_sl_on_t1(self,
                           close: pd.Series,
                           barrier: pd.Series,
                           min_ret: float) -> Optional[pd.Series]:
        """
        Apply profit taking and stop loss barriers
        
        Parameters:
        -----------
        close : pd.Series
            Close prices
        barrier : pd.Series
            Barrier levels [pt, sl, t1]
        min_ret : float
            Minimum return threshold
            
        Returns:
        --------
        pd.Series or None
            Event information
        """
        # Extract barrier components
        pt, sl, t1 = barrier[['pt', 'sl', 't1']]
        
        # Get price path
        if pd.isna(t1):
            t1 = close.index[-1]
            
        # Locate the position
        try:
            loc0 = close.index.get_loc(barrier.name)
            loc1 = close.index.get_loc(t1) + 1
        except KeyError:
            return None
            
        path = close.iloc[loc0:loc1]
        
        if len(path) < 2:
            return None
            
        # Calculate returns
        p0 = path.iloc[0]
        returns = path / p0 - 1.0
        
        # Find first barrier touch
        pt_times = returns[returns >= pt].index if not pd.isna(pt) else pd.Index([])
        sl_times = returns[returns <= sl].index if not pd.isna(sl) else pd.Index([])
        
        # Determine first barrier touched
        if len(pt_times) > 0 and len(sl_times) > 0:
            if pt_times[0] <= sl_times[0]:
                t_first = pt_times[0]
                ret = returns.loc[t_first]
            else:
                t_first = sl_times[0]
                ret = returns.loc[t_first]
        elif len(pt_times) > 0:
            t_first = pt_times[0]
            ret = returns.loc[t_first]
        elif len(sl_times) > 0:
            t_first = sl_times[0]
            ret = returns.loc[t_first]
        else:
            # No barrier touched, use vertical barrier
            t_first = t1
            ret = returns.iloc[-1]
            
        # Apply minimum return filter
        if abs(ret) < min_ret:
            return None
            
        return pd.Series({
            't1': t_first,
            'ret': ret,
            'trgt': barrier[['pt', 'sl']].abs().mean()  # Target as average of barriers
        }, name=barrier.name)
    
    def get_bins(self, events: pd.DataFrame, close: pd.Series) -> pd.DataFrame:
        """
        Get labels from barrier events
        
        Parameters:
        -----------
        events : pd.DataFrame
            Events from get_events()
        close : pd.Series
            Close prices
            
        Returns:
        --------
        pd.DataFrame
            Labels and metadata
        """
        # Initialize
        events = events.copy()
        events['bin'] = np.nan
        
        for i, (t0, event) in enumerate(events.iterrows()):
            ret = event['ret']
            
            # Label based on return
            if ret > 0:
                events.loc[t0, 'bin'] = 1  # Long position
            elif ret < 0:
                events.loc[t0, 'bin'] = -1  # Short position
            else:
                events.loc[t0, 'bin'] = 0  # No position
                
        return events
    
    def drop_labels(self, events: pd.DataFrame, min_pct: float = 0.05) -> pd.DataFrame:
        """
        Drop labels that occur in less than min_pct of cases
        
        This helps with class imbalance issues
        """
        # Count label frequencies
        label_counts = events['bin'].value_counts()
        total_events = len(events)
        
        # Find labels below threshold
        labels_to_drop = label_counts[label_counts / total_events < min_pct].index
        
        # Filter events
        filtered_events = events[~events['bin'].isin(labels_to_drop)]
        
        return filtered_events
    
    def get_sample_weights(self, 
                           events: pd.DataFrame,
                           close: pd.Series,
                           method: str = 'time_decay') -> pd.Series:
        """
        Calculate sample weights for training
        
        Parameters:
        -----------
        events : pd.DataFrame
            Labeled events
        close : pd.Series
            Close prices
        method : str
            Weighting method ('uniform', 'time_decay', 'return_based')
            
        Returns:
        --------
        pd.Series
            Sample weights
        """
        if method == 'uniform':
            weights = pd.Series(1.0, index=events.index)
            
        elif method == 'time_decay':
            # More recent observations get higher weights
            time_diff = (events.index[-1] - events.index).days
            weights = np.exp(-time_diff / 365.25)  # 1-year half-life
            weights = pd.Series(weights, index=events.index)
            
        elif method == 'return_based':
            # Weight by absolute return magnitude
            weights = events['ret'].abs()
            weights = weights / weights.sum()
            
        elif method == 'inverse_variance':
            # Weight by inverse of return variance in a rolling window
            returns = events['ret'].rolling(window=50, min_periods=10).var()
            weights = 1.0 / returns
            weights = weights / weights.sum()
            weights = weights.fillna(weights.mean())
            
        else:
            raise ValueError(f"Unknown weighting method: {method}")
            
        return weights
    
    def get_meta_labeling_events(self,
                                 events: pd.DataFrame,
                                 primary_model_predictions: pd.Series) -> pd.DataFrame:
        """
        Create meta-labeling events for bet sizing
        
        Meta-labeling predicts the probability that a primary model's 
        prediction will be correct
        """
        # Align predictions with events
        aligned_preds = primary_model_predictions.reindex(events.index, method='nearest')
        
        # Create meta-labels: 1 if primary prediction matches actual outcome, 0 otherwise
        events_meta = events.copy()
        events_meta['primary_pred'] = aligned_preds
        events_meta['meta_label'] = (
            (events_meta['bin'] > 0) & (events_meta['primary_pred'] > 0) |
            (events_meta['bin'] < 0) & (events_meta['primary_pred'] < 0) |
            (events_meta['bin'] == 0) & (events_meta['primary_pred'].abs() < 0.1)
        ).astype(int)
        
        return events_meta
    
    def analyze_label_distribution(self, events: pd.DataFrame) -> Dict:
        """Analyze the distribution of labels"""
        label_counts = events['bin'].value_counts().sort_index()
        total = len(events)
        
        analysis = {
            'label_counts': label_counts.to_dict(),
            'label_percentages': (label_counts / total * 100).to_dict(),
            'total_events': total,
            'imbalance_ratio': label_counts.max() / label_counts.min() if len(label_counts) > 1 else 1.0
        }
        
        return analysis