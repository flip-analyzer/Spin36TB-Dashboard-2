import numpy as np
import pandas as pd
from typing import Iterator, Tuple, Optional, List, Dict
from sklearn.model_selection import BaseCrossValidator
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, log_loss
import warnings


class PurgedKFold(BaseCrossValidator):
    """
    Purged K-Fold Cross-Validation for financial time series.
    
    This addresses data leakage issues in time series by:
    1. Purging observations that overlap with the test set
    2. Embargoing observations immediately after the test set
    
    Based on López de Prado's methodology in "Advances in Financial Machine Learning"
    """
    
    def __init__(self, 
                 n_splits: int = 5,
                 embargo_pct: float = 0.01,
                 purge_pct: float = 0.02):
        """
        Parameters:
        -----------
        n_splits : int
            Number of folds
        embargo_pct : float
            Percentage of observations to embargo after test set
        purge_pct : float
            Percentage of observations to purge around test set
        """
        self.n_splits = n_splits
        self.embargo_pct = embargo_pct
        self.purge_pct = purge_pct
        
    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        """Return number of splits"""
        return self.n_splits
    
    def split(self, 
              X: pd.DataFrame, 
              y: Optional[pd.Series] = None,
              groups: Optional[pd.Series] = None) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """
        Generate train/test splits with purging and embargo
        
        Parameters:
        -----------
        X : pd.DataFrame
            Feature matrix with datetime index
        y : pd.Series, optional
            Target variable
        groups : pd.Series, optional
            Sample weights or groups
            
        Yields:
        -------
        tuple
            Train and test indices for each fold
        """
        if not isinstance(X.index, pd.DatetimeIndex):
            raise ValueError("X must have a DatetimeIndex")
            
        indices = np.arange(X.shape[0])
        test_size = X.shape[0] // self.n_splits
        embargo_size = int(self.embargo_pct * X.shape[0])
        purge_size = int(self.purge_pct * X.shape[0])
        
        for fold in range(self.n_splits):
            # Define test set boundaries
            test_start = fold * test_size
            test_end = min((fold + 1) * test_size, X.shape[0])
            
            test_indices = indices[test_start:test_end]
            
            # Apply purging: remove observations that overlap with test set
            purge_start = max(0, test_start - purge_size)
            purge_end = min(X.shape[0], test_end + purge_size)
            
            # Apply embargo: remove observations immediately after test set
            embargo_end = min(X.shape[0], test_end + embargo_size)
            
            # Create train set (excluding purged and embargoed observations)
            train_mask = np.ones(X.shape[0], dtype=bool)
            train_mask[purge_start:embargo_end] = False
            train_indices = indices[train_mask]
            
            # Ensure we have enough training data
            if len(train_indices) < 0.5 * X.shape[0]:
                warnings.warn(f"Fold {fold}: Training set too small ({len(train_indices)} samples)")
                continue
                
            yield train_indices, test_indices


class CombinatoricalPurgedKFold(BaseCrossValidator):
    """
    Combinatorial Purged K-Fold Cross-Validation
    
    This method creates multiple test sets that don't overlap in time,
    providing more robust validation for time series models.
    """
    
    def __init__(self,
                 n_splits: int = 5,
                 n_test_groups: int = 2,
                 embargo_pct: float = 0.01,
                 purge_pct: float = 0.02):
        """
        Parameters:
        -----------
        n_splits : int
            Number of splits to generate
        n_test_groups : int
            Number of non-overlapping test groups per split
        embargo_pct : float
            Embargo percentage
        purge_pct : float
            Purge percentage
        """
        self.n_splits = n_splits
        self.n_test_groups = n_test_groups
        self.embargo_pct = embargo_pct
        self.purge_pct = purge_pct
        
    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        return self.n_splits
        
    def split(self,
              X: pd.DataFrame,
              y: Optional[pd.Series] = None,
              groups: Optional[pd.Series] = None) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """Generate combinatorial purged splits"""
        if not isinstance(X.index, pd.DatetimeIndex):
            raise ValueError("X must have a DatetimeIndex")
            
        n_obs = X.shape[0]
        embargo_size = int(self.embargo_pct * n_obs)
        purge_size = int(self.purge_pct * n_obs)
        
        # Create base groups
        group_size = n_obs // (self.n_test_groups * 2)  # Leave space between groups
        
        for split in range(self.n_splits):
            test_indices = []
            
            # Select test groups for this split
            for group in range(self.n_test_groups):
                start_idx = group * group_size * 2 + (split * group_size // self.n_splits)
                end_idx = start_idx + group_size
                
                if end_idx <= n_obs:
                    test_indices.extend(range(start_idx, end_idx))
                    
            test_indices = np.array(test_indices)
            
            if len(test_indices) == 0:
                continue
                
            # Apply purging and embargo
            train_mask = np.ones(n_obs, dtype=bool)
            
            for idx in test_indices:
                # Purge around each test observation
                purge_start = max(0, idx - purge_size)
                purge_end = min(n_obs, idx + purge_size + 1)
                train_mask[purge_start:purge_end] = False
                
                # Apply embargo after test observation
                embargo_end = min(n_obs, idx + embargo_size + 1)
                train_mask[idx:embargo_end] = False
                
            train_indices = np.where(train_mask)[0]
            
            if len(train_indices) < 0.5 * n_obs:
                continue
                
            yield train_indices, test_indices


class TimeSeriesValidator:
    """
    Time series validation utility with various cross-validation methods
    """
    
    def __init__(self, method: str = 'purged_kfold'):
        """
        Parameters:
        -----------
        method : str
            Validation method ('purged_kfold', 'combinatorial', 'walk_forward')
        """
        self.method = method
        
    def validate_model(self,
                       model,
                       X: pd.DataFrame,
                       y: pd.Series,
                       cv_method: str = 'purged_kfold',
                       n_splits: int = 5,
                       sample_weights: Optional[pd.Series] = None,
                       scoring: List[str] = ['accuracy', 'precision', 'recall', 'f1'],
                       **cv_params) -> pd.DataFrame:
        """
        Validate model using specified cross-validation method
        
        Parameters:
        -----------
        model : sklearn-compatible model
            Model to validate
        X : pd.DataFrame
            Feature matrix
        y : pd.Series
            Target variable
        cv_method : str
            Cross-validation method
        n_splits : int
            Number of splits
        sample_weights : pd.Series, optional
            Sample weights
        scoring : list
            List of scoring metrics
        **cv_params
            Additional parameters for CV method
            
        Returns:
        --------
        pd.DataFrame
            Validation results
        """
        # Select cross-validation method
        if cv_method == 'purged_kfold':
            cv = PurgedKFold(n_splits=n_splits, **cv_params)
        elif cv_method == 'combinatorial':
            cv = CombinatoricalPurgedKFold(n_splits=n_splits, **cv_params)
        else:
            raise ValueError(f"Unknown CV method: {cv_method}")
            
        results = []
        
        for fold, (train_idx, test_idx) in enumerate(cv.split(X, y)):
            # Prepare training and test data
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            
            # Handle sample weights
            if sample_weights is not None:
                w_train = sample_weights.iloc[train_idx]
                try:
                    model.fit(X_train, y_train, sample_weights=w_train)
                except TypeError:
                    # Fallback if model doesn't accept sample_weights
                    model.fit(X_train, y_train)
            else:
                model.fit(X_train, y_train)
                
            # Make predictions
            y_pred = model.predict(X_test)
            
            try:
                y_pred_proba = model.predict_proba(X_test)[:, 1]
            except:
                y_pred_proba = None
                
            # Calculate scores
            fold_scores = {'fold': fold}
            
            for metric in scoring:
                if metric == 'accuracy':
                    score = accuracy_score(y_test, y_pred)
                elif metric == 'precision':
                    score = precision_score(y_test, y_pred, average='weighted', zero_division=0)
                elif metric == 'recall':
                    score = recall_score(y_test, y_pred, average='weighted', zero_division=0)
                elif metric == 'f1':
                    score = f1_score(y_test, y_pred, average='weighted', zero_division=0)
                elif metric == 'log_loss' and y_pred_proba is not None:
                    try:
                        score = log_loss(y_test, y_pred_proba)
                    except:
                        score = np.nan
                else:
                    score = np.nan
                    
                fold_scores[metric] = score
                
            # Add data size information
            fold_scores['train_size'] = len(train_idx)
            fold_scores['test_size'] = len(test_idx)
            fold_scores['test_start'] = X_test.index[0]
            fold_scores['test_end'] = X_test.index[-1]
            
            results.append(fold_scores)
            
        return pd.DataFrame(results)
    
    def walk_forward_validation(self,
                                model,
                                X: pd.DataFrame,
                                y: pd.Series,
                                min_train_size: int = 252,
                                step_size: int = 21,
                                max_train_size: Optional[int] = None,
                                sample_weights: Optional[pd.Series] = None) -> pd.DataFrame:
        """
        Walk-forward validation for time series
        
        Parameters:
        -----------
        model : sklearn-compatible model
            Model to validate
        X : pd.DataFrame
            Feature matrix
        y : pd.Series
            Target variable
        min_train_size : int
            Minimum training window size
        step_size : int
            Step size for moving window
        max_train_size : int, optional
            Maximum training window size (None for expanding window)
        sample_weights : pd.Series, optional
            Sample weights
            
        Returns:
        --------
        pd.DataFrame
            Walk-forward validation results
        """
        results = []
        n_obs = len(X)
        
        for i in range(min_train_size, n_obs, step_size):
            # Define training window
            if max_train_size is None:
                train_start = 0
            else:
                train_start = max(0, i - max_train_size)
                
            train_end = i
            test_start = i
            test_end = min(i + step_size, n_obs)
            
            if test_end <= test_start:
                break
                
            # Prepare data
            X_train = X.iloc[train_start:train_end]
            y_train = y.iloc[train_start:train_end]
            X_test = X.iloc[test_start:test_end]
            y_test = y.iloc[test_start:test_end]
            
            # Train model
            if sample_weights is not None:
                w_train = sample_weights.iloc[train_start:train_end]
                try:
                    model.fit(X_train, y_train, sample_weights=w_train)
                except TypeError:
                    model.fit(X_train, y_train)
            else:
                model.fit(X_train, y_train)
                
            # Make predictions
            y_pred = model.predict(X_test)
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            
            result = {
                'period': len(results),
                'train_start': X_train.index[0],
                'train_end': X_train.index[-1],
                'test_start': X_test.index[0],
                'test_end': X_test.index[-1],
                'train_size': len(X_train),
                'test_size': len(X_test),
                'accuracy': accuracy
            }
            
            results.append(result)
            
        return pd.DataFrame(results)
    
    def calculate_purging_effect(self,
                                 X: pd.DataFrame,
                                 y: pd.Series,
                                 model,
                                 n_splits: int = 5) -> Dict:
        """
        Compare standard K-fold vs Purged K-fold to show purging effect
        """
        from sklearn.model_selection import KFold
        
        # Standard K-fold
        kf_standard = KFold(n_splits=n_splits, shuffle=False)
        scores_standard = []
        
        for train_idx, test_idx in kf_standard.split(X):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            scores_standard.append(accuracy_score(y_test, y_pred))
            
        # Purged K-fold
        kf_purged = PurgedKFold(n_splits=n_splits)
        scores_purged = []
        
        for train_idx, test_idx in kf_purged.split(X, y):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            scores_purged.append(accuracy_score(y_test, y_pred))
            
        return {
            'standard_kfold_mean': np.mean(scores_standard),
            'standard_kfold_std': np.std(scores_standard),
            'purged_kfold_mean': np.mean(scores_purged),
            'purged_kfold_std': np.std(scores_purged),
            'performance_difference': np.mean(scores_standard) - np.mean(scores_purged)
        }