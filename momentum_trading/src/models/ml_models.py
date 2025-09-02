import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple, List, Any
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline
import warnings
import joblib
from datetime import datetime


class MomentumMLModel:
    """
    Machine Learning model for momentum trading following López de Prado's methodology.
    
    Implements proper validation, feature selection, and model evaluation
    techniques for financial time series.
    """
    
    def __init__(self, 
                 model_type: str = 'random_forest',
                 use_feature_selection: bool = True,
                 feature_selection_method: str = 'importance',
                 n_features_select: int = 20):
        """
        Parameters:
        -----------
        model_type : str
            Type of ML model ('random_forest', 'gradient_boosting', 'logistic', 'svm')
        use_feature_selection : bool
            Whether to perform feature selection
        feature_selection_method : str
            Method for feature selection ('importance', 'correlation', 'mutual_info')
        n_features_select : int
            Number of top features to select
        """
        self.model_type = model_type
        self.use_feature_selection = use_feature_selection
        self.feature_selection_method = feature_selection_method
        self.n_features_select = n_features_select
        
        self.model = None
        self.scaler = StandardScaler()
        self.selected_features = None
        self.feature_importance = None
        self.is_fitted = False
        
    def _get_base_model(self) -> Any:
        """Get base model based on model_type"""
        if self.model_type == 'random_forest':
            return RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            )
        elif self.model_type == 'gradient_boosting':
            return GradientBoostingClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=6,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42
            )
        elif self.model_type == 'logistic':
            return LogisticRegression(
                C=1.0,
                penalty='l2',
                solver='liblinear',
                random_state=42,
                max_iter=1000
            )
        elif self.model_type == 'svm':
            return SVC(
                C=1.0,
                kernel='rbf',
                gamma='scale',
                probability=True,
                random_state=42
            )
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
    
    def feature_selection(self, 
                          X: pd.DataFrame,
                          y: pd.Series,
                          method: str = 'importance') -> List[str]:
        """
        Perform feature selection
        
        Parameters:
        -----------
        X : pd.DataFrame
            Feature matrix
        y : pd.Series
            Target variable
        method : str
            Selection method
            
        Returns:
        --------
        list
            Selected feature names
        """
        if method == 'importance':
            return self._importance_based_selection(X, y)
        elif method == 'correlation':
            return self._correlation_based_selection(X, y)
        elif method == 'mutual_info':
            return self._mutual_info_selection(X, y)
        else:
            raise ValueError(f"Unknown feature selection method: {method}")
    
    def _importance_based_selection(self, X: pd.DataFrame, y: pd.Series) -> List[str]:
        """Feature selection based on model importance"""
        # Use a simple model to get feature importance
        temp_model = RandomForestClassifier(n_estimators=50, random_state=42)
        temp_model.fit(X, y)
        
        # Get importance scores
        importance_scores = pd.Series(
            temp_model.feature_importances_,
            index=X.columns
        ).sort_values(ascending=False)
        
        # Store for later use
        self.feature_importance = importance_scores
        
        return importance_scores.head(self.n_features_select).index.tolist()
    
    def _correlation_based_selection(self, X: pd.DataFrame, y: pd.Series) -> List[str]:
        """Feature selection based on correlation with target"""
        correlations = X.corrwith(y).abs().sort_values(ascending=False)
        return correlations.head(self.n_features_select).index.tolist()
    
    def _mutual_info_selection(self, X: pd.DataFrame, y: pd.Series) -> List[str]:
        """Feature selection based on mutual information"""
        from sklearn.feature_selection import mutual_info_classif
        
        mi_scores = mutual_info_classif(X, y, random_state=42)
        mi_scores = pd.Series(mi_scores, index=X.columns).sort_values(ascending=False)
        
        return mi_scores.head(self.n_features_select).index.tolist()
    
    def fit(self, 
            X: pd.DataFrame,
            y: pd.Series,
            sample_weights: Optional[pd.Series] = None,
            hyperparameter_tuning: bool = False) -> 'MomentumMLModel':
        """
        Fit the model
        
        Parameters:
        -----------
        X : pd.DataFrame
            Feature matrix
        y : pd.Series
            Target variable
        sample_weights : pd.Series, optional
            Sample weights
        hyperparameter_tuning : bool
            Whether to perform hyperparameter tuning
            
        Returns:
        --------
        self
        """
        # Handle missing values
        X_clean = X.fillna(X.median())
        
        # Feature selection
        if self.use_feature_selection:
            selected_features = self.feature_selection(
                X_clean, y, self.feature_selection_method
            )
            self.selected_features = selected_features
            X_clean = X_clean[selected_features]
        else:
            self.selected_features = X_clean.columns.tolist()
        
        # Scale features
        X_scaled = pd.DataFrame(
            self.scaler.fit_transform(X_clean),
            columns=X_clean.columns,
            index=X_clean.index
        )
        
        # Get base model
        base_model = self._get_base_model()
        
        # Hyperparameter tuning
        if hyperparameter_tuning:
            self.model = self._tune_hyperparameters(X_scaled, y, sample_weights)
        else:
            self.model = base_model
            
        # Fit the model
        try:
            if sample_weights is not None:
                self.model.fit(X_scaled, y, sample_weight=sample_weights)
            else:
                self.model.fit(X_scaled, y)
        except TypeError:
            # Some models don't accept sample_weight parameter
            self.model.fit(X_scaled, y)
            
        self.is_fitted = True
        return self
    
    def _tune_hyperparameters(self, 
                              X: pd.DataFrame,
                              y: pd.Series,
                              sample_weights: Optional[pd.Series] = None) -> Any:
        """Perform hyperparameter tuning"""
        base_model = self._get_base_model()
        
        # Define parameter grids
        if self.model_type == 'random_forest':
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [5, 10, 15, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }
        elif self.model_type == 'gradient_boosting':
            param_grid = {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.05, 0.1, 0.2],
                'max_depth': [3, 6, 9],
                'min_samples_split': [2, 5, 10]
            }
        elif self.model_type == 'logistic':
            param_grid = {
                'C': [0.1, 1.0, 10.0],
                'penalty': ['l1', 'l2'],
                'solver': ['liblinear']
            }
        elif self.model_type == 'svm':
            param_grid = {
                'C': [0.1, 1.0, 10.0],
                'gamma': ['scale', 'auto', 0.001, 0.01, 0.1],
                'kernel': ['rbf', 'poly']
            }
        else:
            return base_model
        
        # Randomized search
        search = RandomizedSearchCV(
            base_model,
            param_grid,
            n_iter=20,
            cv=3,
            scoring='f1_weighted',
            random_state=42,
            n_jobs=-1
        )
        
        # Fit with sample weights if provided
        if sample_weights is not None:
            search.fit(X, y, sample_weight=sample_weights)
        else:
            search.fit(X, y)
            
        return search.best_estimator_
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
            
        X_processed = self._preprocess_features(X)
        return self.model.predict(X_processed)
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict probabilities"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
            
        X_processed = self._preprocess_features(X)
        
        if hasattr(self.model, 'predict_proba'):
            return self.model.predict_proba(X_processed)
        else:
            # For models without predict_proba, return binary predictions
            predictions = self.model.predict(X_processed)
            proba = np.zeros((len(predictions), 3))  # Assuming 3 classes
            for i, pred in enumerate(predictions):
                proba[i, int(pred + 1)] = 1.0  # Convert -1,0,1 to 0,1,2 indices
            return proba
    
    def _preprocess_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Preprocess features for prediction"""
        # Handle missing values
        X_clean = X.fillna(X.median())
        
        # Select features
        if self.selected_features is not None:
            # Only use features that exist in both training and test sets
            available_features = [f for f in self.selected_features if f in X_clean.columns]
            X_clean = X_clean[available_features]
        
        # Scale features
        X_scaled = pd.DataFrame(
            self.scaler.transform(X_clean),
            columns=X_clean.columns,
            index=X_clean.index
        )
        
        return X_scaled
    
    def evaluate(self, 
                 X_test: pd.DataFrame,
                 y_test: pd.Series,
                 sample_weights: Optional[pd.Series] = None) -> Dict:
        """
        Evaluate model performance
        
        Parameters:
        -----------
        X_test : pd.DataFrame
            Test feature matrix
        y_test : pd.Series
            Test target variable
        sample_weights : pd.Series, optional
            Test sample weights
            
        Returns:
        --------
        dict
            Performance metrics
        """
        predictions = self.predict(X_test)
        probabilities = self.predict_proba(X_test)
        
        # Basic metrics
        metrics = {
            'accuracy': accuracy_score(y_test, predictions),
            'precision': precision_score(y_test, predictions, average='weighted', zero_division=0),
            'recall': recall_score(y_test, predictions, average='weighted', zero_division=0),
            'f1': f1_score(y_test, predictions, average='weighted', zero_division=0)
        }
        
        # Class-specific metrics
        try:
            report = classification_report(y_test, predictions, output_dict=True)
            metrics['classification_report'] = report
        except:
            pass
        
        # Financial-specific metrics
        if hasattr(probabilities, 'shape') and probabilities.shape[1] > 1:
            metrics.update(self._calculate_financial_metrics(y_test, predictions, probabilities))
        
        return metrics
    
    def _calculate_financial_metrics(self, 
                                     y_true: pd.Series,
                                     y_pred: np.ndarray,
                                     y_proba: np.ndarray) -> Dict:
        """Calculate financial-specific metrics"""
        metrics = {}
        
        # Hit rate (percentage of correct directional predictions)
        correct_direction = (np.sign(y_true) == np.sign(y_pred)).mean()
        metrics['hit_rate'] = correct_direction
        
        # Precision by class
        for class_label in [-1, 0, 1]:
            if class_label in y_true.values:
                class_mask = (y_true == class_label)
                if class_mask.sum() > 0:
                    class_precision = (y_pred[class_mask] == class_label).mean()
                    metrics[f'precision_class_{class_label}'] = class_precision
        
        # Confidence-weighted accuracy
        if y_proba.shape[1] == 3:  # 3-class problem
            confidence = np.max(y_proba, axis=1)
            high_conf_mask = confidence > 0.6
            if high_conf_mask.sum() > 0:
                high_conf_accuracy = (y_true[high_conf_mask] == y_pred[high_conf_mask]).mean()
                metrics['high_confidence_accuracy'] = high_conf_accuracy
                metrics['high_confidence_samples'] = high_conf_mask.mean()
        
        return metrics
    
    def get_feature_importance(self) -> pd.Series:
        """Get feature importance scores"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")
            
        if hasattr(self.model, 'feature_importances_'):
            importance = pd.Series(
                self.model.feature_importances_,
                index=self.selected_features if self.selected_features else []
            ).sort_values(ascending=False)
            return importance
        elif hasattr(self.model, 'coef_'):
            # For linear models, use absolute coefficients
            coef = np.abs(self.model.coef_[0]) if len(self.model.coef_.shape) > 1 else np.abs(self.model.coef_)
            importance = pd.Series(
                coef,
                index=self.selected_features if self.selected_features else []
            ).sort_values(ascending=False)
            return importance
        else:
            return pd.Series()
    
    def save_model(self, filepath: str) -> None:
        """Save the trained model"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before saving")
            
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'selected_features': self.selected_features,
            'feature_importance': self.feature_importance,
            'model_type': self.model_type,
            'n_features_select': self.n_features_select,
            'feature_selection_method': self.feature_selection_method,
            'timestamp': datetime.now().isoformat()
        }
        
        joblib.dump(model_data, filepath)
    
    def load_model(self, filepath: str) -> 'MomentumMLModel':
        """Load a trained model"""
        model_data = joblib.load(filepath)
        
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.selected_features = model_data['selected_features']
        self.feature_importance = model_data.get('feature_importance')
        self.model_type = model_data['model_type']
        self.n_features_select = model_data['n_features_select']
        self.feature_selection_method = model_data['feature_selection_method']
        self.is_fitted = True
        
        return self
    
    def get_model_summary(self) -> Dict:
        """Get summary of the trained model"""
        if not self.is_fitted:
            return {"status": "Model not fitted"}
        
        summary = {
            "model_type": self.model_type,
            "n_selected_features": len(self.selected_features) if self.selected_features else 0,
            "feature_selection_method": self.feature_selection_method,
            "selected_features": self.selected_features,
            "is_fitted": self.is_fitted
        }
        
        # Add model-specific parameters
        if hasattr(self.model, 'get_params'):
            summary['model_parameters'] = self.model.get_params()
        
        return summary


class EnsembleModel:
    """
    Ensemble model combining multiple momentum models
    """
    
    def __init__(self, models: List[MomentumMLModel], weights: Optional[List[float]] = None):
        """
        Parameters:
        -----------
        models : list
            List of MomentumMLModel instances
        weights : list, optional
            Weights for ensemble combination
        """
        self.models = models
        self.weights = weights or [1.0] * len(models)
        
        if len(self.weights) != len(self.models):
            raise ValueError("Number of weights must match number of models")
    
    def fit(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> 'EnsembleModel':
        """Fit all models in ensemble"""
        for model in self.models:
            model.fit(X, y, **kwargs)
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make ensemble predictions"""
        predictions = []
        for model, weight in zip(self.models, self.weights):
            pred = model.predict(X)
            predictions.append(pred * weight)
        
        # Weighted average
        ensemble_pred = np.sum(predictions, axis=0) / sum(self.weights)
        
        # Convert to class predictions
        return np.round(ensemble_pred).astype(int)
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Make ensemble probability predictions"""
        probabilities = []
        for model, weight in zip(self.models, self.weights):
            prob = model.predict_proba(X)
            probabilities.append(prob * weight)
        
        # Weighted average
        ensemble_proba = np.sum(probabilities, axis=0) / sum(self.weights)
        return ensemble_proba