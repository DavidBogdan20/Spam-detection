"""
Bias Mitigation Techniques for Spam Detection

Implements techniques from Assignment 4:
- Re-weighting: Adjust sample weights by group
- Re-sampling: Balance group representation
- Threshold Adjustment: Group-specific decision thresholds
"""
import numpy as np
from typing import Dict, List, Tuple, Optional
from sklearn.utils import resample
from collections import Counter


class BiasMitigation:
    """
    Implements bias mitigation techniques for spam detection.
    """
    
    def __init__(self, length_threshold: int = 50):
        """
        Initialize bias mitigation.
        
        Args:
            length_threshold: Threshold for short/long message classification
        """
        self.length_threshold = length_threshold
    
    def get_protected_attribute(self, messages: List[str]) -> np.ndarray:
        """Get protected group membership based on message length."""
        return np.array([1 if len(msg) > self.length_threshold else 0 for msg in messages])
    
    def calculate_sample_weights(self,
                                  y: np.ndarray,
                                  protected: np.ndarray) -> np.ndarray:
        """
        Calculate sample weights to balance groups.
        
        Re-weighting technique: Increase importance of underrepresented
        group-label combinations.
        
        Args:
            y: Labels (0 = ham, 1 = spam)
            protected: Protected attribute (0 = short, 1 = long)
            
        Returns:
            Array of sample weights
        """
        n_samples = len(y)
        
        # Count group-label combinations
        combinations = {}
        for i in range(n_samples):
            key = (protected[i], y[i])
            combinations[key] = combinations.get(key, 0) + 1
        
        # Calculate expected count (uniform distribution)
        n_groups = 2  # short/long
        n_labels = 2  # ham/spam
        expected_count = n_samples / (n_groups * n_labels)
        
        # Calculate weights
        weights = np.ones(n_samples)
        for i in range(n_samples):
            key = (protected[i], y[i])
            actual_count = combinations[key]
            # Weight inversely proportional to frequency
            weights[i] = expected_count / actual_count if actual_count > 0 else 1.0
        
        # Normalize weights
        weights = weights / weights.mean()
        
        return weights
    
    def resample_balanced(self,
                          X: np.ndarray,
                          y: np.ndarray,
                          messages: List[str],
                          random_state: int = 42) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Resample data to balance group representation.
        
        Uses oversampling of minority group-label combinations.
        
        Args:
            X: Feature matrix
            y: Labels
            messages: Original messages
            random_state: Random seed
            
        Returns:
            Tuple of (resampled_X, resampled_y, resampled_messages)
        """
        protected = self.get_protected_attribute(messages)
        
        # Convert to dense if sparse
        if hasattr(X, 'toarray'):
            X = X.toarray()
        
        # Find the maximum count for any group-label combination
        combinations = {}
        for i in range(len(y)):
            key = (protected[i], y[i])
            if key not in combinations:
                combinations[key] = []
            combinations[key].append(i)
        
        max_count = max(len(indices) for indices in combinations.values())
        
        # Resample each group to match the maximum
        resampled_indices = []
        np.random.seed(random_state)
        
        for key, indices in combinations.items():
            if len(indices) < max_count:
                # Oversample
                oversampled = resample(
                    indices,
                    replace=True,
                    n_samples=max_count,
                    random_state=random_state
                )
                resampled_indices.extend(oversampled)
            else:
                resampled_indices.extend(indices)
        
        # Shuffle
        np.random.shuffle(resampled_indices)
        
        # Create resampled data
        resampled_X = X[resampled_indices]
        resampled_y = y[resampled_indices]
        resampled_messages = [messages[i] for i in resampled_indices]
        
        return resampled_X, resampled_y, resampled_messages
    
    def find_group_thresholds(self,
                               y_true: np.ndarray,
                               y_proba: np.ndarray,
                               protected: np.ndarray,
                               target_metric: str = 'equalized_odds',
                               n_thresholds: int = 100) -> Dict[int, float]:
        """
        Find optimal thresholds for each group to satisfy fairness constraints.
        
        Args:
            y_true: True labels
            y_proba: Predicted probabilities for positive class
            protected: Protected attribute
            target_metric: Which fairness metric to optimize
            n_thresholds: Number of threshold values to try
            
        Returns:
            Dict mapping group (0=short, 1=long) to optimal threshold
        """
        thresholds = np.linspace(0.1, 0.9, n_thresholds)
        
        best_thresholds = {0: 0.5, 1: 0.5}
        best_score = float('inf')
        
        # Grid search over threshold pairs
        for t0 in thresholds:
            for t1 in thresholds:
                # Apply group-specific thresholds
                y_pred = np.zeros(len(y_true))
                y_pred[protected == 0] = (y_proba[protected == 0] >= t0).astype(int)
                y_pred[protected == 1] = (y_proba[protected == 1] >= t1).astype(int)
                
                # Calculate fairness metric
                if target_metric == 'equalized_odds':
                    score = self._equalized_odds_score(y_true, y_pred, protected)
                elif target_metric == 'equal_opportunity':
                    score = self._equal_opportunity_score(y_true, y_pred, protected)
                else:
                    score = abs(t0 - t1)  # Default: minimize threshold difference
                
                if score < best_score:
                    best_score = score
                    best_thresholds = {0: t0, 1: t1}
        
        return best_thresholds
    
    def _equalized_odds_score(self,
                               y_true: np.ndarray,
                               y_pred: np.ndarray,
                               protected: np.ndarray) -> float:
        """Calculate equalized odds difference (lower is better)."""
        tpr_diff, fpr_diff = 0.0, 0.0
        
        for group_label, group_name in [(0, 'short'), (1, 'long')]:
            mask = protected == group_label
            if mask.sum() == 0:
                continue
            
            y_t = y_true[mask]
            y_p = y_pred[mask]
            
            # TPR
            pos_mask = y_t == 1
            if pos_mask.sum() > 0:
                tpr = (y_p[pos_mask] == 1).mean()
            else:
                tpr = 0
            
            # FPR
            neg_mask = y_t == 0
            if neg_mask.sum() > 0:
                fpr = (y_p[neg_mask] == 1).mean()
            else:
                fpr = 0
            
            if group_label == 0:
                tpr_0, fpr_0 = tpr, fpr
            else:
                tpr_1, fpr_1 = tpr, fpr
        
        tpr_diff = abs(tpr_0 - tpr_1) if 'tpr_0' in dir() else 0
        fpr_diff = abs(fpr_0 - fpr_1) if 'fpr_0' in dir() else 0
        
        return max(tpr_diff, fpr_diff)
    
    def _equal_opportunity_score(self,
                                  y_true: np.ndarray,
                                  y_pred: np.ndarray,
                                  protected: np.ndarray) -> float:
        """Calculate equal opportunity difference (lower is better)."""
        tprs = {}
        
        for group_label in [0, 1]:
            mask = protected == group_label
            if mask.sum() == 0:
                tprs[group_label] = 0
                continue
            
            y_t = y_true[mask]
            y_p = y_pred[mask]
            
            pos_mask = y_t == 1
            if pos_mask.sum() > 0:
                tprs[group_label] = (y_p[pos_mask] == 1).mean()
            else:
                tprs[group_label] = 0
        
        return abs(tprs[0] - tprs[1])
    
    def apply_group_thresholds(self,
                                y_proba: np.ndarray,
                                protected: np.ndarray,
                                thresholds: Dict[int, float]) -> np.ndarray:
        """
        Apply group-specific thresholds to probabilities.
        
        Args:
            y_proba: Predicted probabilities
            protected: Protected attribute
            thresholds: Dict mapping group to threshold
            
        Returns:
            Predictions with group-specific thresholds applied
        """
        y_pred = np.zeros(len(y_proba))
        
        for group, threshold in thresholds.items():
            mask = protected == group
            y_pred[mask] = (y_proba[mask] >= threshold).astype(int)
        
        return y_pred


class FairClassifierWrapper:
    """
    Wrapper that adds fairness-aware prediction to any classifier.
    """
    
    def __init__(self, 
                 classifier,
                 mitigation_strategy: str = 'threshold',
                 length_threshold: int = 50):
        """
        Initialize the fair classifier wrapper.
        
        Args:
            classifier: Base classifier with predict_proba method
            mitigation_strategy: 'threshold', 'reweight', or 'resample'
            length_threshold: Message length threshold
        """
        self.classifier = classifier
        self.mitigation_strategy = mitigation_strategy
        self.mitigation = BiasMitigation(length_threshold)
        self.group_thresholds = None
    
    def fit(self, X, y, messages: List[str]):
        """
        Fit the classifier with fairness-aware training.
        """
        protected = self.mitigation.get_protected_attribute(messages)
        
        if self.mitigation_strategy == 'reweight':
            weights = self.mitigation.calculate_sample_weights(y, protected)
            self.classifier.fit(X, y, sample_weight=weights)
        elif self.mitigation_strategy == 'resample':
            X_res, y_res, _ = self.mitigation.resample_balanced(X, y, messages)
            self.classifier.fit(X_res, y_res)
        else:
            self.classifier.fit(X, y)
        
        # For threshold adjustment, calculate optimal thresholds
        if self.mitigation_strategy == 'threshold':
            y_proba = self.classifier.predict_proba(X)[:, 1]
            self.group_thresholds = self.mitigation.find_group_thresholds(
                y, y_proba, protected
            )
        
        return self
    
    def predict(self, X, messages: List[str]) -> np.ndarray:
        """
        Make fairness-aware predictions.
        """
        if self.mitigation_strategy == 'threshold' and self.group_thresholds:
            protected = self.mitigation.get_protected_attribute(messages)
            y_proba = self.classifier.predict_proba(X)[:, 1]
            return self.mitigation.apply_group_thresholds(
                y_proba, protected, self.group_thresholds
            )
        else:
            return self.classifier.predict(X)
    
    def predict_proba(self, X) -> np.ndarray:
        """Get probability predictions."""
        return self.classifier.predict_proba(X)
