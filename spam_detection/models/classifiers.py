"""
Classification Models for Spam Detection

Implements:
1. Logistic Regression - Baseline classifier
2. K-Means Dictionary Learning - Reconstruction error based classification
"""
import os
import joblib
import numpy as np
from typing import List, Tuple, Optional, Dict, Any, Union
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from scipy.sparse import spmatrix

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    RANDOM_STATE, DEFAULT_SPAM_THRESHOLD,
    KMEANS_N_CLUSTERS_SPAM, KMEANS_N_CLUSTERS_HAM, MODELS_DIR
)


class SpamClassifier:
    """
    Logistic Regression based spam classifier with probability output.
    Supports configurable decision thresholds for precision/recall tuning.
    """
    
    def __init__(self,
                 threshold: float = DEFAULT_SPAM_THRESHOLD,
                 random_state: int = RANDOM_STATE,
                 class_weight: Optional[Union[str, Dict]] = 'balanced',
                 C: float = 1.0,
                 max_iter: int = 1000):
        """
        Initialize the spam classifier.
        
        Args:
            threshold: Decision threshold for spam classification (0-1)
            random_state: Random seed for reproducibility
            class_weight: Class weights for handling imbalanced data
            C: Regularization strength (smaller = stronger regularization)
            max_iter: Maximum iterations for optimization
        """
        self.threshold = threshold
        self.random_state = random_state
        self.class_weight = class_weight
        self.C = C
        self.max_iter = max_iter
        
        self.model = LogisticRegression(
            random_state=random_state,
            class_weight=class_weight,
            C=C,
            max_iter=max_iter,
            solver='lbfgs'
        )
        
        self._is_fitted = False
        self._classes = ['ham', 'spam']
    
    def fit(self, X: spmatrix, y: np.ndarray) -> 'SpamClassifier':
        """
        Train the classifier.
        
        Args:
            X: Feature matrix (sparse)
            y: Labels (0 = ham, 1 = spam)
            
        Returns:
            self for method chaining
        """
        self.model.fit(X, y)
        self._is_fitted = True
        return self
    
    def predict_proba(self, X: spmatrix) -> np.ndarray:
        """
        Get probability predictions.
        
        Args:
            X: Feature matrix
            
        Returns:
            Array of shape (n_samples, 2) with [ham_prob, spam_prob]
        """
        if not self._is_fitted:
            raise ValueError("Classifier must be fitted before prediction")
        return self.model.predict_proba(X)
    
    def predict(self, X: spmatrix, threshold: Optional[float] = None) -> np.ndarray:
        """
        Predict spam/ham labels.
        
        Args:
            X: Feature matrix
            threshold: Optional threshold override
            
        Returns:
            Array of predictions (0 = ham, 1 = spam)
        """
        thresh = threshold if threshold is not None else self.threshold
        proba = self.predict_proba(X)
        return (proba[:, 1] >= thresh).astype(int)
    
    def predict_with_confidence(self, X: spmatrix) -> List[Dict[str, Any]]:
        """
        Get predictions with confidence scores.
        
        Args:
            X: Feature matrix
            
        Returns:
            List of dicts with prediction details
        """
        proba = self.predict_proba(X)
        predictions = self.predict(X)
        
        results = []
        for i in range(len(predictions)):
            results.append({
                'prediction': 'spam' if predictions[i] == 1 else 'ham',
                'spam_probability': float(proba[i, 1]),
                'ham_probability': float(proba[i, 0]),
                'confidence': float(max(proba[i, 0], proba[i, 1])),
                'label': int(predictions[i])
            })
        
        return results
    
    def evaluate(self, X: spmatrix, y: np.ndarray) -> Dict[str, float]:
        """
        Evaluate classifier performance.
        
        Args:
            X: Feature matrix
            y: True labels
            
        Returns:
            Dictionary of metrics
        """
        y_pred = self.predict(X)
        
        return {
            'accuracy': accuracy_score(y, y_pred),
            'precision': precision_score(y, y_pred, zero_division=0),
            'recall': recall_score(y, y_pred, zero_division=0),
            'f1': f1_score(y, y_pred, zero_division=0),
            'confusion_matrix': confusion_matrix(y, y_pred).tolist()
        }
    
    def set_threshold(self, threshold: float) -> None:
        """Update the decision threshold."""
        if not 0 <= threshold <= 1:
            raise ValueError("Threshold must be between 0 and 1")
        self.threshold = threshold
    
    def get_feature_importance(self, feature_names: List[str], top_n: int = 20) -> Dict[str, List]:
        """
        Get the most important features for spam/ham classification.
        
        Args:
            feature_names: List of feature names
            top_n: Number of top features to return
            
        Returns:
            Dict with 'spam_features' and 'ham_features'
        """
        if not self._is_fitted:
            raise ValueError("Classifier must be fitted first")
        
        coefficients = self.model.coef_[0]
        
        # Positive coefficients indicate spam
        spam_indices = np.argsort(coefficients)[-top_n:][::-1]
        ham_indices = np.argsort(coefficients)[:top_n]
        
        return {
            'spam_features': [(feature_names[i], float(coefficients[i])) for i in spam_indices],
            'ham_features': [(feature_names[i], float(coefficients[i])) for i in ham_indices]
        }
    
    def save(self, filepath: Optional[str] = None) -> str:
        """Save the classifier to disk."""
        if not self._is_fitted:
            raise ValueError("Cannot save unfitted classifier")
        
        if filepath is None:
            os.makedirs(MODELS_DIR, exist_ok=True)
            filepath = os.path.join(MODELS_DIR, 'spam_classifier.joblib')
        
        joblib.dump({
            'model': self.model,
            'threshold': self.threshold,
            'random_state': self.random_state,
            'class_weight': self.class_weight,
            'C': self.C,
            'max_iter': self.max_iter,
        }, filepath)
        
        return filepath
    
    @classmethod
    def load(cls, filepath: Optional[str] = None) -> 'SpamClassifier':
        """Load a classifier from disk."""
        if filepath is None:
            filepath = os.path.join(MODELS_DIR, 'spam_classifier.joblib')
        
        data = joblib.load(filepath)
        
        classifier = cls(
            threshold=data['threshold'],
            random_state=data['random_state'],
            class_weight=data['class_weight'],
            C=data['C'],
            max_iter=data['max_iter'],
        )
        classifier.model = data['model']
        classifier._is_fitted = True
        
        return classifier


class DictionaryClassifier:
    """
    K-Means Dictionary Learning based classifier.
    Uses reconstruction error to classify messages.
    
    Two dictionaries are learned:
    - Normal (ham) message dictionary
    - Spam message dictionary
    
    Classification is based on which dictionary better reconstructs the input.
    """
    
    def __init__(self,
                 n_clusters_spam: int = KMEANS_N_CLUSTERS_SPAM,
                 n_clusters_ham: int = KMEANS_N_CLUSTERS_HAM,
                 random_state: int = RANDOM_STATE):
        """
        Initialize the dictionary classifier.
        
        Args:
            n_clusters_spam: Number of clusters for spam dictionary
            n_clusters_ham: Number of clusters for ham dictionary
            random_state: Random seed
        """
        self.n_clusters_spam = n_clusters_spam
        self.n_clusters_ham = n_clusters_ham
        self.random_state = random_state
        
        self.spam_kmeans = KMeans(
            n_clusters=n_clusters_spam,
            random_state=random_state,
            n_init=10
        )
        
        self.ham_kmeans = KMeans(
            n_clusters=n_clusters_ham,
            random_state=random_state,
            n_init=10
        )
        
        self._is_fitted = False
    
    def fit(self, X: spmatrix, y: np.ndarray) -> 'DictionaryClassifier':
        """
        Train the dictionary classifier.
        
        Args:
            X: Feature matrix
            y: Labels (0 = ham, 1 = spam)
            
        Returns:
            self for method chaining
        """
        # Convert sparse to dense if needed
        X_dense = X.toarray() if hasattr(X, 'toarray') else X
        
        # Split into spam and ham
        spam_indices = np.where(y == 1)[0]
        ham_indices = np.where(y == 0)[0]
        
        X_spam = X_dense[spam_indices]
        X_ham = X_dense[ham_indices]
        
        # Fit separate dictionaries
        self.spam_kmeans.fit(X_spam)
        self.ham_kmeans.fit(X_ham)
        
        self._is_fitted = True
        return self
    
    def _reconstruction_error(self, X: np.ndarray, kmeans: KMeans) -> np.ndarray:
        """
        Calculate reconstruction error for each sample.
        
        The error is the distance to the nearest cluster center.
        """
        # Get nearest cluster for each sample
        labels = kmeans.predict(X)
        centers = kmeans.cluster_centers_
        
        # Calculate distance to assigned center
        errors = np.zeros(len(X))
        for i in range(len(X)):
            errors[i] = np.linalg.norm(X[i] - centers[labels[i]])
        
        return errors
    
    def predict(self, X: spmatrix) -> np.ndarray:
        """
        Predict spam/ham using reconstruction error.
        
        Args:
            X: Feature matrix
            
        Returns:
            Predictions (0 = ham, 1 = spam)
        """
        if not self._is_fitted:
            raise ValueError("Classifier must be fitted before prediction")
        
        X_dense = X.toarray() if hasattr(X, 'toarray') else X
        
        # Calculate reconstruction errors
        spam_errors = self._reconstruction_error(X_dense, self.spam_kmeans)
        ham_errors = self._reconstruction_error(X_dense, self.ham_kmeans)
        
        # Lower error = better match to that dictionary
        # If spam_error < ham_error, classify as spam
        predictions = (spam_errors < ham_errors).astype(int)
        
        return predictions
    
    def predict_with_confidence(self, X: spmatrix) -> List[Dict[str, Any]]:
        """
        Get predictions with reconstruction error details.
        """
        X_dense = X.toarray() if hasattr(X, 'toarray') else X
        
        spam_errors = self._reconstruction_error(X_dense, self.spam_kmeans)
        ham_errors = self._reconstruction_error(X_dense, self.ham_kmeans)
        predictions = self.predict(X)
        
        results = []
        for i in range(len(predictions)):
            total_error = spam_errors[i] + ham_errors[i]
            # Convert to probability-like score
            spam_score = 1 - (spam_errors[i] / total_error) if total_error > 0 else 0.5
            
            results.append({
                'prediction': 'spam' if predictions[i] == 1 else 'ham',
                'spam_error': float(spam_errors[i]),
                'ham_error': float(ham_errors[i]),
                'spam_score': float(spam_score),
                'label': int(predictions[i])
            })
        
        return results
    
    def evaluate(self, X: spmatrix, y: np.ndarray) -> Dict[str, float]:
        """Evaluate classifier performance."""
        y_pred = self.predict(X)
        
        return {
            'accuracy': accuracy_score(y, y_pred),
            'precision': precision_score(y, y_pred, zero_division=0),
            'recall': recall_score(y, y_pred, zero_division=0),
            'f1': f1_score(y, y_pred, zero_division=0),
            'confusion_matrix': confusion_matrix(y, y_pred).tolist()
        }
    
    def save(self, filepath: Optional[str] = None) -> str:
        """Save the classifier to disk."""
        if not self._is_fitted:
            raise ValueError("Cannot save unfitted classifier")
        
        if filepath is None:
            os.makedirs(MODELS_DIR, exist_ok=True)
            filepath = os.path.join(MODELS_DIR, 'dictionary_classifier.joblib')
        
        joblib.dump({
            'spam_kmeans': self.spam_kmeans,
            'ham_kmeans': self.ham_kmeans,
            'n_clusters_spam': self.n_clusters_spam,
            'n_clusters_ham': self.n_clusters_ham,
            'random_state': self.random_state,
        }, filepath)
        
        return filepath
    
    @classmethod
    def load(cls, filepath: Optional[str] = None) -> 'DictionaryClassifier':
        """Load a classifier from disk."""
        if filepath is None:
            filepath = os.path.join(MODELS_DIR, 'dictionary_classifier.joblib')
        
        data = joblib.load(filepath)
        
        classifier = cls(
            n_clusters_spam=data['n_clusters_spam'],
            n_clusters_ham=data['n_clusters_ham'],
            random_state=data['random_state'],
        )
        classifier.spam_kmeans = data['spam_kmeans']
        classifier.ham_kmeans = data['ham_kmeans']
        classifier._is_fitted = True
        
        return classifier


class EnsembleClassifier:
    """
    Ensemble classifier combining Logistic Regression and Dictionary Learning.
    """
    
    def __init__(self,
                 lr_weight: float = 0.6,
                 dict_weight: float = 0.4,
                 threshold: float = DEFAULT_SPAM_THRESHOLD):
        """
        Initialize the ensemble.
        
        Args:
            lr_weight: Weight for logistic regression predictions
            dict_weight: Weight for dictionary classifier predictions
            threshold: Decision threshold
        """
        self.lr_weight = lr_weight
        self.dict_weight = dict_weight
        self.threshold = threshold
        
        self.lr_classifier = SpamClassifier()
        self.dict_classifier = DictionaryClassifier()
        
        self._is_fitted = False
    
    def fit(self, X: spmatrix, y: np.ndarray) -> 'EnsembleClassifier':
        """Train both classifiers."""
        self.lr_classifier.fit(X, y)
        self.dict_classifier.fit(X, y)
        self._is_fitted = True
        return self
    
    def predict(self, X: spmatrix) -> np.ndarray:
        """
        Predict using weighted ensemble.
        """
        lr_proba = self.lr_classifier.predict_proba(X)[:, 1]
        
        # Get dictionary classifier scores
        dict_results = self.dict_classifier.predict_with_confidence(X)
        dict_scores = np.array([r['spam_score'] for r in dict_results])
        
        # Weighted average
        combined_scores = (self.lr_weight * lr_proba + 
                          self.dict_weight * dict_scores)
        
        return (combined_scores >= self.threshold).astype(int)
    
    def predict_proba(self, X: spmatrix) -> np.ndarray:
        """
        Get probability predictions from ensemble.
        
        Args:
            X: Feature matrix
            
        Returns:
            Array of shape (n_samples, 2) with [ham_prob, spam_prob]
        """
        lr_proba = self.lr_classifier.predict_proba(X)[:, 1]
        
        # Get dictionary classifier scores
        dict_results = self.dict_classifier.predict_with_confidence(X)
        dict_scores = np.array([r['spam_score'] for r in dict_results])
        
        # Weighted average for spam probability
        spam_proba = (self.lr_weight * lr_proba + self.dict_weight * dict_scores)
        ham_proba = 1 - spam_proba
        
        return np.column_stack([ham_proba, spam_proba])
    
    def predict_with_confidence(self, X: spmatrix) -> List[Dict[str, Any]]:
        """
        Get predictions with confidence scores.
        
        Args:
            X: Feature matrix
            
        Returns:
            List of dicts with prediction details
        """
        proba = self.predict_proba(X)
        predictions = self.predict(X)
        
        results = []
        for i in range(len(predictions)):
            results.append({
                'prediction': 'spam' if predictions[i] == 1 else 'ham',
                'spam_probability': float(proba[i, 1]),
                'ham_probability': float(proba[i, 0]),
                'confidence': float(max(proba[i, 0], proba[i, 1])),
                'label': int(predictions[i])
            })
        
        return results
    
    def evaluate(self, X: spmatrix, y: np.ndarray) -> Dict[str, float]:
        """Evaluate ensemble performance."""
        y_pred = self.predict(X)
        
        return {
            'accuracy': accuracy_score(y, y_pred),
            'precision': precision_score(y, y_pred, zero_division=0),
            'recall': recall_score(y, y_pred, zero_division=0),
            'f1': f1_score(y, y_pred, zero_division=0),
        }
    
    def save(self, filepath: Optional[str] = None) -> str:
        """Save the ensemble classifier to disk."""
        if not self._is_fitted:
            raise ValueError("Cannot save unfitted classifier")
        
        if filepath is None:
            os.makedirs(MODELS_DIR, exist_ok=True)
            filepath = os.path.join(MODELS_DIR, 'ensemble_classifier.joblib')
        
        joblib.dump({
            'lr_classifier_model': self.lr_classifier.model,
            'lr_classifier_threshold': self.lr_classifier.threshold,
            'dict_classifier_spam_kmeans': self.dict_classifier.spam_kmeans,
            'dict_classifier_ham_kmeans': self.dict_classifier.ham_kmeans,
            'dict_classifier_n_clusters_spam': self.dict_classifier.n_clusters_spam,
            'dict_classifier_n_clusters_ham': self.dict_classifier.n_clusters_ham,
            'lr_weight': self.lr_weight,
            'dict_weight': self.dict_weight,
            'threshold': self.threshold,
        }, filepath)
        
        return filepath
    
    @classmethod
    def load(cls, filepath: Optional[str] = None) -> 'EnsembleClassifier':
        """Load an ensemble classifier from disk."""
        if filepath is None:
            filepath = os.path.join(MODELS_DIR, 'ensemble_classifier.joblib')
        
        data = joblib.load(filepath)
        
        ensemble = cls(
            lr_weight=data['lr_weight'],
            dict_weight=data['dict_weight'],
            threshold=data['threshold'],
        )
        
        # Restore LR classifier
        ensemble.lr_classifier.model = data['lr_classifier_model']
        ensemble.lr_classifier.threshold = data['lr_classifier_threshold']
        ensemble.lr_classifier._is_fitted = True
        
        # Restore Dictionary classifier
        ensemble.dict_classifier.spam_kmeans = data['dict_classifier_spam_kmeans']
        ensemble.dict_classifier.ham_kmeans = data['dict_classifier_ham_kmeans']
        ensemble.dict_classifier.n_clusters_spam = data['dict_classifier_n_clusters_spam']
        ensemble.dict_classifier.n_clusters_ham = data['dict_classifier_n_clusters_ham']
        ensemble.dict_classifier._is_fitted = True
        
        ensemble._is_fitted = True
        
        return ensemble


if __name__ == '__main__':
    # Test the classifiers with the SMS dataset
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from preprocessor import TextPreprocessor
    from feature_extractor import FeatureExtractor
    
    # Load data
    print("Loading dataset...")
    data_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'SMSSpamCollection')
    df = pd.read_csv(data_path, sep='\t', header=None, names=['label', 'message'])
    
    # Convert labels
    df['label_num'] = (df['label'] == 'spam').astype(int)
    
    print(f"Dataset size: {len(df)}")
    print(f"Spam messages: {df['label_num'].sum()}")
    print(f"Ham messages: {len(df) - df['label_num'].sum()}")
    
    # Preprocess
    print("\nPreprocessing...")
    preprocessor = TextPreprocessor()
    df['processed'] = preprocessor.preprocess_batch(df['message'].tolist())
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        df['processed'], df['label_num'],
        test_size=0.2, random_state=42, stratify=df['label_num']
    )
    
    # Extract features
    print("Extracting features...")
    feature_extractor = FeatureExtractor()
    X_train_features = feature_extractor.fit_transform(X_train.tolist())
    X_test_features = feature_extractor.transform(X_test.tolist())
    
    print(f"Feature matrix shape: {X_train_features.shape}")
    
    # Train and evaluate Logistic Regression
    print("\n--- Logistic Regression Classifier ---")
    lr_classifier = SpamClassifier()
    lr_classifier.fit(X_train_features, y_train.values)
    lr_metrics = lr_classifier.evaluate(X_test_features, y_test.values)
    print(f"Accuracy: {lr_metrics['accuracy']:.4f}")
    print(f"Precision: {lr_metrics['precision']:.4f}")
    print(f"Recall: {lr_metrics['recall']:.4f}")
    print(f"F1 Score: {lr_metrics['f1']:.4f}")
    
    # Train and evaluate Dictionary Classifier
    print("\n--- Dictionary (K-Means) Classifier ---")
    dict_classifier = DictionaryClassifier()
    dict_classifier.fit(X_train_features, y_train.values)
    dict_metrics = dict_classifier.evaluate(X_test_features, y_test.values)
    print(f"Accuracy: {dict_metrics['accuracy']:.4f}")
    print(f"Precision: {dict_metrics['precision']:.4f}")
    print(f"Recall: {dict_metrics['recall']:.4f}")
    print(f"F1 Score: {dict_metrics['f1']:.4f}")
    
    # Save models
    print("\nSaving models...")
    feature_extractor.save()
    lr_classifier.save()
    dict_classifier.save()
    print("Models saved successfully!")
