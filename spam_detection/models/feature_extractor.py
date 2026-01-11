"""
Feature Extraction Module for Spam Detection

Implements TF-IDF vectorization with configurable parameters.
Ensures pipeline consistency between training and testing.
"""
import os
import joblib
import numpy as np
from typing import List, Tuple, Optional, Union
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import spmatrix

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import TFIDF_MAX_FEATURES, TFIDF_NGRAM_RANGE, MODELS_DIR


class FeatureExtractor:
    """
    TF-IDF based feature extractor for text classification.
    Supports both training and inference modes.
    """
    
    def __init__(self,
                 max_features: int = TFIDF_MAX_FEATURES,
                 ngram_range: Tuple[int, int] = TFIDF_NGRAM_RANGE,
                 min_df: int = 2,
                 max_df: float = 0.95,
                 sublinear_tf: bool = True):
        """
        Initialize the feature extractor.
        
        Args:
            max_features: Maximum number of features to extract
            ngram_range: Tuple of (min_n, max_n) for n-gram range
            min_df: Minimum document frequency for a term
            max_df: Maximum document frequency for a term
            sublinear_tf: Apply sublinear TF scaling (1 + log(tf))
        """
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.min_df = min_df
        self.max_df = max_df
        self.sublinear_tf = sublinear_tf
        
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            min_df=min_df,
            max_df=max_df,
            sublinear_tf=sublinear_tf,
            strip_accents='unicode',
            analyzer='word',
            token_pattern=r'\b[a-zA-Z]{2,}\b'  # Words with 2+ letters
        )
        
        self._is_fitted = False
    
    def fit(self, texts: List[str]) -> 'FeatureExtractor':
        """
        Fit the vectorizer on training texts.
        
        Args:
            texts: List of preprocessed text documents
            
        Returns:
            self for method chaining
        """
        self.vectorizer.fit(texts)
        self._is_fitted = True
        return self
    
    def transform(self, texts: List[str]) -> spmatrix:
        """
        Transform texts to TF-IDF feature vectors.
        
        Args:
            texts: List of preprocessed text documents
            
        Returns:
            Sparse matrix of TF-IDF features
        """
        if not self._is_fitted:
            raise ValueError("FeatureExtractor must be fitted before transform")
        return self.vectorizer.transform(texts)
    
    def fit_transform(self, texts: List[str]) -> spmatrix:
        """
        Fit and transform in one step.
        
        Args:
            texts: List of preprocessed text documents
            
        Returns:
            Sparse matrix of TF-IDF features
        """
        self._is_fitted = True
        return self.vectorizer.fit_transform(texts)
    
    def get_feature_names(self) -> List[str]:
        """Get the feature names (vocabulary)."""
        if not self._is_fitted:
            raise ValueError("FeatureExtractor must be fitted first")
        return self.vectorizer.get_feature_names_out().tolist()
    
    def get_top_features(self, n: int = 20) -> List[str]:
        """Get the top n most common features."""
        return self.get_feature_names()[:n]
    
    def save(self, filepath: Optional[str] = None) -> str:
        """
        Save the fitted vectorizer to disk.
        
        Args:
            filepath: Path to save the vectorizer (default: models/trained/vectorizer.joblib)
            
        Returns:
            Path where the vectorizer was saved
        """
        if not self._is_fitted:
            raise ValueError("Cannot save unfitted FeatureExtractor")
        
        if filepath is None:
            os.makedirs(MODELS_DIR, exist_ok=True)
            filepath = os.path.join(MODELS_DIR, 'vectorizer.joblib')
        
        joblib.dump({
            'vectorizer': self.vectorizer,
            'max_features': self.max_features,
            'ngram_range': self.ngram_range,
            'min_df': self.min_df,
            'max_df': self.max_df,
            'sublinear_tf': self.sublinear_tf,
        }, filepath)
        
        return filepath
    
    @classmethod
    def load(cls, filepath: Optional[str] = None) -> 'FeatureExtractor':
        """
        Load a fitted vectorizer from disk.
        
        Args:
            filepath: Path to the saved vectorizer
            
        Returns:
            Loaded FeatureExtractor instance
        """
        if filepath is None:
            filepath = os.path.join(MODELS_DIR, 'vectorizer.joblib')
        
        data = joblib.load(filepath)
        
        extractor = cls(
            max_features=data['max_features'],
            ngram_range=data['ngram_range'],
            min_df=data['min_df'],
            max_df=data['max_df'],
            sublinear_tf=data['sublinear_tf'],
        )
        extractor.vectorizer = data['vectorizer']
        extractor._is_fitted = True
        
        return extractor
    
    @property
    def is_fitted(self) -> bool:
        """Check if the vectorizer has been fitted."""
        return self._is_fitted
    
    @property
    def n_features(self) -> int:
        """Get the number of features."""
        if not self._is_fitted:
            return 0
        return len(self.vectorizer.vocabulary_)


def extract_features(texts: List[str], vectorizer: Optional[FeatureExtractor] = None) -> Tuple[spmatrix, FeatureExtractor]:
    """
    Convenience function to extract features from texts.
    
    Args:
        texts: List of preprocessed texts
        vectorizer: Optional pre-fitted vectorizer
        
    Returns:
        Tuple of (feature matrix, vectorizer)
    """
    if vectorizer is None:
        vectorizer = FeatureExtractor()
        features = vectorizer.fit_transform(texts)
    else:
        features = vectorizer.transform(texts)
    
    return features, vectorizer
