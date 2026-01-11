"""
Drift Detection for Spam Detection

Monitors for:
- Data drift (changes in message distribution)
- Concept drift (changes in spam patterns)
- Performance degradation

Based on Assignment 2 and Assignment 3 requirements.
"""
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from collections import deque
from datetime import datetime
import json
import os


class DriftDetector:
    """
    Detect data and concept drift in spam detection.
    """
    
    def __init__(self,
                 window_size: int = 1000,
                 threshold_psi: float = 0.2,
                 threshold_accuracy_drop: float = 0.05,
                 storage_path: Optional[str] = None):
        """
        Initialize drift detector.
        
        Args:
            window_size: Number of samples in sliding window
            threshold_psi: PSI threshold for drift detection
            threshold_accuracy_drop: Accuracy drop threshold for alerts
            storage_path: Path to store drift statistics
        """
        self.window_size = window_size
        self.threshold_psi = threshold_psi
        self.threshold_accuracy_drop = threshold_accuracy_drop
        self.storage_path = storage_path
        
        # Sliding windows for predictions and feedback
        self.predictions = deque(maxlen=window_size)
        self.confidences = deque(maxlen=window_size)
        self.feedback = deque(maxlen=window_size)
        self.message_lengths = deque(maxlen=window_size)
        
        # Reference distributions (from training)
        self.reference_spam_rate = None
        self.reference_confidence_mean = None
        self.reference_length_distribution = None
        
        # Historical metrics for trend analysis
        self.metrics_history = []
    
    def set_reference_distribution(self,
                                    predictions: np.ndarray,
                                    confidences: np.ndarray,
                                    message_lengths: np.ndarray) -> None:
        """
        Set reference distribution from training/validation data.
        
        Args:
            predictions: Model predictions on reference data
            confidences: Confidence scores on reference data
            message_lengths: Message lengths in reference data
        """
        self.reference_spam_rate = predictions.mean()
        self.reference_confidence_mean = confidences.mean()
        self.reference_length_distribution = self._create_histogram(message_lengths)
    
    def _create_histogram(self, values: np.ndarray, n_bins: int = 10) -> np.ndarray:
        """Create a normalized histogram of values."""
        counts, _ = np.histogram(values, bins=n_bins, range=(0, max(values.max(), 500)))
        return counts / counts.sum() if counts.sum() > 0 else counts
    
    def _calculate_psi(self, 
                       reference: np.ndarray, 
                       current: np.ndarray,
                       epsilon: float = 1e-10) -> float:
        """
        Calculate Population Stability Index (PSI).
        
        PSI measures the shift in distribution between reference and current.
        PSI < 0.1: No significant change
        0.1 <= PSI < 0.2: Moderate change
        PSI >= 0.2: Significant change (drift detected)
        """
        # Ensure same size
        if len(reference) != len(current):
            return 0.0
        
        # Add epsilon to avoid log(0)
        reference = reference + epsilon
        current = current + epsilon
        
        # Normalize
        reference = reference / reference.sum()
        current = current / current.sum()
        
        # Calculate PSI
        psi = np.sum((current - reference) * np.log(current / reference))
        
        return float(psi)
    
    def record_prediction(self,
                          prediction: int,
                          confidence: float,
                          message_length: int,
                          timestamp: Optional[datetime] = None) -> None:
        """
        Record a new prediction for drift monitoring.
        
        Args:
            prediction: Model prediction (0 or 1)
            confidence: Model confidence score
            message_length: Length of the message
            timestamp: Prediction timestamp
        """
        self.predictions.append(prediction)
        self.confidences.append(confidence)
        self.message_lengths.append(message_length)
    
    def record_feedback(self, 
                        predicted: int, 
                        actual: int,
                        timestamp: Optional[datetime] = None) -> None:
        """
        Record user feedback for a prediction.
        
        Args:
            predicted: Model prediction
            actual: User-corrected label
            timestamp: Feedback timestamp
        """
        self.feedback.append({
            'predicted': predicted,
            'actual': actual,
            'correct': predicted == actual,
            'timestamp': (timestamp or datetime.now()).isoformat()
        })
    
    def check_drift(self) -> Dict[str, Any]:
        """
        Check for data and concept drift.
        
        Returns:
            Drift detection results with alerts
        """
        results = {
            'data_drift_detected': False,
            'concept_drift_detected': False,
            'alerts': [],
            'metrics': {}
        }
        
        if len(self.predictions) < 100:
            results['alerts'].append('Insufficient data for drift detection')
            return results
        
        predictions = np.array(self.predictions)
        confidences = np.array(self.confidences)
        message_lengths = np.array(self.message_lengths)
        
        # Check spam rate shift
        current_spam_rate = predictions.mean()
        if self.reference_spam_rate is not None:
            spam_rate_shift = abs(current_spam_rate - self.reference_spam_rate)
            results['metrics']['spam_rate_shift'] = float(spam_rate_shift)
            
            if spam_rate_shift > 0.1:  # 10% shift
                results['data_drift_detected'] = True
                results['alerts'].append(
                    f'Spam rate shifted: {self.reference_spam_rate:.2%} → {current_spam_rate:.2%}'
                )
        
        # Check confidence distribution shift
        current_confidence_mean = confidences.mean()
        if self.reference_confidence_mean is not None:
            confidence_shift = abs(current_confidence_mean - self.reference_confidence_mean)
            results['metrics']['confidence_shift'] = float(confidence_shift)
            
            if confidence_shift > 0.05:
                results['alerts'].append(
                    f'Confidence drift: mean shifted by {confidence_shift:.3f}'
                )
        
        # Check message length distribution
        if self.reference_length_distribution is not None:
            current_length_dist = self._create_histogram(message_lengths)
            psi = self._calculate_psi(self.reference_length_distribution, current_length_dist)
            results['metrics']['length_distribution_psi'] = psi
            
            if psi >= self.threshold_psi:
                results['data_drift_detected'] = True
                results['alerts'].append(
                    f'Message length distribution shifted (PSI={psi:.3f})'
                )
        
        # Check feedback-based concept drift
        if len(self.feedback) >= 50:
            recent_feedback = list(self.feedback)[-100:]
            accuracy = sum(1 for f in recent_feedback if f['correct']) / len(recent_feedback)
            results['metrics']['recent_accuracy'] = float(accuracy)
            
            # Compare with expected accuracy
            if accuracy < 0.9:  # Assuming baseline ~95%
                results['concept_drift_detected'] = True
                results['alerts'].append(
                    f'Performance degradation detected: accuracy={accuracy:.2%}'
                )
        
        return results
    
    def get_current_metrics(self) -> Dict[str, float]:
        """
        Get current monitoring metrics.
        
        Returns:
            Dictionary of current metrics
        """
        if len(self.predictions) == 0:
            return {}
        
        predictions = np.array(self.predictions)
        confidences = np.array(self.confidences)
        
        metrics = {
            'current_spam_rate': float(predictions.mean()),
            'mean_confidence': float(confidences.mean()),
            'std_confidence': float(confidences.std()),
            'low_confidence_rate': float((confidences < 0.6).mean()),
            'sample_count': len(self.predictions)
        }
        
        if len(self.feedback) > 0:
            recent_feedback = list(self.feedback)
            metrics['feedback_count'] = len(recent_feedback)
            metrics['feedback_accuracy'] = float(
                sum(1 for f in recent_feedback if f['correct']) / len(recent_feedback)
            )
        
        return metrics
    
    def save_state(self) -> None:
        """Save drift detector state to disk."""
        if self.storage_path is None:
            return
        
        state = {
            'reference_spam_rate': self.reference_spam_rate,
            'reference_confidence_mean': self.reference_confidence_mean,
            'reference_length_distribution': (
                self.reference_length_distribution.tolist() 
                if self.reference_length_distribution is not None else None
            ),
            'metrics_history': self.metrics_history[-100:],  # Keep last 100
        }
        
        os.makedirs(os.path.dirname(self.storage_path), exist_ok=True)
        with open(self.storage_path, 'w') as f:
            json.dump(state, f)
    
    def load_state(self) -> bool:
        """Load drift detector state from disk."""
        if self.storage_path is None or not os.path.exists(self.storage_path):
            return False
        
        with open(self.storage_path, 'r') as f:
            state = json.load(f)
        
        self.reference_spam_rate = state.get('reference_spam_rate')
        self.reference_confidence_mean = state.get('reference_confidence_mean')
        
        if state.get('reference_length_distribution'):
            self.reference_length_distribution = np.array(
                state['reference_length_distribution']
            )
        
        self.metrics_history = state.get('metrics_history', [])
        
        return True
