"""
Metrics Tracker for Spam Detection

Real-time tracking of:
- Precision, recall, F1
- False positive/negative rates
- User feedback statistics
- Performance over time
"""
import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from collections import defaultdict


class MetricsTracker:
    """
    Track and store metrics for spam detection system.
    """
    
    def __init__(self, storage_path: Optional[str] = None):
        """
        Initialize metrics tracker.
        
        Args:
            storage_path: Path to store metrics data
        """
        self.storage_path = storage_path
        
        # Counters
        self.total_predictions = 0
        self.spam_predictions = 0
        self.ham_predictions = 0
        
        # Confusion matrix components (from feedback)
        self.true_positives = 0
        self.true_negatives = 0
        self.false_positives = 0
        self.false_negatives = 0
        
        # User feedback
        self.feedback_received = 0
        self.mark_as_spam_count = 0
        self.mark_as_ham_count = 0
        
        # Time-based metrics
        self.hourly_predictions = defaultdict(int)
        self.hourly_spam_rate = defaultdict(float)
        
        # Recent predictions for dashboard
        self.recent_predictions = []
        self.max_recent = 100
        
        # Historical metrics
        self.daily_metrics = []
    
    def record_prediction(self,
                          message_id: str,
                          message_preview: str,
                          prediction: int,
                          confidence: float,
                          timestamp: Optional[datetime] = None) -> None:
        """
        Record a new prediction.
        
        Args:
            message_id: Unique message identifier
            message_preview: First N characters of message
            prediction: 0 = ham, 1 = spam
            confidence: Model confidence
            timestamp: Prediction time
        """
        timestamp = timestamp or datetime.now()
        
        self.total_predictions += 1
        if prediction == 1:
            self.spam_predictions += 1
        else:
            self.ham_predictions += 1
        
        # Track hourly
        hour_key = timestamp.strftime('%Y-%m-%d-%H')
        self.hourly_predictions[hour_key] += 1
        
        # Store recent prediction
        self.recent_predictions.append({
            'id': message_id,
            'preview': message_preview[:100],
            'prediction': 'spam' if prediction == 1 else 'ham',
            'confidence': confidence,
            'timestamp': timestamp.isoformat(),
            'feedback': None
        })
        
        # Trim to max
        if len(self.recent_predictions) > self.max_recent:
            self.recent_predictions = self.recent_predictions[-self.max_recent:]
    
    def record_feedback(self,
                        message_id: str,
                        original_prediction: int,
                        user_label: int) -> None:
        """
        Record user feedback on a prediction.
        
        Args:
            message_id: Message identifier
            original_prediction: What was predicted
            user_label: What user says it should be
        """
        self.feedback_received += 1
        
        if user_label == 1:
            self.mark_as_spam_count += 1
        else:
            self.mark_as_ham_count += 1
        
        # Update confusion matrix
        if original_prediction == 1 and user_label == 1:
            self.true_positives += 1
        elif original_prediction == 0 and user_label == 0:
            self.true_negatives += 1
        elif original_prediction == 1 and user_label == 0:
            self.false_positives += 1
        elif original_prediction == 0 and user_label == 1:
            self.false_negatives += 1
        
        # Update recent predictions if found
        for pred in self.recent_predictions:
            if pred['id'] == message_id:
                pred['feedback'] = 'spam' if user_label == 1 else 'ham'
                break
    
    def get_current_metrics(self) -> Dict[str, Any]:
        """
        Get current performance metrics.
        
        Returns:
            Dictionary of metrics
        """
        metrics = {
            'total_predictions': self.total_predictions,
            'spam_predictions': self.spam_predictions,
            'ham_predictions': self.ham_predictions,
            'spam_rate': (
                self.spam_predictions / self.total_predictions 
                if self.total_predictions > 0 else 0
            ),
            'feedback_received': self.feedback_received,
            'feedback_rate': (
                self.feedback_received / self.total_predictions
                if self.total_predictions > 0 else 0
            )
        }
        
        # Calculate precision, recall if we have feedback
        total_feedback = (self.true_positives + self.true_negatives + 
                         self.false_positives + self.false_negatives)
        
        if total_feedback > 0:
            # Precision: TP / (TP + FP)
            if (self.true_positives + self.false_positives) > 0:
                metrics['precision'] = (
                    self.true_positives / 
                    (self.true_positives + self.false_positives)
                )
            else:
                metrics['precision'] = 0.0
            
            # Recall: TP / (TP + FN)
            if (self.true_positives + self.false_negatives) > 0:
                metrics['recall'] = (
                    self.true_positives / 
                    (self.true_positives + self.false_negatives)
                )
            else:
                metrics['recall'] = 0.0
            
            # F1: 2 * (precision * recall) / (precision + recall)
            if metrics['precision'] + metrics['recall'] > 0:
                metrics['f1'] = (
                    2 * metrics['precision'] * metrics['recall'] / 
                    (metrics['precision'] + metrics['recall'])
                )
            else:
                metrics['f1'] = 0.0
            
            # Accuracy
            metrics['accuracy'] = (
                (self.true_positives + self.true_negatives) / total_feedback
            )
            
            # False positive rate: FP / (FP + TN)
            if (self.false_positives + self.true_negatives) > 0:
                metrics['false_positive_rate'] = (
                    self.false_positives / 
                    (self.false_positives + self.true_negatives)
                )
            else:
                metrics['false_positive_rate'] = 0.0
            
            # False negative rate: FN / (FN + TP)
            if (self.false_negatives + self.true_positives) > 0:
                metrics['false_negative_rate'] = (
                    self.false_negatives / 
                    (self.false_negatives + self.true_positives)
                )
            else:
                metrics['false_negative_rate'] = 0.0
            
            metrics['confusion_matrix'] = {
                'true_positives': self.true_positives,
                'true_negatives': self.true_negatives,
                'false_positives': self.false_positives,
                'false_negatives': self.false_negatives
            }
        
        return metrics
    
    def get_recent_predictions(self, n: int = 10) -> List[Dict]:
        """Get most recent predictions."""
        return self.recent_predictions[-n:]
    
    def get_hourly_stats(self, hours: int = 24) -> Dict[str, int]:
        """Get prediction counts by hour for the last N hours."""
        now = datetime.now()
        result = {}
        
        for i in range(hours):
            hour = now - timedelta(hours=i)
            key = hour.strftime('%Y-%m-%d-%H')
            result[hour.strftime('%H:00')] = self.hourly_predictions.get(key, 0)
        
        return result
    
    def get_dashboard_data(self) -> Dict[str, Any]:
        """
        Get all data needed for dashboard display.
        
        Returns:
            Complete dashboard data
        """
        metrics = self.get_current_metrics()
        
        return {
            'metrics': metrics,
            'recent_predictions': self.get_recent_predictions(20),
            'hourly_stats': self.get_hourly_stats(24),
            'alerts': self._generate_alerts(),
            'timestamp': datetime.now().isoformat()
        }
    
    def _generate_alerts(self) -> List[Dict[str, str]]:
        """Generate alerts based on current metrics."""
        alerts = []
        
        metrics = self.get_current_metrics()
        
        # High false positive rate alert
        if metrics.get('false_positive_rate', 0) > 0.05:
            alerts.append({
                'level': 'warning',
                'message': f"High false positive rate: {metrics['false_positive_rate']:.2%}",
                'action': 'Consider adjusting classification threshold'
            })
        
        # Low precision alert
        if metrics.get('precision', 1) < 0.8 and self.feedback_received > 20:
            alerts.append({
                'level': 'warning',
                'message': f"Low precision: {metrics['precision']:.2%}",
                'action': 'Model may need retraining'
            })
        
        # Unusual spam rate
        current_spam_rate = metrics.get('spam_rate', 0)
        if current_spam_rate > 0.5:
            alerts.append({
                'level': 'info',
                'message': f"High spam rate detected: {current_spam_rate:.2%}",
                'action': 'May indicate spam attack or model issue'
            })
        
        return alerts
    
    def save_state(self) -> None:
        """Save metrics state to disk."""
        if self.storage_path is None:
            return
        
        state = {
            'total_predictions': self.total_predictions,
            'spam_predictions': self.spam_predictions,
            'ham_predictions': self.ham_predictions,
            'true_positives': self.true_positives,
            'true_negatives': self.true_negatives,
            'false_positives': self.false_positives,
            'false_negatives': self.false_negatives,
            'feedback_received': self.feedback_received,
            'mark_as_spam_count': self.mark_as_spam_count,
            'mark_as_ham_count': self.mark_as_ham_count,
            'recent_predictions': self.recent_predictions[-50:],
            'saved_at': datetime.now().isoformat()
        }
        
        os.makedirs(os.path.dirname(self.storage_path), exist_ok=True)
        with open(self.storage_path, 'w') as f:
            json.dump(state, f, indent=2)
    
    def load_state(self) -> bool:
        """Load metrics state from disk."""
        if self.storage_path is None or not os.path.exists(self.storage_path):
            return False
        
        with open(self.storage_path, 'r') as f:
            state = json.load(f)
        
        self.total_predictions = state.get('total_predictions', 0)
        self.spam_predictions = state.get('spam_predictions', 0)
        self.ham_predictions = state.get('ham_predictions', 0)
        self.true_positives = state.get('true_positives', 0)
        self.true_negatives = state.get('true_negatives', 0)
        self.false_positives = state.get('false_positives', 0)
        self.false_negatives = state.get('false_negatives', 0)
        self.feedback_received = state.get('feedback_received', 0)
        self.mark_as_spam_count = state.get('mark_as_spam_count', 0)
        self.mark_as_ham_count = state.get('mark_as_ham_count', 0)
        self.recent_predictions = state.get('recent_predictions', [])
        
        return True
    
    def reset(self) -> None:
        """Reset all metrics."""
        self.total_predictions = 0
        self.spam_predictions = 0
        self.ham_predictions = 0
        self.true_positives = 0
        self.true_negatives = 0
        self.false_positives = 0
        self.false_negatives = 0
        self.feedback_received = 0
        self.mark_as_spam_count = 0
        self.mark_as_ham_count = 0
        self.recent_predictions = []
        self.hourly_predictions.clear()
