"""
Fairness Metrics for Spam Detection

Implements metrics from Assignment 4:
- Disparate Impact
- Equal Opportunity
- Equalized Odds
- Predictive Parity

Protected attribute: Message length (short vs long messages)
"""
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from sklearn.metrics import confusion_matrix


class FairnessMetrics:
    """
    Calculate fairness metrics for spam detection.
    
    Based on the fairness analysis from Assignment 4, using message length
    as a proxy protected attribute.
    """
    
    def __init__(self, length_threshold: int = 50):
        """
        Initialize fairness metrics calculator.
        
        Args:
            length_threshold: Character count threshold for short/long classification
        """
        self.length_threshold = length_threshold
    
    def get_protected_attribute(self, messages: List[str]) -> np.ndarray:
        """
        Determine protected group membership based on message length.
        
        Args:
            messages: List of original (unprocessed) messages
            
        Returns:
            Binary array: 0 = short message, 1 = long message
        """
        return np.array([1 if len(msg) > self.length_threshold else 0 for msg in messages])
    
    def _split_by_group(self, 
                        y_true: np.ndarray, 
                        y_pred: np.ndarray,
                        protected: np.ndarray) -> Tuple[Dict, Dict]:
        """
        Split predictions by protected group.
        
        Returns:
            Tuple of (short_message_data, long_message_data)
        """
        short_mask = protected == 0
        long_mask = protected == 1
        
        short_data = {
            'y_true': y_true[short_mask],
            'y_pred': y_pred[short_mask],
            'count': short_mask.sum()
        }
        
        long_data = {
            'y_true': y_true[long_mask],
            'y_pred': y_pred[long_mask],
            'count': long_mask.sum()
        }
        
        return short_data, long_data
    
    def _calculate_rates(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """
        Calculate confusion matrix-based rates.
        """
        if len(y_true) == 0:
            return {
                'tpr': 0.0, 'fpr': 0.0, 'tnr': 0.0, 'fnr': 0.0,
                'precision': 0.0, 'selection_rate': 0.0
            }
        
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
        
        # True Positive Rate (Recall/Sensitivity)
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        
        # False Positive Rate
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        
        # True Negative Rate (Specificity)
        tnr = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        
        # False Negative Rate
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
        
        # Precision (Positive Predictive Value)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        
        # Selection Rate (proportion predicted positive)
        selection_rate = (tp + fp) / len(y_true) if len(y_true) > 0 else 0.0
        
        return {
            'tpr': tpr,
            'fpr': fpr,
            'tnr': tnr,
            'fnr': fnr,
            'precision': precision,
            'selection_rate': selection_rate,
            'tp': int(tp),
            'fp': int(fp),
            'tn': int(tn),
            'fn': int(fn)
        }
    
    def disparate_impact(self,
                         y_pred: np.ndarray,
                         protected: np.ndarray) -> Dict[str, float]:
        """
        Calculate Disparate Impact.
        
        Measures whether one group receives favorable outcomes (ham prediction)
        at a significantly different rate than another.
        
        A value below 0.8 indicates potential unfair impact (four-fifths rule).
        
        Args:
            y_pred: Predicted labels (0 = ham, 1 = spam)
            protected: Protected attribute (0 = short, 1 = long)
            
        Returns:
            Dict with disparate impact metrics
        """
        short_mask = protected == 0
        long_mask = protected == 1
        
        # Calculate favorable outcome (ham prediction) rates
        # For spam detection, ham (0) is the favorable outcome
        short_ham_rate = (y_pred[short_mask] == 0).mean() if short_mask.sum() > 0 else 0
        long_ham_rate = (y_pred[long_mask] == 0).mean() if long_mask.sum() > 0 else 0
        
        # Disparate impact: ratio of favorable outcome rates
        # Use the smaller rate in numerator
        if short_ham_rate == 0 and long_ham_rate == 0:
            di = 1.0
        elif long_ham_rate == 0:
            di = float('inf')
        else:
            di = short_ham_rate / long_ham_rate
        
        return {
            'disparate_impact': di,
            'short_message_ham_rate': float(short_ham_rate),
            'long_message_ham_rate': float(long_ham_rate),
            'passes_four_fifths_rule': di >= 0.8
        }
    
    def equal_opportunity(self,
                          y_true: np.ndarray,
                          y_pred: np.ndarray,
                          protected: np.ndarray) -> Dict[str, float]:
        """
        Calculate Equal Opportunity.
        
        Requires equal True Positive Rates across groups.
        For spam detection: equal detection rates for actual spam.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            protected: Protected attribute
            
        Returns:
            Dict with equal opportunity metrics
        """
        short_data, long_data = self._split_by_group(y_true, y_pred, protected)
        
        short_rates = self._calculate_rates(short_data['y_true'], short_data['y_pred'])
        long_rates = self._calculate_rates(long_data['y_true'], long_data['y_pred'])
        
        tpr_difference = abs(short_rates['tpr'] - long_rates['tpr'])
        
        return {
            'equal_opportunity_difference': tpr_difference,
            'short_message_tpr': short_rates['tpr'],
            'long_message_tpr': long_rates['tpr'],
            'satisfies_equal_opportunity': tpr_difference < 0.1  # 10% threshold
        }
    
    def equalized_odds(self,
                       y_true: np.ndarray,
                       y_pred: np.ndarray,
                       protected: np.ndarray) -> Dict[str, float]:
        """
        Calculate Equalized Odds.
        
        Requires equal TPR AND equal FPR across groups.
        More stringent than equal opportunity.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            protected: Protected attribute
            
        Returns:
            Dict with equalized odds metrics
        """
        short_data, long_data = self._split_by_group(y_true, y_pred, protected)
        
        short_rates = self._calculate_rates(short_data['y_true'], short_data['y_pred'])
        long_rates = self._calculate_rates(long_data['y_true'], long_data['y_pred'])
        
        tpr_difference = abs(short_rates['tpr'] - long_rates['tpr'])
        fpr_difference = abs(short_rates['fpr'] - long_rates['fpr'])
        
        # Combined metric
        equalized_odds_diff = max(tpr_difference, fpr_difference)
        
        return {
            'equalized_odds_difference': equalized_odds_diff,
            'tpr_difference': tpr_difference,
            'fpr_difference': fpr_difference,
            'short_message_tpr': short_rates['tpr'],
            'long_message_tpr': long_rates['tpr'],
            'short_message_fpr': short_rates['fpr'],
            'long_message_fpr': long_rates['fpr'],
            'satisfies_equalized_odds': equalized_odds_diff < 0.1
        }
    
    def predictive_parity(self,
                          y_true: np.ndarray,
                          y_pred: np.ndarray,
                          protected: np.ndarray) -> Dict[str, float]:
        """
        Calculate Predictive Parity.
        
        Requires equal precision across groups.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            protected: Protected attribute
            
        Returns:
            Dict with predictive parity metrics
        """
        short_data, long_data = self._split_by_group(y_true, y_pred, protected)
        
        short_rates = self._calculate_rates(short_data['y_true'], short_data['y_pred'])
        long_rates = self._calculate_rates(long_data['y_true'], long_data['y_pred'])
        
        precision_difference = abs(short_rates['precision'] - long_rates['precision'])
        
        return {
            'predictive_parity_difference': precision_difference,
            'short_message_precision': short_rates['precision'],
            'long_message_precision': long_rates['precision'],
            'satisfies_predictive_parity': precision_difference < 0.1
        }
    
    def calculate_all_metrics(self,
                              y_true: np.ndarray,
                              y_pred: np.ndarray,
                              protected: np.ndarray) -> Dict[str, Any]:
        """
        Calculate all fairness metrics.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            protected: Protected attribute
            
        Returns:
            Comprehensive fairness report
        """
        short_data, long_data = self._split_by_group(y_true, y_pred, protected)
        
        return {
            'disparate_impact': self.disparate_impact(y_pred, protected),
            'equal_opportunity': self.equal_opportunity(y_true, y_pred, protected),
            'equalized_odds': self.equalized_odds(y_true, y_pred, protected),
            'predictive_parity': self.predictive_parity(y_true, y_pred, protected),
            'group_statistics': {
                'short_messages': {
                    'count': int(short_data['count']),
                    'spam_count': int(short_data['y_true'].sum()) if len(short_data['y_true']) > 0 else 0,
                    'ham_count': int(len(short_data['y_true']) - short_data['y_true'].sum()) if len(short_data['y_true']) > 0 else 0
                },
                'long_messages': {
                    'count': int(long_data['count']),
                    'spam_count': int(long_data['y_true'].sum()) if len(long_data['y_true']) > 0 else 0,
                    'ham_count': int(len(long_data['y_true']) - long_data['y_true'].sum()) if len(long_data['y_true']) > 0 else 0
                }
            }
        }
    
    def generate_report(self,
                        y_true: np.ndarray,
                        y_pred: np.ndarray,
                        messages: List[str]) -> str:
        """
        Generate a human-readable fairness report.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            messages: Original messages
            
        Returns:
            Formatted report string
        """
        protected = self.get_protected_attribute(messages)
        metrics = self.calculate_all_metrics(y_true, y_pred, protected)
        
        report = []
        report.append("=" * 60)
        report.append("FAIRNESS ANALYSIS REPORT")
        report.append(f"Protected Attribute: Message Length (threshold: {self.length_threshold} chars)")
        report.append("=" * 60)
        
        # Group Statistics
        report.append("\n--- Group Statistics ---")
        gs = metrics['group_statistics']
        report.append(f"Short messages (≤{self.length_threshold} chars): {gs['short_messages']['count']}")
        report.append(f"  - Spam: {gs['short_messages']['spam_count']}, Ham: {gs['short_messages']['ham_count']}")
        report.append(f"Long messages (>{self.length_threshold} chars): {gs['long_messages']['count']}")
        report.append(f"  - Spam: {gs['long_messages']['spam_count']}, Ham: {gs['long_messages']['ham_count']}")
        
        # Disparate Impact
        report.append("\n--- Disparate Impact ---")
        di = metrics['disparate_impact']
        report.append(f"Disparate Impact Ratio: {di['disparate_impact']:.4f}")
        report.append(f"Short message ham rate: {di['short_message_ham_rate']:.4f}")
        report.append(f"Long message ham rate: {di['long_message_ham_rate']:.4f}")
        status = "✓ PASS" if di['passes_four_fifths_rule'] else "✗ FAIL"
        report.append(f"Four-fifths rule (≥0.8): {status}")
        
        # Equal Opportunity
        report.append("\n--- Equal Opportunity ---")
        eo = metrics['equal_opportunity']
        report.append(f"TPR Difference: {eo['equal_opportunity_difference']:.4f}")
        report.append(f"Short message TPR: {eo['short_message_tpr']:.4f}")
        report.append(f"Long message TPR: {eo['long_message_tpr']:.4f}")
        status = "✓ PASS" if eo['satisfies_equal_opportunity'] else "✗ FAIL"
        report.append(f"Satisfies Equal Opportunity (<0.1 diff): {status}")
        
        # Equalized Odds
        report.append("\n--- Equalized Odds ---")
        eod = metrics['equalized_odds']
        report.append(f"Equalized Odds Difference: {eod['equalized_odds_difference']:.4f}")
        report.append(f"TPR Difference: {eod['tpr_difference']:.4f}")
        report.append(f"FPR Difference: {eod['fpr_difference']:.4f}")
        status = "✓ PASS" if eod['satisfies_equalized_odds'] else "✗ FAIL"
        report.append(f"Satisfies Equalized Odds (<0.1 diff): {status}")
        
        # Predictive Parity
        report.append("\n--- Predictive Parity ---")
        pp = metrics['predictive_parity']
        report.append(f"Precision Difference: {pp['predictive_parity_difference']:.4f}")
        report.append(f"Short message precision: {pp['short_message_precision']:.4f}")
        report.append(f"Long message precision: {pp['long_message_precision']:.4f}")
        status = "✓ PASS" if pp['satisfies_predictive_parity'] else "✗ FAIL"
        report.append(f"Satisfies Predictive Parity (<0.1 diff): {status}")
        
        report.append("\n" + "=" * 60)
        
        return "\n".join(report)
