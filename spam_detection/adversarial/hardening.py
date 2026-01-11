"""
Model Hardening for Spam Detection

Implements techniques to make the classifier more robust against adversarial attacks:
- Adversarial training data augmentation
- Input sanitization
- Robustness testing
"""
import numpy as np
from typing import List, Tuple, Dict, Optional, Any
from .generator import AdversarialGenerator


class ModelHardening:
    """
    Harden spam detection models against adversarial attacks.
    """
    
    def __init__(self, 
                 generator: Optional[AdversarialGenerator] = None,
                 augmentation_ratio: float = 0.3):
        """
        Initialize model hardening.
        
        Args:
            generator: AdversarialGenerator instance
            augmentation_ratio: Ratio of adversarial samples to add
        """
        self.generator = generator or AdversarialGenerator()
        self.augmentation_ratio = augmentation_ratio
    
    def augment_training_data(self,
                               messages: List[str],
                               labels: np.ndarray) -> Tuple[List[str], np.ndarray]:
        """
        Augment training data with adversarial samples.
        
        Only spam messages are augmented (we want to detect adversarial spam).
        
        Args:
            messages: Original messages
            labels: Labels (0 = ham, 1 = spam)
            
        Returns:
            Tuple of (augmented_messages, augmented_labels)
        """
        augmented_messages = list(messages)
        augmented_labels = list(labels)
        
        # Find spam messages
        spam_indices = np.where(labels == 1)[0]
        n_augment = int(len(spam_indices) * self.augmentation_ratio)
        
        # Generate adversarial versions of random spam samples
        selected_indices = np.random.choice(spam_indices, n_augment, replace=True)
        
        for idx in selected_indices:
            original = messages[idx]
            adversarial = self.generator.generate_adversarial_sample(original)
            augmented_messages.append(adversarial)
            augmented_labels.append(1)  # Still spam
        
        # Add synthetic spam
        n_synthetic = int(n_augment * 0.5)
        synthetic_spam = self.generator.create_synthetic_spam(n_synthetic)
        augmented_messages.extend(synthetic_spam)
        augmented_labels.extend([1] * n_synthetic)
        
        return augmented_messages, np.array(augmented_labels)
    
    def test_robustness(self,
                        classifier,
                        feature_extractor,
                        preprocessor,
                        messages: List[str],
                        labels: np.ndarray,
                        n_iterations: int = 3) -> Dict[str, Any]:
        """
        Test classifier robustness against adversarial attacks.
        
        Args:
            classifier: Trained spam classifier
            feature_extractor: Fitted feature extractor
            preprocessor: Text preprocessor
            messages: Test messages
            labels: True labels
            n_iterations: Number of adversarial iterations
            
        Returns:
            Robustness report
        """
        results = {
            'original_accuracy': None,
            'adversarial_accuracy': [],
            'evasion_success_rate': [],
            'techniques_tested': []
        }
        
        # Test on original data
        processed = preprocessor.preprocess_batch(messages)
        features = feature_extractor.transform(processed)
        original_preds = classifier.predict(features)
        original_accuracy = (original_preds == labels).mean()
        results['original_accuracy'] = float(original_accuracy)
        
        # Test with different adversarial techniques
        technique_sets = [
            ['homoglyph'],
            ['zero_width'],
            ['noise'],
            ['homoglyph', 'zero_width'],
            ['homoglyph', 'zero_width', 'noise', 'padding']
        ]
        
        for techniques in technique_sets:
            # Generate adversarial versions
            adversarial_messages = [
                self.generator.generate_adversarial_sample(msg, techniques)
                for msg in messages
            ]
            
            # Preprocess and predict
            processed = preprocessor.preprocess_batch(adversarial_messages)
            features = feature_extractor.transform(processed)
            adv_preds = classifier.predict(features)
            
            adv_accuracy = (adv_preds == labels).mean()
            results['adversarial_accuracy'].append(float(adv_accuracy))
            
            # Calculate evasion rate (spam predicted as ham after perturbation)
            spam_mask = labels == 1
            if spam_mask.sum() > 0:
                original_spam_detected = original_preds[spam_mask] == 1
                adv_spam_detected = adv_preds[spam_mask] == 1
                evasion_rate = (original_spam_detected & ~adv_spam_detected).mean()
            else:
                evasion_rate = 0.0
            
            results['evasion_success_rate'].append(float(evasion_rate))
            results['techniques_tested'].append(techniques)
        
        return results
    
    def generate_robustness_report(self, results: Dict[str, Any]) -> str:
        """
        Generate a human-readable robustness report.
        
        Args:
            results: Results from test_robustness
            
        Returns:
            Formatted report string
        """
        report = []
        report.append("=" * 60)
        report.append("ADVERSARIAL ROBUSTNESS REPORT")
        report.append("=" * 60)
        
        report.append(f"\nOriginal Accuracy: {results['original_accuracy']:.4f}")
        
        report.append("\n--- Adversarial Attack Results ---")
        
        for i, techniques in enumerate(results['techniques_tested']):
            tech_str = ' + '.join(techniques)
            acc = results['adversarial_accuracy'][i]
            evasion = results['evasion_success_rate'][i]
            
            report.append(f"\nTechniques: {tech_str}")
            report.append(f"  Accuracy: {acc:.4f} (drop: {results['original_accuracy'] - acc:.4f})")
            report.append(f"  Evasion Success Rate: {evasion:.4f}")
            
            if evasion > 0.2:
                report.append("  ⚠️ HIGH VULNERABILITY")
            elif evasion > 0.1:
                report.append("  ⚡ MODERATE VULNERABILITY")
            else:
                report.append("  ✓ ROBUST")
        
        report.append("\n" + "=" * 60)
        
        # Overall assessment
        avg_evasion = np.mean(results['evasion_success_rate'])
        report.append(f"\nOverall Average Evasion Rate: {avg_evasion:.4f}")
        
        if avg_evasion < 0.05:
            report.append("Assessment: HIGHLY ROBUST ✓")
        elif avg_evasion < 0.15:
            report.append("Assessment: MODERATELY ROBUST")
        else:
            report.append("Assessment: VULNERABLE - Consider adversarial training ⚠️")
        
        return "\n".join(report)


class InputSanitizer:
    """
    Sanitize inputs before classification to remove adversarial perturbations.
    """
    
    # Characters to strip (zero-width and invisible)
    STRIP_CHARS = set([
        '\u200b', '\u200c', '\u200d', '\ufeff', '\u2060',
        '\u00ad',  # Soft hyphen
    ])
    
    # Homoglyph to ASCII mappings
    NORMALIZE_MAP = {
        # Cyrillic to Latin
        'а': 'a', 'е': 'e', 'о': 'o', 'р': 'p', 'с': 'c', 'у': 'y', 'х': 'x',
        'А': 'A', 'В': 'B', 'Е': 'E', 'К': 'K', 'М': 'M', 'Н': 'H', 'О': 'O',
        'Р': 'P', 'С': 'C', 'Т': 'T', 'Х': 'X', 'і': 'i', 'ј': 'j',
        # Greek to Latin
        'α': 'a', 'ε': 'e', 'ι': 'i', 'ο': 'o', 'ρ': 'p', 'χ': 'x', 'γ': 'y',
        'Α': 'A', 'Β': 'B', 'Ε': 'E', 'Η': 'H', 'Ι': 'I', 'Κ': 'K', 'Μ': 'M',
        'Ν': 'N', 'Ο': 'O', 'Ρ': 'P', 'Τ': 'T', 'Υ': 'Y', 'Χ': 'X',
        # Lookalikes
        'ɑ': 'a', 'ı': 'i', 'ⅼ': 'l', 'ӏ': 'l',
        # Fullwidth
        '０': '0', '１': '1', '２': '2', '３': '3', '４': '4',
        '５': '5', '６': '6', '７': '7', '８': '8', '９': '9',
    }
    
    @classmethod
    def sanitize(cls, text: str) -> str:
        """
        Sanitize text by removing adversarial perturbations.
        
        Args:
            text: Potentially adversarial text
            
        Returns:
            Sanitized text
        """
        # Strip invisible characters
        result = ''.join(c for c in text if c not in cls.STRIP_CHARS)
        
        # Normalize homoglyphs
        result = ''.join(cls.NORMALIZE_MAP.get(c, c) for c in result)
        
        return result
    
    @classmethod
    def sanitize_batch(cls, texts: List[str]) -> List[str]:
        """Sanitize a batch of texts."""
        return [cls.sanitize(text) for text in texts]
