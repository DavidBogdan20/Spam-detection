"""
Adversarial Sample Generator for Spam Detection

Generates realistic attack variants to test and harden the spam classifier.
Based on the Adversarial Defense & Hardening Service (ADV-HS) from Assignment 3.

Techniques:
- Character-level perturbation
- Homoglyph substitution
- Zero-width character insertion
- URL obfuscation
- Content mixing
"""
import random
import re
from typing import List, Tuple, Dict, Optional


class AdversarialGenerator:
    """
    Generates adversarial spam samples to test classifier robustness.
    """
    
    # Homoglyph mappings (ASCII to lookalikes)
    HOMOGLYPHS = {
        'a': ['а', 'ɑ', 'α', 'а'],  # Cyrillic, Latin alpha
        'e': ['е', 'ε', 'ė', 'ē'],  # Cyrillic
        'o': ['о', 'ο', 'ø', 'ö'],  # Cyrillic, Greek
        'i': ['і', 'ι', 'ı', 'ï'],  # Cyrillic, Greek
        'c': ['с', 'ϲ', 'ç'],       # Cyrillic
        'p': ['р', 'ρ'],            # Cyrillic, Greek
        's': ['ѕ', 'ş'],            # Cyrillic
        'x': ['х', 'χ'],            # Cyrillic, Greek
        'y': ['у', 'γ'],            # Cyrillic, Greek
        'n': ['п', 'η'],            # Similar looking
        'm': ['м', 'ṁ'],
        'l': ['ӏ', 'ⅼ', '1'],       # Cyrillic, numeral
        'A': ['А', 'Α', 'Λ'],       # Cyrillic, Greek
        'B': ['В', 'Β'],
        'C': ['С', 'Ϲ'],
        'E': ['Е', 'Ε'],
        'H': ['Н', 'Η'],
        'I': ['І', 'Ι', '1', 'ӏ'],
        'K': ['К', 'Κ'],
        'M': ['М', 'Μ'],
        'N': ['Ν'],
        'O': ['О', 'Ο', '0'],
        'P': ['Р', 'Ρ'],
        'S': ['Ѕ'],
        'T': ['Т', 'Τ'],
        'X': ['Х', 'Χ'],
        'Y': ['Υ', 'У'],
    }
    
    # Zero-width characters
    ZERO_WIDTH_CHARS = [
        '\u200b',  # Zero-width space
        '\u200c',  # Zero-width non-joiner
        '\u200d',  # Zero-width joiner
        '\ufeff',  # Zero-width no-break space
    ]
    
    # Common spam keywords to target
    SPAM_KEYWORDS = [
        'free', 'winner', 'prize', 'urgent', 'claim', 'cash', 'credit',
        'congratulations', 'selected', 'offer', 'limited', 'act now',
        'click', 'subscribe', 'unsubscribe', 'buy', 'order', 'discount',
        'guarantee', 'risk-free', 'call now', 'text', 'reply'
    ]
    
    def __init__(self, 
                 perturbation_rate: float = 0.3,
                 random_seed: Optional[int] = None):
        """
        Initialize the adversarial generator.
        
        Args:
            perturbation_rate: Probability of perturbing each character
            random_seed: Random seed for reproducibility
        """
        self.perturbation_rate = perturbation_rate
        if random_seed is not None:
            random.seed(random_seed)
    
    def apply_homoglyph_substitution(self, text: str, rate: Optional[float] = None) -> str:
        """
        Replace characters with visually similar Unicode homoglyphs.
        
        This technique makes "free" look like "frее" (with Cyrillic 'е').
        
        Args:
            text: Original text
            rate: Substitution rate (default: self.perturbation_rate)
            
        Returns:
            Text with homoglyph substitutions
        """
        rate = rate if rate is not None else self.perturbation_rate
        result = []
        
        for char in text:
            if char in self.HOMOGLYPHS and random.random() < rate:
                result.append(random.choice(self.HOMOGLYPHS[char]))
            else:
                result.append(char)
        
        return ''.join(result)
    
    def insert_zero_width_chars(self, text: str, rate: Optional[float] = None) -> str:
        """
        Insert zero-width characters between letters.
        
        These characters are invisible but break up words for pattern matching.
        "free" becomes "f​r​e​e" with zero-width spaces.
        
        Args:
            text: Original text
            rate: Insertion rate
            
        Returns:
            Text with zero-width characters inserted
        """
        rate = rate if rate is not None else self.perturbation_rate
        result = []
        
        for i, char in enumerate(text):
            result.append(char)
            if char.isalpha() and i < len(text) - 1 and random.random() < rate:
                result.append(random.choice(self.ZERO_WIDTH_CHARS))
        
        return ''.join(result)
    
    def add_character_noise(self, text: str, rate: Optional[float] = None) -> str:
        """
        Add character-level noise (typos, substitutions).
        
        Args:
            text: Original text
            rate: Noise rate
            
        Returns:
            Text with character noise
        """
        rate = rate if rate is not None else self.perturbation_rate * 0.5
        
        noisy_text = []
        i = 0
        while i < len(text):
            char = text[i]
            
            if char.isalpha() and random.random() < rate:
                noise_type = random.choice(['substitute', 'duplicate', 'delete', 'swap'])
                
                if noise_type == 'substitute':
                    # Replace with number or symbol that looks similar
                    subs = {'a': '4', 'e': '3', 'i': '1', 'o': '0', 's': '$', 't': '7'}
                    noisy_text.append(subs.get(char.lower(), char))
                elif noise_type == 'duplicate':
                    noisy_text.append(char)
                    noisy_text.append(char)
                elif noise_type == 'delete':
                    pass  # Skip this character
                elif noise_type == 'swap' and i < len(text) - 1:
                    noisy_text.append(text[i + 1])
                    noisy_text.append(char)
                    i += 1
                else:
                    noisy_text.append(char)
            else:
                noisy_text.append(char)
            
            i += 1
        
        return ''.join(noisy_text)
    
    def obfuscate_urls(self, text: str) -> str:
        """
        Obfuscate URLs to evade detection.
        
        Techniques:
        - Add tracking parameters
        - Use URL shortener patterns
        - Add subdomain confusion
        
        Args:
            text: Original text
            
        Returns:
            Text with obfuscated URLs
        """
        # Pattern to find URLs
        url_pattern = r'(https?://[^\s]+)'
        
        def obfuscate_url(match):
            url = match.group(1)
            techniques = [
                lambda u: u + '?ref=' + ''.join(random.choices('abcdef0123456789', k=8)),
                lambda u: u.replace('http://', 'http://www.'),
                lambda u: u + '#' + ''.join(random.choices('ABCDEF', k=4)),
            ]
            return random.choice(techniques)(url)
        
        return re.sub(url_pattern, obfuscate_url, text)
    
    def add_benign_padding(self, text: str) -> str:
        """
        Add benign-looking content to spam messages.
        
        This technique mixes spam with legitimate-looking text to confuse classifiers.
        
        Args:
            text: Spam text
            
        Returns:
            Text with benign padding
        """
        benign_phrases = [
            "Hope you're doing well.",
            "Just wanted to check in.",
            "Looking forward to hearing from you.",
            "Have a great day!",
            "Best regards,",
            "Thanks for your time.",
            "Let me know if you have any questions.",
        ]
        
        if random.random() < 0.5:
            # Add to beginning
            return random.choice(benign_phrases) + " " + text
        else:
            # Add to end
            return text + " " + random.choice(benign_phrases)
    
    def generate_adversarial_sample(self, 
                                     text: str, 
                                     techniques: Optional[List[str]] = None) -> str:
        """
        Generate an adversarial version of a spam message.
        
        Args:
            text: Original spam message
            techniques: List of techniques to apply. Options:
                       ['homoglyph', 'zero_width', 'noise', 'url', 'padding']
                       Default: apply all
            
        Returns:
            Adversarial sample
        """
        if techniques is None:
            techniques = ['homoglyph', 'zero_width', 'noise', 'padding']
        
        result = text
        
        if 'homoglyph' in techniques:
            result = self.apply_homoglyph_substitution(result)
        
        if 'zero_width' in techniques:
            result = self.insert_zero_width_chars(result)
        
        if 'noise' in techniques:
            result = self.add_character_noise(result)
        
        if 'url' in techniques:
            result = self.obfuscate_urls(result)
        
        if 'padding' in techniques:
            result = self.add_benign_padding(result)
        
        return result
    
    def generate_adversarial_batch(self,
                                    texts: List[str],
                                    techniques: Optional[List[str]] = None) -> List[Tuple[str, str]]:
        """
        Generate adversarial versions of multiple messages.
        
        Args:
            texts: List of original messages
            techniques: Techniques to apply
            
        Returns:
            List of (original, adversarial) tuples
        """
        return [(text, self.generate_adversarial_sample(text, techniques)) 
                for text in texts]
    
    def create_synthetic_spam(self, n_samples: int = 100) -> List[str]:
        """
        Create synthetic spam messages for augmentation.
        
        Args:
            n_samples: Number of samples to generate
            
        Returns:
            List of synthetic spam messages
        """
        templates = [
            "Congratulations! You've won ${amount}! Call {phone} to claim your prize!",
            "URGENT: Your account needs verification. Click {url} immediately!",
            "FREE {product}! Limited time offer. Text {code} to {phone}!",
            "You've been selected for a special {offer}. Reply YES to claim!",
            "WINNER! Claim your {prize} now. Call {phone} before midnight!",
            "Your {service} is expiring. Update now at {url}",
            "Get {discount}% off! Use code {code}. Shop at {url}",
        ]
        
        fillers = {
            'amount': ['$1000', '£500', '€2000', '$10,000'],
            'phone': ['0800-123-456', '09061234567', '1-800-FREE'],
            'url': ['http://bit.ly/claim', 'www.prize-claim.com', 'http://special-offer.net'],
            'product': ['iPhone', 'iPad', 'Samsung Galaxy', 'laptop'],
            'code': ['WIN123', 'FREE99', 'PRIZE', 'CLAIM'],
            'offer': ['exclusive deal', 'VIP membership', 'cash reward'],
            'prize': ['$5000 cash prize', 'brand new car', 'luxury vacation'],
            'service': ['subscription', 'account', 'membership'],
            'discount': ['50', '70', '80', '90'],
        }
        
        samples = []
        for _ in range(n_samples):
            template = random.choice(templates)
            
            # Fill in template
            for key, values in fillers.items():
                template = template.replace('{' + key + '}', random.choice(values))
            
            # Apply adversarial techniques
            adversarial = self.generate_adversarial_sample(template)
            samples.append(adversarial)
        
        return samples


def test_generator():
    """Test the adversarial generator."""
    generator = AdversarialGenerator(perturbation_rate=0.3, random_seed=42)
    
    test_messages = [
        "Congratulations! You've won a FREE iPhone! Call now to claim!",
        "URGENT: Your account will be suspended. Click here to verify.",
        "Winner! You've been selected for £1000 cash prize. Text WIN to 80800"
    ]
    
    print("Original vs Adversarial Samples:")
    print("=" * 60)
    
    for original in test_messages:
        adversarial = generator.generate_adversarial_sample(original)
        print(f"\nOriginal: {original}")
        print(f"Adversarial: {adversarial}")
        print("-" * 60)
    
    print("\n\nSynthetic Spam Samples:")
    print("=" * 60)
    synthetic = generator.create_synthetic_spam(5)
    for i, sample in enumerate(synthetic):
        print(f"{i+1}. {sample}")


if __name__ == '__main__':
    test_generator()
