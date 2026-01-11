"""
Text Preprocessing Module for Spam Detection

Handles:
- Tokenization
- Cleaning and noise removal
- Obfuscation handling (special chars, Unicode normalization)
- Text standardization
"""
import re
import string
import unicodedata
from typing import List, Optional


class TextPreprocessor:
    """
    Preprocesses text messages for spam detection.
    Handles various obfuscation techniques used by spammers.
    """
    
    # Common homoglyph mappings (characters that look similar)
    HOMOGLYPHS = {
        'а': 'a', 'е': 'e', 'о': 'o', 'р': 'p', 'с': 'c', 'у': 'y', 'х': 'x',
        'А': 'A', 'В': 'B', 'Е': 'E', 'К': 'K', 'М': 'M', 'Н': 'H', 'О': 'O',
        'Р': 'P', 'С': 'C', 'Т': 'T', 'Х': 'X',
        '０': '0', '１': '1', '２': '2', '３': '3', '４': '4',
        '５': '5', '６': '6', '７': '7', '８': '8', '９': '9',
        '＄': '$', '€': 'E', '£': 'L', '¥': 'Y',
        'ı': 'i', 'ł': 'l', 'ø': 'o', 'ß': 'ss',
    }
    
    # Zero-width characters to remove
    ZERO_WIDTH_CHARS = [
        '\u200b',  # Zero-width space
        '\u200c',  # Zero-width non-joiner
        '\u200d',  # Zero-width joiner
        '\ufeff',  # Zero-width no-break space
        '\u2060',  # Word joiner
    ]
    
    def __init__(self, 
                 lowercase: bool = True,
                 remove_urls: bool = True,
                 remove_numbers: bool = False,
                 remove_punctuation: bool = True,
                 normalize_unicode: bool = True,
                 remove_zero_width: bool = True,
                 replace_homoglyphs: bool = True):
        """
        Initialize preprocessor with configuration options.
        
        Args:
            lowercase: Convert text to lowercase
            remove_urls: Remove URLs from text
            remove_numbers: Remove numeric characters
            remove_punctuation: Remove punctuation marks
            normalize_unicode: Normalize Unicode characters
            remove_zero_width: Remove zero-width characters
            replace_homoglyphs: Replace homoglyph characters
        """
        self.lowercase = lowercase
        self.remove_urls = remove_urls
        self.remove_numbers = remove_numbers
        self.remove_punctuation = remove_punctuation
        self.normalize_unicode = normalize_unicode
        self.remove_zero_width = remove_zero_width
        self.replace_homoglyphs = replace_homoglyphs
        
        # Compile regex patterns for efficiency
        self.url_pattern = re.compile(
            r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+|'
            r'www\.(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
        )
        self.phone_pattern = re.compile(r'\b\d{5,}\b')  # Numbers 5+ digits (phone-like)
        self.email_pattern = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b')
        self.whitespace_pattern = re.compile(r'\s+')
    
    def _remove_zero_width_chars(self, text: str) -> str:
        """Remove zero-width characters used for obfuscation."""
        for char in self.ZERO_WIDTH_CHARS:
            text = text.replace(char, '')
        return text
    
    def _replace_homoglyphs(self, text: str) -> str:
        """Replace homoglyph characters with ASCII equivalents."""
        for homoglyph, replacement in self.HOMOGLYPHS.items():
            text = text.replace(homoglyph, replacement)
        return text
    
    def _normalize_unicode(self, text: str) -> str:
        """Normalize Unicode characters to ASCII where possible."""
        # NFKD normalization decomposes characters
        text = unicodedata.normalize('NFKD', text)
        # Encode to ASCII, ignoring non-ASCII characters
        text = text.encode('ascii', 'ignore').decode('ascii')
        return text
    
    def _remove_urls(self, text: str) -> str:
        """Remove URLs from text."""
        text = self.url_pattern.sub(' URL ', text)
        text = self.email_pattern.sub(' EMAIL ', text)
        return text
    
    def _clean_text(self, text: str) -> str:
        """Apply all cleaning steps."""
        # Remove zero-width characters first
        if self.remove_zero_width:
            text = self._remove_zero_width_chars(text)
        
        # Replace homoglyphs
        if self.replace_homoglyphs:
            text = self._replace_homoglyphs(text)
        
        # Normalize Unicode
        if self.normalize_unicode:
            text = self._normalize_unicode(text)
        
        # Remove URLs
        if self.remove_urls:
            text = self._remove_urls(text)
        
        # Lowercase
        if self.lowercase:
            text = text.lower()
        
        # Remove punctuation
        if self.remove_punctuation:
            text = text.translate(str.maketrans('', '', string.punctuation))
        
        # Remove numbers (optional)
        if self.remove_numbers:
            text = re.sub(r'\d+', '', text)
        
        # Normalize whitespace
        text = self.whitespace_pattern.sub(' ', text)
        
        return text.strip()
    
    def preprocess(self, text: str) -> str:
        """
        Preprocess a single text message.
        
        Args:
            text: Raw text message
            
        Returns:
            Preprocessed text
        """
        if not text or not isinstance(text, str):
            return ""
        return self._clean_text(text)
    
    def preprocess_batch(self, texts: List[str]) -> List[str]:
        """
        Preprocess a batch of text messages.
        
        Args:
            texts: List of raw text messages
            
        Returns:
            List of preprocessed texts
        """
        return [self.preprocess(text) for text in texts]
    
    def get_message_length(self, text: str) -> int:
        """Get the length of the original message (for fairness analysis)."""
        return len(text) if text else 0
    
    def is_short_message(self, text: str, threshold: int = 50) -> bool:
        """Check if message is considered 'short' for fairness analysis."""
        return self.get_message_length(text) <= threshold


# Convenience function for quick preprocessing
def preprocess_text(text: str, **kwargs) -> str:
    """Convenience function to preprocess text with default settings."""
    preprocessor = TextPreprocessor(**kwargs)
    return preprocessor.preprocess(text)
