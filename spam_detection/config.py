"""
Configuration settings for the Spam Detection System
"""
import os

# Base paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
MODELS_DIR = os.path.join(BASE_DIR, 'models', 'trained')

# Dataset
DATASET_PATH = os.path.join(DATA_DIR, 'SMSSpamCollection')

# Model settings
TFIDF_MAX_FEATURES = 5000
TFIDF_NGRAM_RANGE = (1, 2)
RANDOM_STATE = 42
TEST_SIZE = 0.2

# Classification thresholds
DEFAULT_SPAM_THRESHOLD = 0.6
HIGH_PRECISION_THRESHOLD = 0.7  # Minimize false positives

# Fairness settings
MESSAGE_LENGTH_THRESHOLD = 50  # Characters - short vs long messages

# K-Means dictionary settings
KMEANS_N_CLUSTERS_SPAM = 50
KMEANS_N_CLUSTERS_HAM = 100

# Flask settings
SECRET_KEY = os.environ.get('SECRET_KEY', 'dev-secret-key-change-in-production')
DEBUG = True
