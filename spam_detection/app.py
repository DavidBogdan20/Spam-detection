"""
Spam Detection System - Flask Application

A comprehensive AI-enabled spam detection system with:
- ML-based classification (Logistic Regression + K-Means Dictionary)
- Fairness metrics and bias mitigation
- Adversarial robustness testing
- Real-time monitoring and drift detection
- Modern web interface for inbox simulation
"""
import os
import sys
import pandas as pd
from flask import Flask, render_template, jsonify

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import SECRET_KEY, DEBUG, DATASET_PATH
from models.preprocessor import TextPreprocessor
from models.feature_extractor import FeatureExtractor
from models.classifiers import SpamClassifier
from monitoring.metrics_tracker import MetricsTracker
from monitoring.drift_detector import DriftDetector
from fairness.metrics import FairnessMetrics
from api.routes import api_bp, init_api


def create_app():
    """Create and configure the Flask application."""
    app = Flask(__name__)
    app.secret_key = SECRET_KEY
    
    # Register API blueprint
    app.register_blueprint(api_bp)
    
    # Load and initialize components
    with app.app_context():
        init_components(app)
    
    # Main routes
    @app.route('/')
    def index():
        """Render the main inbox interface."""
        return render_template('index.html')
    
    @app.route('/dashboard')
    def dashboard():
        """Render the monitoring dashboard."""
        return render_template('dashboard.html')
    
    @app.errorhandler(404)
    def not_found(e):
        return jsonify({'error': 'Not found'}), 404
    
    @app.errorhandler(500)
    def server_error(e):
        return jsonify({'error': 'Internal server error'}), 500
    
    return app


def init_components(app):
    """Initialize ML components and load models."""
    print("Initializing spam detection system...")
    
    # Check if trained models exist
    models_dir = os.path.join(os.path.dirname(__file__), 'models', 'trained')
    classifier_path = os.path.join(models_dir, 'spam_classifier.joblib')
    vectorizer_path = os.path.join(models_dir, 'vectorizer.joblib')
    
    if os.path.exists(classifier_path) and os.path.exists(vectorizer_path):
        print("Loading pre-trained models...")
        classifier = SpamClassifier.load(classifier_path)
        feature_extractor = FeatureExtractor.load(vectorizer_path)
    else:
        print("Training new models...")
        classifier, feature_extractor = train_models()
    
    # Initialize preprocessor
    preprocessor = TextPreprocessor()
    
    # Initialize monitoring
    data_dir = os.path.join(os.path.dirname(__file__), 'data', 'processed')
    os.makedirs(data_dir, exist_ok=True)
    
    metrics_tracker = MetricsTracker(
        storage_path=os.path.join(data_dir, 'metrics.json')
    )
    metrics_tracker.load_state()
    
    drift_detector = DriftDetector(
        storage_path=os.path.join(data_dir, 'drift_state.json')
    )
    drift_detector.load_state()
    
    # Initialize fairness metrics
    fairness_metrics = FairnessMetrics()
    
    # Load sample messages for inbox
    sample_messages = load_sample_messages()
    
    # Initialize API with components
    init_api(
        classifier, feature_extractor, preprocessor,
        metrics_tracker, drift_detector, fairness_metrics,
        sample_messages
    )
    
    # Store in app config for access
    app.config['classifier'] = classifier
    app.config['feature_extractor'] = feature_extractor
    app.config['preprocessor'] = preprocessor
    app.config['metrics_tracker'] = metrics_tracker
    
    print(f"System initialized! Feature count: {feature_extractor.n_features}")


def train_models():
    """Train the spam detection models on the SMS dataset."""
    from sklearn.model_selection import train_test_split
    import numpy as np
    
    print("Loading dataset...")
    
    # Load SMS dataset
    df = pd.read_csv(DATASET_PATH, sep='\t', header=None, 
                     names=['label', 'message'], encoding='latin-1')
    
    # Convert labels
    df['label_num'] = (df['label'] == 'spam').astype(int)
    
    print(f"Dataset size: {len(df)}")
    print(f"Spam: {df['label_num'].sum()}, Ham: {len(df) - df['label_num'].sum()}")
    
    # Preprocess
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
    
    # Train classifier
    print("Training classifier...")
    classifier = SpamClassifier()
    classifier.fit(X_train_features, y_train.values)
    
    # Evaluate
    metrics = classifier.evaluate(X_test_features, y_test.values)
    print(f"Test Accuracy: {metrics['accuracy']:.4f}")
    print(f"Test Precision: {metrics['precision']:.4f}")
    print(f"Test Recall: {metrics['recall']:.4f}")
    print(f"Test F1: {metrics['f1']:.4f}")
    
    # Save models
    print("Saving models...")
    feature_extractor.save()
    classifier.save()
    
    return classifier, feature_extractor


def load_sample_messages(limit=500):
    """Load sample messages from the dataset for inbox display."""
    try:
        df = pd.read_csv(DATASET_PATH, sep='\t', header=None,
                         names=['label', 'message'], encoding='latin-1')
        
        # Get a mix of spam and ham
        spam = df[df['label'] == 'spam'].sample(min(100, len(df[df['label'] == 'spam'])))
        ham = df[df['label'] == 'ham'].sample(min(400, len(df[df['label'] == 'ham'])))
        
        combined = pd.concat([spam, ham]).sample(frac=1)  # Shuffle
        
        return [(row['label'], row['message']) for _, row in combined.iterrows()]
    except Exception as e:
        print(f"Error loading sample messages: {e}")
        return []


# Create the application
app = create_app()


if __name__ == '__main__':
    print("\n" + "="*60)
    print("    SPAM DETECTION SYSTEM")
    print("="*60)
    print("\nStarting server...")
    print("Open http://localhost:5000 in your browser")
    print("Dashboard: http://localhost:5000/dashboard")
    print("API Health: http://localhost:5000/api/health")
    print("\nPress Ctrl+C to stop\n")
    
    app.run(debug=DEBUG, host='0.0.0.0', port=5000)
