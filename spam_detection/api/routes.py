"""
REST API Routes for Spam Detection System

Endpoints:
- POST /api/classify - Classify a message
- POST /api/feedback - Submit user feedback
- GET /api/metrics - Get current metrics
- GET /api/messages - Get sample messages for inbox
- GET /api/fairness - Get fairness metrics
"""
import os
import uuid
from flask import Blueprint, request, jsonify
import numpy as np

api_bp = Blueprint('api', __name__, url_prefix='/api')

# Global references (set by app.py)
classifier = None
feature_extractor = None
preprocessor = None
metrics_tracker = None
drift_detector = None
fairness_metrics = None
sample_messages = []


def init_api(clf, fe, pp, mt, dd, fm, msgs):
    """Initialize API with trained components."""
    global classifier, feature_extractor, preprocessor
    global metrics_tracker, drift_detector, fairness_metrics, sample_messages
    
    classifier = clf
    feature_extractor = fe
    preprocessor = pp
    metrics_tracker = mt
    drift_detector = dd
    fairness_metrics = fm
    sample_messages = msgs


@api_bp.route('/classify', methods=['POST'])
def classify_message():
    """
    Classify a message as spam or ham.
    
    Request body:
        {
            "message": "Your message text here"
        }
    
    Response:
        {
            "id": "unique-message-id",
            "prediction": "spam" or "ham",
            "confidence": 0.95,
            "spam_probability": 0.95,
            "ham_probability": 0.05
        }
    """
    if classifier is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    data = request.get_json()
    if not data or 'message' not in data:
        return jsonify({'error': 'Missing message field'}), 400
    
    message = data['message']
    message_id = str(uuid.uuid4())
    
    # Preprocess
    processed = preprocessor.preprocess(message)
    
    # Extract features
    features = feature_extractor.transform([processed])
    
    # Predict
    result = classifier.predict_with_confidence(features)[0]
    
    # Record for metrics
    if metrics_tracker:
        metrics_tracker.record_prediction(
            message_id=message_id,
            message_preview=message[:100],
            prediction=result['label'],
            confidence=result['confidence']
        )
    
    # Record for drift detection
    if drift_detector:
        drift_detector.record_prediction(
            prediction=result['label'],
            confidence=result['confidence'],
            message_length=len(message)
        )
    
    return jsonify({
        'id': message_id,
        'prediction': result['prediction'],
        'confidence': result['confidence'],
        'spam_probability': result['spam_probability'],
        'ham_probability': result['ham_probability']
    })


@api_bp.route('/classify/batch', methods=['POST'])
def classify_batch():
    """
    Classify multiple messages.
    
    Request body:
        {
            "messages": ["message1", "message2", ...]
        }
    
    Response:
        {
            "results": [
                {"id": "...", "prediction": "...", ...},
                ...
            ]
        }
    """
    if classifier is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    data = request.get_json()
    if not data or 'messages' not in data:
        return jsonify({'error': 'Missing messages field'}), 400
    
    messages = data['messages']
    
    # Preprocess all
    processed = preprocessor.preprocess_batch(messages)
    
    # Extract features
    features = feature_extractor.transform(processed)
    
    # Predict
    results = classifier.predict_with_confidence(features)
    
    # Add IDs
    for i, result in enumerate(results):
        result['id'] = str(uuid.uuid4())
        result['original_message'] = messages[i][:100]
    
    return jsonify({'results': results})


@api_bp.route('/feedback', methods=['POST'])
def submit_feedback():
    """
    Submit user feedback on a classification.
    
    Request body:
        {
            "message_id": "id from classify response",
            "original_prediction": 1,
            "correct_label": 0,
            "message": "optional - original message for retraining"
        }
    
    Response:
        {
            "status": "recorded",
            "feedback_count": 42
        }
    """
    data = request.get_json()
    if not data:
        return jsonify({'error': 'Missing request body'}), 400
    
    message_id = data.get('message_id', str(uuid.uuid4()))
    original_prediction = data.get('original_prediction')
    correct_label = data.get('correct_label')
    
    if original_prediction is None or correct_label is None:
        return jsonify({'error': 'Missing prediction or label'}), 400
    
    # Record feedback
    if metrics_tracker:
        metrics_tracker.record_feedback(
            message_id=message_id,
            original_prediction=original_prediction,
            user_label=correct_label
        )
    
    if drift_detector:
        drift_detector.record_feedback(
            predicted=original_prediction,
            actual=correct_label
        )
    
    return jsonify({
        'status': 'recorded',
        'feedback_count': metrics_tracker.feedback_received if metrics_tracker else 0
    })


@api_bp.route('/messages', methods=['GET'])
def get_messages():
    """
    Get sample messages for inbox display.
    
    Query params:
        - limit: Number of messages (default: 20)
        - offset: Pagination offset (default: 0)
        - filter: 'all', 'spam', 'ham' (default: 'all')
    
    Response:
        {
            "messages": [
                {
                    "id": "...",
                    "content": "message text",
                    "prediction": "spam",
                    "confidence": 0.95,
                    "timestamp": "2024-01-01T12:00:00"
                },
                ...
            ],
            "total": 100
        }
    """
    limit = request.args.get('limit', 20, type=int)
    offset = request.args.get('offset', 0, type=int)
    filter_type = request.args.get('filter', 'all')
    
    # Get messages and classify them
    messages_subset = sample_messages[offset:offset+limit]
    
    if not messages_subset:
        return jsonify({'messages': [], 'total': len(sample_messages)})
    
    # Classify if we have the model
    if classifier and feature_extractor and preprocessor:
        processed = preprocessor.preprocess_batch([m[1] for m in messages_subset])
        features = feature_extractor.transform(processed)
        predictions = classifier.predict_with_confidence(features)
        
        results = []
        for i, (label, content) in enumerate(messages_subset):
            pred = predictions[i]
            
            # Apply filter
            if filter_type == 'spam' and pred['prediction'] != 'spam':
                continue
            if filter_type == 'ham' and pred['prediction'] != 'ham':
                continue
            
            results.append({
                'id': str(uuid.uuid4()),
                'content': content,
                'true_label': label,
                'prediction': pred['prediction'],
                'confidence': pred['confidence'],
                'spam_probability': pred['spam_probability']
            })
    else:
        results = [
            {
                'id': str(uuid.uuid4()),
                'content': content,
                'true_label': label,
                'prediction': label,
                'confidence': 1.0,
                'spam_probability': 1.0 if label == 'spam' else 0.0
            }
            for label, content in messages_subset
        ]
    
    return jsonify({
        'messages': results,
        'total': len(sample_messages)
    })


@api_bp.route('/metrics', methods=['GET'])
def get_metrics():
    """
    Get current system metrics.
    
    Response:
        {
            "metrics": {
                "accuracy": 0.95,
                "precision": 0.92,
                "recall": 0.98,
                ...
            },
            "recent_predictions": [...],
            "alerts": [...]
        }
    """
    if metrics_tracker is None:
        return jsonify({'error': 'Metrics tracker not initialized'}), 500
    
    dashboard_data = metrics_tracker.get_dashboard_data()
    
    # Add drift detection results
    if drift_detector:
        dashboard_data['drift'] = drift_detector.check_drift()
    
    return jsonify(dashboard_data)


@api_bp.route('/fairness', methods=['GET'])
def get_fairness():
    """
    Get fairness metrics.
    
    Response:
        {
            "disparate_impact": {...},
            "equal_opportunity": {...},
            "equalized_odds": {...},
            "predictive_parity": {...}
        }
    """
    # This would require test data - return cached or computed metrics
    return jsonify({
        'status': 'Fairness metrics available after evaluation',
        'message': 'Use /api/evaluate-fairness with test data'
    })


@api_bp.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'model_loaded': classifier is not None,
        'feature_extractor_loaded': feature_extractor is not None
    })


@api_bp.route('/model/info', methods=['GET'])
def model_info():
    """Get information about the loaded model."""
    info = {
        'model_type': 'LogisticRegression (SpamClassifier)',
        'threshold': classifier.threshold if classifier else None,
        'n_features': feature_extractor.n_features if feature_extractor else None,
    }
    
    if classifier and feature_extractor:
        try:
            feature_names = feature_extractor.get_feature_names()
            importance = classifier.get_feature_importance(feature_names, top_n=10)
            info['top_spam_features'] = [f[0] for f in importance['spam_features']]
            info['top_ham_features'] = [f[0] for f in importance['ham_features']]
        except:
            pass
    
    return jsonify(info)
