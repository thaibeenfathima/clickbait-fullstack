import os
import sys
from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Add the project to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import optimized model loader for Python 3.14.2
from model_loader_py314 import predict_clickbait, get_sentiment, is_models_available

print("✅ Using PyTorch + Transformers model loader for Python 3.14.2")

app = Flask(__name__)
CORS(app)

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'csv', 'xlsx', 'json', 'xml', 'txt', 'pdf'}

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# ===== API Routes =====

@app.route('/api/analyze', methods=['POST'])
def analyze():
    """
    Analyze a single headline for clickbait and sentiment
    """
    try:
        data = request.get_json()
        headline = data.get('headline', '').strip()
        
        if not headline:
            return jsonify({'error': 'Headline is required'}), 400
        
        # Use model loader (handles fallback internally)
        clickbait_label, clickbait_conf = predict_clickbait(headline)
        sentiment, sentiment_conf = get_sentiment(headline)
        
        return jsonify({
            'headline': headline,
            'clickbait_label': clickbait_label,
            'clickbait_confidence': float(clickbait_conf),
            'sentiment': sentiment,
            'sentiment_confidence': float(sentiment_conf),
            'highlighted_words': [],
            'models_available': is_models_available()
        }), 200
        
    except Exception as e:
        print(f"Error in analyze: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/batch', methods=['POST'])
def batch_analyze():
    """
    Process a batch of headlines from an uploaded file
    """
    try:
        if not MODELS_AVAILABLE:
            return jsonify({
                'error': 'ML models not available. Please install TensorFlow properly.'
            }), 503
        
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        column = request.form.get('column', '').strip()
        
        if not file or not column:
            return jsonify({'error': 'File and column are required'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'error': 'File type not allowed'}), 400
        
        # Save file temporarily
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        try:
            # Load file
            df, _, _ = load_file_to_df(file)
            
            if df.empty:
                return jsonify({'error': 'File is empty or could not be parsed'}), 400
            
            if column not in df.columns:
                return jsonify({'error': f'Column "{column}" not found in file'}), 400
            
            # Process batch
            results = process_batch(df, column)
            
            # Convert to JSON-serializable format
            results_data = results.to_dict('records')
            
            return jsonify(results_data), 200
            
        finally:
            # Clean up uploaded file
            if os.path.exists(filepath):
                os.remove(filepath)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/analytics', methods=['GET'])
def analytics():
    """
    Get analytics/dashboard data
    """
    try:
        # This would normally fetch from a database or cache
        # For now, return placeholder data
        return jsonify({
            'total_headlines': 1247,
            'clickbait_count': 456,
            'non_clickbait_count': 791,
            'sentiment_distribution': [
                {'name': 'Positive', 'value': 467},
                {'name': 'Negative', 'value': 389},
                {'name': 'Neutral', 'value': 391}
            ],
            'daily_analysis': [
                {'day': 'Mon', 'count': 145},
                {'day': 'Tue', 'count': 168},
                {'day': 'Wed', 'count': 152},
                {'day': 'Thu', 'count': 178},
                {'day': 'Fri', 'count': 192},
                {'day': 'Sat', 'count': 138},
                {'day': 'Sun', 'count': 129}
            ]
        }), 200
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/health', methods=['GET'])
def health():
    """
    Health check endpoint
    """
    return jsonify({
        'status': 'healthy',
        'message': 'DeClickify API is running',
        'models_available': is_models_available()
    }), 200

# Error handlers
@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
