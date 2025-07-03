import os
import boto3
import joblib
import numpy as np
from flask import Flask, request, jsonify

app = Flask(__name__)

# Load model from S3 at startup
def load_model():
    model_s3_path = os.environ.get('MODEL_S3_PATH')
    if not model_s3_path:
        raise ValueError("MODEL_S3_PATH environment variable not set")
    
    # Parse S3 path
    parts = model_s3_path.replace('s3://', '').split('/', 1)
    bucket = parts[0]
    key = parts[1]
    
    # Download model
    s3 = boto3.client('s3')
    local_model_path = '/opt/ml/model/model.joblib'
    os.makedirs(os.path.dirname(local_model_path), exist_ok=True)
    
    s3.download_file(bucket, key, local_model_path)
    return joblib.load(local_model_path)

model = load_model()

@app.route('/ping', methods=['GET'])
def ping():
    return jsonify({'status': 'healthy'})

@app.route('/invocations', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        features = np.array(data['instances'])
        predictions = model.predict(features).tolist()
        return jsonify({'predictions': predictions})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)