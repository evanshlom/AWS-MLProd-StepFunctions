import os
import boto3
import joblib
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from datetime import datetime

def train():
    # Get S3 bucket from environment
    bucket = os.environ.get('S3_MODEL_BUCKET')
    if not bucket:
        raise ValueError("S3_MODEL_BUCKET environment variable not set")
    
    # Generate dataset
    X, y = make_classification(n_samples=1000, n_features=4, n_classes=2, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train model
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X_train, y_train)
    accuracy = model.score(X_test, y_test)
    
    # Save model locally
    local_model_path = '/tmp/model.joblib'
    joblib.dump(model, local_model_path)
    
    # Upload to S3
    s3 = boto3.client('s3')
    timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
    s3_key = f'models/model-{timestamp}.joblib'
    
    s3.upload_file(local_model_path, bucket, s3_key)
    
    # Output for Step Functions
    output = {
        'model_s3_path': f's3://{bucket}/{s3_key}',
        'accuracy': accuracy,
        'timestamp': timestamp
    }
    
    print(f"Model trained. Accuracy: {accuracy:.3f}")
    print(f"Model saved to: {output['model_s3_path']}")
    
    # Write output for Step Functions to read
    with open('/tmp/training_output.txt', 'w') as f:
        f.write(f"{bucket}|{s3_key}|{timestamp}")

if __name__ == "__main__":
    train()