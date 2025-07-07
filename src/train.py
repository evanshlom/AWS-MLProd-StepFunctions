import os
import boto3
import joblib
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from datetime import datetime
import time

def train():
    # Get S3 bucket from environment
    bucket = os.environ.get('S3_MODEL_BUCKET')
    if not bucket:
        raise ValueError("S3_MODEL_BUCKET environment variable not set")
    
    # Get timestamp from environment (passed by Step Functions)
    timestamp = os.environ.get('TIMESTAMP')
    if not timestamp:
        # Fallback to generating timestamp if not provided
        timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
    
    # Generate dataset
    X, y = make_classification(n_samples=1000, n_features=4, n_classes=2, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train model
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X_train, y_train)
    accuracy = model.score(X_test, y_test)
    
    # Get current version from DynamoDB and increment
    dynamodb = boto3.client('dynamodb')
    try:
        response = dynamodb.get_item(
            TableName='MLModelRegistry',
            Key={'model_type': {'S': 'simple-ml'}}
        )
        current_version = int(response.get('Item', {}).get('latest_version', {}).get('N', '0'))
    except:
        current_version = 0
    
    new_version = current_version + 1
    
    # Save model locally
    local_model_path = '/tmp/model.joblib'
    joblib.dump(model, local_model_path)
    
    # Upload to S3 with version number
    s3 = boto3.client('s3')
    s3_key = f'models/model-v{new_version}.joblib'
    
    s3.upload_file(local_model_path, bucket, s3_key)
    
    # Write to DynamoDB with version number
    dynamodb.put_item(
        TableName='MLModelRegistry',
        Item={
            'model_type': {'S': 'simple-ml'},
            'latest_version': {'N': str(new_version)},
            's3_path': {'S': f's3://{bucket}/{s3_key}'},
            'accuracy': {'N': str(accuracy)},
            'timestamp': {'S': timestamp},
            'version': {'N': str(new_version)}
        }
    )
    
    print(f"Model trained. Accuracy: {accuracy:.3f}")
    print(f"Model saved to: s3://{bucket}/{s3_key}")
    print(f"Model registered in DynamoDB with version: {new_version}")

if __name__ == "__main__":
    train()