import boto3
import json

def test_endpoint():
    client = boto3.client('sagemaker-runtime', region_name='us-east-1')
    
    test_data = {
        'instances': [
            [1.0, 2.0, 3.0, 4.0],
            [0.5, 1.5, 2.5, 3.5],
            [-1.0, 0.0, 1.0, 2.0],
            [2.5, 3.0, 1.5, 0.5]
        ]
    }
    
    try:
        print("Testing SageMaker endpoint...")
        print(f"Input data: {test_data}")
        
        response = client.invoke_endpoint(
            EndpointName='simple-ml-endpoint',
            ContentType='application/json',
            Body=json.dumps(test_data)
        )
        
        result = json.loads(response['Body'].read().decode())
        
        print("\nSuccess!")
        print(f"Predictions: {result['predictions']}")
        
        print("\nInput -> Output:")
        for i, (features, prediction) in enumerate(zip(test_data['instances'], result['predictions'])):
            print(f"  Sample {i+1}: {features} -> Class {prediction}")
            
    except Exception as e:
        print(f"Error: {e}")
        print("\nTroubleshooting:")
        print("1. Check endpoint status: aws sagemaker describe-endpoint --endpoint-name simple-ml-endpoint")
        print("2. Verify AWS credentials")
        print("3. Check region (us-east-1)")

if __name__ == "__main__":
    test_endpoint()