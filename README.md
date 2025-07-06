# AWS-MLProd-StepFunctions
V2 of my other project AWS-MLOps-Pipeline, using CICD to run AWS jobs for ML training and deployment.

Easy ML pipeline with Step Functions to train and deploy ML models using AWS Step Functions, ECS Fargate, and SageMaker.

## Project Structure

```
simple-ml-pipeline/
├── Dockerfile.train
├── Dockerfile.serve
├── README.md
├── requirements.txt
├── .github/
│   └── workflows/
│       └── deploy.yml
├── src/
│   ├── train.py
│   └── inference.py
├── infrastructure/
│   └── step-function.json
└── scripts/
    ├── setup.sh
    └── test_endpoint.py
```

## Setup Instructions

1. **AWS Prerequisites**
   ```bash
   # Create S3 bucket for models
   aws s3 mb s3://your-ml-models-bucket
   
   # Create ECR repositories
   aws ecr create-repository --repository-name ml-training
   aws ecr create-repository --repository-name ml-serving
   ```

2. **GitHub Secrets**
   - `AWS_ACCESS_KEY_ID`
   - `AWS_SECRET_ACCESS_KEY`
   - `SAGEMAKER_ROLE_ARN`
   - `ECS_TASK_ROLE_ARN`
   - `ECS_EXECUTION_ROLE_ARN`
   - `STEP_FUNCTIONS_ROLE_ARN`
   - `S3_MODEL_BUCKET`

3. **Create Step Function**
   ```bash
   aws stepfunctions create-state-machine \
     --name ml-training-pipeline \
     --definition file://infrastructure/step-function.json \
     --role-arn $STEP_FUNCTIONS_ROLE_ARN
   ```

4. **Deploy**
   ```bash
   git push origin main
   ```

## Demo Instructions

1. Push code to trigger training
2. Watch Step Functions console for progress
3. Test endpoint after deployment:
   ```bash
   cd final_test
   docker build -f Dockerfile.test_endpoint -t test-endpoint .
   docker run test-endpoint
   ```

## Architecture

1. GitHub Actions builds Docker images
2. Step Functions orchestrates training workflow
3. ECS Fargate runs training job
4. Model saved to S3
5. SageMaker endpoint deployed

## License

MIT License