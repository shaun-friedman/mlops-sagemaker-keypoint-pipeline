# SageMaker MLOps Pipeline: Cat Detection and Facial Keypoint Regression

> **Platform:** AWS SageMaker  
> **Scope:** Graduate coursework, built by a two-person team for AAI 540 (Machine Learning Operations) in the University of San Diego M.S. in Applied Artificial Intelligence program, January–February 2026. It is a learning project run in a single AWS account, not a production system. Changes made after the course ended are visible in the git history.

---

## Overview

This project implements an end-to-end MLOps pipeline for **cat pose estimation and action classification** using AWS cloud services. The pipeline covers everything from raw data ingestion through model training, monitoring, and live endpoint deployment — following production-grade MLOps best practices.

The system combines two complementary models:
- A **keypoint regression model** that predicts cat facial keypoint coordinates from image data
- A **classification model** that maps those keypoints (or raw features) to action/pose class labels

---

## Repository Structure

```
mlops-sagemaker-keypoint-pipeline/
├── src/                              # Supporting Python modules and scripts
├── 1_Setup.ipynb                     # AWS environment setup (IAM, S3, SageMaker config)
├── 2_Ingestion.ipynb                 # Data ingestion and S3 upload
├── 3_Create_Athena_EDA.ipynb         # Athena table creation and exploratory data analysis
├── 4_Feature_store.ipynb             # SageMaker Feature Store ingestion and management
├── 5_Classification_Model.ipynb      # Training and evaluation of the classification model
├── 5-1_Keypoint_Regression_Model.ipynb  # Training and evaluation of the keypoint regression model
├── 6_Monitoring.ipynb                # SageMaker Model Monitor setup and drift detection
├── 7_Endpoint_Demo.ipynb             # Live endpoint deployment and inference demo
├── docs/figures/                     # Figures referenced in this README
├── requirements.txt                  # Python dependencies
├── LICENSE
└── README.md
```

---

## Pipeline Walkthrough

### 1. Setup (`1_Setup.ipynb`)
Configures the AWS environment: IAM roles, S3 bucket creation, and SageMaker session initialization. Run this notebook first to establish all prerequisites.

### 2. Data Ingestion (`2_Ingestion.ipynb`)
Loads the raw dataset (pose/keypoint images and annotations) and uploads it to S3 for downstream processing.

### 3. Athena EDA (`3_Create_Athena_EDA.ipynb`)
Creates an AWS Glue/Athena table over the S3 data and performs exploratory data analysis — querying distributions, inspecting label balance, and visualizing keypoint statistics.

### 4. Feature Store (`4_Feature_store.ipynb`)
Ingests processed features into the **SageMaker Feature Store**, enabling reproducible, versioned feature retrieval for model training and serving.

### 5a. Classification Model (`5_Classification_Model.ipynb`)
Trains a classifier to detect cats in frame. Covers data preparation, training job configuration, hyperparameter settings, and evaluation metrics.

### 5b. Keypoint Regression Model (`5-1_Keypoint_Regression_Model.ipynb`)
Trains a regression model to predict (x, y) coordinates for cat facial keypoints. Covers model architecture, training, and evaluation (e.g., RMSE, normalized error).

### 6. Monitoring (`6_Monitoring.ipynb`)
Sets up **SageMaker Model Monitor** to capture inference traffic, establish data quality baselines, and schedule drift detection jobs against the deployed endpoint.

### 7. Endpoint Demo (`7_Endpoint_Demo.ipynb`)
Deploys the trained model(s) to a SageMaker real-time endpoint and demonstrates end-to-end inference — sending sample inputs and visualizing predictions.

---

## Getting Started

### Prerequisites

- Python 3.12 (the notebooks use 3.12-only f-string syntax)
- An AWS account with SageMaker, S3, Glue, and Athena access
- Appropriate IAM role with SageMaker execution permissions
- AWS CLI configured locally (or run notebooks inside SageMaker Studio)

### Installation

```bash
git clone https://github.com/shaun-friedman/mlops-sagemaker-keypoint-pipeline.git
cd mlops-sagemaker-keypoint-pipeline
pip install -r requirements.txt
```

### Running the Pipeline

Execute the notebooks **in order**:

```
1_Setup → 2_Ingestion → 3_Create_Athena_EDA → 4_Feature_store
       → 5_Classification_Model / 5-1_Keypoint_Regression_Model
       → 6_Monitoring → 7_Endpoint_Demo
```

> **Tip:** The notebooks are designed to be run inside **Amazon SageMaker Studio** for seamless AWS integration. Running locally requires valid AWS credentials and may incur costs for SageMaker training jobs and endpoints.

---

## AWS Services Used

| Service | Purpose |
|---|---|
| Amazon S3 | Raw data and artifact storage |
| AWS Glue + Amazon Athena | Metadata catalog and SQL-based EDA |
| SageMaker Feature Store | Centralized, versioned feature management |
| SageMaker Training Jobs | Managed model training (classification & regression) |
| SageMaker Endpoints | Real-time model serving |
| SageMaker Model Monitor | Data quality and drift monitoring |

---

## Models

| Model | Type | Task |
|---|---|---|
| Classification Model | Multi-class classifier | Predict pose/action label from features |
| Keypoint Regression Model | Coordinate regression | Predict (x, y) body keypoint locations |

---

## Example Output

![Chained inference: classifier, then keypoint regressor](docs/figures/chained_inference.jpg)

Chained inference on held-out test images, from `7_Endpoint_Demo.ipynb`. Each image goes to the classifier endpoint first. Images predicted as cats are then sent to the keypoint endpoint, and its predicted keypoints are drawn in red. These are model predictions, not ground truth. Ear points are visibly less accurate than eyes and mouth.

---

## Team and my contributions

This was a two-person team: **Shaun Friedman** and **Deepti Pamula**. The split below comes from the git history.

**Deepti Pamula**
- Created the repository and wrote the first versions of notebooks 1–4 (setup, ingestion, Athena EDA, Feature Store). Most of `1_Setup.ipynb` is still her code.
- Built the first cat vs. non-cat classifier pipeline: a small CNN, heuristic and brightness benchmarks, the training loop, and the endpoint deployment pattern used by the original classifier notebook.
- Set up the first endpoint data capture and test traffic, which `6_Monitoring.ipynb` builds on.
- Wrote the original SageMaker training script for the small CNN (`src/train.py`, since replaced; see git history).

**Shaun Friedman (my contributions)**
- `2_Ingestion.ipynb`: rewrote ingestion, covering Kaggle cat download and annotation parsing, image augmentation, the COCO non-cat images, the train/validation/test/production split, and S3 upload. Also made every notebook work with the account's default bucket.
- `3_Create_Athena_EDA.ipynb` and `4_Feature_store.ipynb`: refactored the Glue/Athena tables, EDA queries and Feature Store ingestion.
- `5_Classification_Model.ipynb` (joint with Deepti): split the classifier out of her combined notebook, replaced the small CNN with a pretrained ResNet-18, then retrained, deployed and evaluated it.
- `5-1_Keypoint_Regression_Model.ipynb` and all of `src/keypoint_regression/`: built the keypoint model end to end as a SageMaker Pipeline with preprocessing, training, evaluation, a quality gate and model registration.
- `6_Monitoring.ipynb`: rebuilt monitoring. Deployed both models with data capture, ran Model Monitor baseline jobs for both, and created monitoring schedules and CloudWatch alarms.
- `7_Endpoint_Demo.ipynb`: the chained classifier-to-keypoint inference demo.
- This README.

**After the course** (visible in the git history)
- Rebuilt the classifier as a SageMaker Pipeline matching the keypoint model's: preprocessing, managed training, evaluation, an F1 quality gate and model registration (`src/classification/`).
- Registered both models with the PyTorch inference image so `6_Monitoring.ipynb` deploys them directly from the model registry.
- Fixed left/right keypoint labels on horizontally flipped training images.
- Added normalized mean error (NME) to the keypoint evaluation, and replaced the monitoring baselines with model-quality and prediction-drift baselines computed from endpoint traffic tagged with inference IDs.

---

## Teardown

Running the notebooks creates AWS resources that bill while they exist. The real-time endpoints cost the most. Delete everything when you are done.

Some resource names include a UUID or timestamp that changes on every run. For those, the commands below list the resources first; substitute your names into the delete command. Run the steps in order, because monitoring schedules must be removed before the endpoints they watch.

```bash
export AWS_DEFAULT_REGION=us-east-1

# 1. Monitoring schedules and alarms created by earlier versions of 6_Monitoring
#    (deleting a schedule also stops it)
aws sagemaker delete-monitoring-schedule --monitoring-schedule-name keypoint-model-quality-schedule
aws sagemaker delete-monitoring-schedule --monitoring-schedule-name cls-model-quality-schedule

# 2. CloudWatch alarms
aws cloudwatch delete-alarms --alarm-names KeypointModel-MSE-Violation ClsModel-Accuracy-Violation

# 3. Endpoints, endpoint configs and models
#    6_Monitoring creates keypoint-regression-<uuid> and cls-<uuid> plus a model for each, and the keypoint
#    pipeline creates a model. Earlier versions of 5_Classification_Model also created pytorch-inference-<uuid>
#    (config catcls-epc-<uuid>, model catcls-model-<uuid>).
aws sagemaker list-endpoints --query 'Endpoints[].EndpointName'
aws sagemaker delete-endpoint --endpoint-name <endpoint-name>
aws sagemaker list-endpoint-configs --query 'EndpointConfigs[].EndpointConfigName'
aws sagemaker delete-endpoint-config --endpoint-config-name <endpoint-config-name>
aws sagemaker list-models --query 'Models[].ModelName'
aws sagemaker delete-model --model-name <model-name>

# 4. Model registry and pipelines
for group in KeypointModelPackageGroupName CatClassifierModelPackageGroup; do
  for arn in $(aws sagemaker list-model-packages --model-package-group-name "$group" \
                 --query 'ModelPackageSummaryList[].ModelPackageArn' --output text); do
    aws sagemaker delete-model-package --model-package-name "$arn"
  done
  aws sagemaker delete-model-package-group --model-package-group-name "$group"
done
aws sagemaker delete-pipeline --pipeline-name KeypointPipeline
aws sagemaker delete-pipeline --pipeline-name CatClassifierPipeline

# 5. Feature Store (the online store bills while the feature group exists)
aws sagemaker list-feature-groups --name-contains landmarks-feature-group \
  --query 'FeatureGroupSummaries[].FeatureGroupName'
aws sagemaker delete-feature-group --feature-group-name <feature-group-name>

# 6. Glue Data Catalog. Deleting a feature group leaves its offline-store table behind.
aws glue delete-database --name cat_landmarking
aws glue get-tables --database-name sagemaker_featurestore \
  --query "TableList[?starts_with(Name, 'landmarks_feature_group')].Name"
aws glue delete-table --database-name sagemaker_featurestore --name <table-name>

# 7. S3 data and artifacts in the SageMaker default bucket
BUCKET="sagemaker-${AWS_DEFAULT_REGION}-$(aws sts get-caller-identity --query Account --output text)"
aws s3 rm "s3://${BUCKET}/cat-landmarks-project/" --recursive
aws s3 rm "s3://${BUCKET}/KeypointPipeline/" --recursive
aws s3 rm "s3://${BUCKET}/CatClassifierPipeline/" --recursive
aws s3 ls "s3://${BUCKET}/"   # review what remains, e.g. pytorch-inference-<timestamp>/ model repacks

# 8. Confirm nothing billable is left
aws sagemaker list-endpoints --query 'Endpoints[].EndpointName'
aws sagemaker list-monitoring-schedules --query 'MonitoringScheduleSummaries[].MonitoringScheduleName'
aws sagemaker list-feature-groups --query 'FeatureGroupSummaries[].FeatureGroupName'
```

Also stop any running SageMaker Studio JupyterLab spaces. The notebooks were run on a GPU instance, which bills while the space is running.

---

## License

Released under the [MIT License](LICENSE).