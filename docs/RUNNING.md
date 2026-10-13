# Running the pipeline

Part of the [SageMaker MLOps pipeline](../README.md).

## Prerequisites

- An AWS account and a SageMaker Studio domain. The notebooks are written for Studio (Python 3.12).
- An execution role with access to SageMaker, S3, Glue (including `glue:DeleteTable`), Athena and CloudWatch. Optional: `sns:CreateTopic` and `sns:Subscribe` for alarm notifications.
- Instance quotas for `ml.g5.4xlarge` and `ml.g5.2xlarge` training, `ml.g5.4xlarge` and `ml.m5.xlarge` processing, and two `ml.g5.xlarge` endpoints.

## Steps

```bash
git clone https://github.com/shaun-friedman/mlops-sagemaker-keypoint-pipeline.git
cd mlops-sagemaker-keypoint-pipeline
pip install -r requirements.txt
```

Run the notebooks in order:

```
1_Setup → 2_Ingestion → 3_Create_Athena_EDA → 4_Feature_store
       → 5_Classification_Model / 5-1_Keypoint_Regression_Model
       → 6_Monitoring → 6-1_Monitoring_Schedules → 7_Endpoint_Demo
```

Notebooks pass resource names to each other with `%store`. The keypoint pipeline took 59 minutes in the committed run. `6-1_Monitoring_Schedules` takes about two hours, mostly waiting for clock hours to end, because Model Monitor analyzes whole hours. Training, endpoints and monitoring jobs bill while they run; see [TEARDOWN.md](TEARDOWN.md).

The optional hourly schedule for the keypoint quality check needs an extra IAM role; see [SCHEDULING.md](SCHEDULING.md).
