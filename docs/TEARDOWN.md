# Teardown

Part of the [SageMaker MLOps pipeline](../README.md). Running the notebooks creates AWS resources that bill while they exist. The real-time endpoints cost the most. Delete everything when you are done.

The quickest way to remove what one run created is the cleanup cell at the end of `6-1_Monitoring_Schedules.ipynb` (schedules, alarms, keypoint quality pipeline), then the one at the end of `6_Monitoring.ipynb` (endpoints). Each is switched off by default; set its flag to `True` to run it. The commands below cover everything, including feature groups, the model registry and S3.

Some resource names include a UUID or timestamp that changes on every run. For those, the commands below list the resources first; substitute your names into the delete command. Run the steps in order, because monitoring schedules must be removed before the endpoints they watch.

```bash
export AWS_DEFAULT_REGION=us-east-1
ACCOUNT=$(aws sts get-caller-identity --query Account --output text)

# 1. Monitoring schedules. 6-1_Monitoring_Schedules creates cls-quality-hourly-<id>, cls-drift-hourly-<id>
#    and kp-drift-hourly-<id>; earlier versions created keypoint-model-quality-schedule and
#    cls-model-quality-schedule. Deleting a schedule also stops it.
aws sagemaker list-monitoring-schedules --query 'MonitoringScheduleSummaries[].MonitoringScheduleName'
aws sagemaker delete-monitoring-schedule --monitoring-schedule-name <schedule-name>

# 2. Hourly keypoint quality check (only if SCHEDULER_ROLE_ARN was set), alarms and notifications
aws scheduler delete-schedule --name keypoint-quality-hourly
aws cloudwatch describe-alarms --alarm-name-prefix cls- --query 'MetricAlarms[].AlarmName'
aws cloudwatch describe-alarms --alarm-name-prefix kp- --query 'MetricAlarms[].AlarmName'
aws cloudwatch delete-alarms --alarm-names <alarm-name> [<alarm-name> ...]
aws cloudwatch delete-alarms --alarm-names KeypointModel-MSE-Violation ClsModel-Accuracy-Violation   # earlier versions
aws sns delete-topic --topic-arn "arn:aws:sns:${AWS_DEFAULT_REGION}:${ACCOUNT}:cat-pipeline-model-monitoring-alerts"

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
aws sagemaker delete-pipeline --pipeline-name KeypointQualityMonitor

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
BUCKET="sagemaker-${AWS_DEFAULT_REGION}-${ACCOUNT}"
aws s3 rm "s3://${BUCKET}/cat-landmarks-project/" --recursive
aws s3 rm "s3://${BUCKET}/KeypointPipeline/" --recursive
aws s3 rm "s3://${BUCKET}/CatClassifierPipeline/" --recursive
aws s3 rm "s3://${BUCKET}/KeypointQualityMonitor/" --recursive
aws s3 ls "s3://${BUCKET}/"   # review what remains, e.g. pytorch-inference-<timestamp>/ model repacks

# 8. Confirm nothing billable is left
aws sagemaker list-endpoints --query 'Endpoints[].EndpointName'
aws sagemaker list-monitoring-schedules --query 'MonitoringScheduleSummaries[].MonitoringScheduleName'
aws sagemaker list-feature-groups --query 'FeatureGroupSummaries[].FeatureGroupName'
```

If you created the scheduler role, remove it too:

```bash
aws iam delete-role-policy --role-name SageMakerPipelineSchedulerRole --policy-name StartKeypointQualityMonitor
aws iam delete-role --role-name SageMakerPipelineSchedulerRole
```

Also stop any running SageMaker Studio JupyterLab spaces. The notebooks were run on a GPU instance, which bills while the space is running.
