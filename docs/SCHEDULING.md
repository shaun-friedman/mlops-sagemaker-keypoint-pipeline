# Scheduling the keypoint quality check

Part of the [SageMaker MLOps pipeline](../README.md). `6-1_Monitoring_Schedules.ipynb` runs the keypoint quality check on an hourly EventBridge schedule when `SCHEDULER_ROLE_ARN` is set. EventBridge Scheduler needs a role it can assume, and the Studio execution role needs permission to create the schedule and pass that role:

```bash
export AWS_DEFAULT_REGION=us-east-1
ACCOUNT=$(aws sts get-caller-identity --query Account --output text)
STUDIO_ROLE=<your SageMaker execution role name>

aws iam create-role --role-name SageMakerPipelineSchedulerRole --assume-role-policy-document \
  '{"Version":"2012-10-17","Statement":[{"Effect":"Allow","Principal":{"Service":"scheduler.amazonaws.com"},"Action":"sts:AssumeRole"}]}'

aws iam put-role-policy --role-name SageMakerPipelineSchedulerRole --policy-name StartKeypointQualityMonitor --policy-document \
  "{\"Version\":\"2012-10-17\",\"Statement\":[{\"Effect\":\"Allow\",\"Action\":\"sagemaker:StartPipelineExecution\",\"Resource\":\"arn:aws:sagemaker:${AWS_DEFAULT_REGION}:${ACCOUNT}:pipeline/KeypointQualityMonitor\"}]}"

aws iam put-role-policy --role-name "$STUDIO_ROLE" --policy-name ScheduleKeypointQualityMonitor --policy-document \
  "{\"Version\":\"2012-10-17\",\"Statement\":[{\"Effect\":\"Allow\",\"Action\":[\"scheduler:CreateSchedule\",\"scheduler:UpdateSchedule\",\"scheduler:GetSchedule\",\"scheduler:DeleteSchedule\"],\"Resource\":\"arn:aws:scheduler:${AWS_DEFAULT_REGION}:${ACCOUNT}:schedule/*\"},{\"Effect\":\"Allow\",\"Action\":\"iam:PassRole\",\"Resource\":\"arn:aws:iam::${ACCOUNT}:role/SageMakerPipelineSchedulerRole\"}]}"

echo "arn:aws:iam::${ACCOUNT}:role/SageMakerPipelineSchedulerRole"   # set this as SCHEDULER_ROLE_ARN
```

To remove the role later, see [TEARDOWN.md](TEARDOWN.md).
