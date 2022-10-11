

import os
import json
import boto3
import sagemaker
import sagemaker.session

from sagemaker.estimator import Estimator
from sagemaker.inputs import TrainingInput
from sagemaker.model_metrics import (
    MetricsSource,
    ModelMetrics,
)
from sagemaker.processing import (
    ProcessingInput,
    ProcessingOutput,
    ScriptProcessor,
)
from sagemaker.sklearn.processing import SKLearnProcessor
from sagemaker.workflow.conditions import ConditionLessThanOrEqualTo, ConditionGreaterThanOrEqualTo
from sagemaker.workflow.condition_step import (
    ConditionStep,
    JsonGet,
)
from sagemaker.workflow.parameters import (
    ParameterInteger,
    ParameterString,
)
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.properties import PropertyFile
from sagemaker.workflow.steps import (
    ProcessingStep,
    TrainingStep,
)
from sagemaker.workflow.step_collections import RegisterModel
from sagemaker.sklearn.estimator import SKLearn
from sagemaker.sklearn.model import SKLearnModel
from sagemaker.workflow.functions import JsonGet
BASE_DIR = os.path.dirname(os.path.realpath(__file__))
BUCKET_NAME = "refinedcsv"

def get_sagemaker_client(region):
     """Gets the sagemaker client.

        Args:
            region: the aws region to start the session
            default_bucket: the bucket to use for storing the artifacts

        Returns:
            `sagemaker.session.Session instance
        """
     boto_session = boto3.Session(region_name=region)
     sagemaker_client = boto_session.client("sagemaker")
     return sagemaker_client


def get_session(region, default_bucket):
    """Gets the sagemaker session based on the region.

    Args:
        region: the aws region to start the session
        default_bucket: the bucket to use for storing the artifacts

    Returns:
        `sagemaker.session.Session instance
    """

    boto_session = boto3.Session(region_name=region)

    sagemaker_client = boto_session.client("sagemaker")
    runtime_client = boto_session.client("sagemaker-runtime")
    return sagemaker.session.Session(
        boto_session=boto_session,
        sagemaker_client=sagemaker_client,
        sagemaker_runtime_client=runtime_client,
        default_bucket=default_bucket,
    )

def get_pipeline_custom_tags(new_tags, region, sagemaker_project_arn=None):
    try:
        sm_client = get_sagemaker_client(region)
        response = sm_client.list_tags(
            ResourceArn=sagemaker_project_arn)
        project_tags = response["Tags"]
        for project_tag in project_tags:
            new_tags.append(project_tag)
    except Exception as e:
        print(f"Error getting project tags: {e}")
    return new_tags


def get_pipeline(
    region,
    sagemaker_project_arn=None,
    role=None,
    default_bucket=None,
    model_package_group_name="AbalonePackageGroup",
    pipeline_name="AbalonePipeline",
    base_job_prefix="Abalone",
):
    """Gets a SageMaker ML Pipeline instance working with on abalone data.

    Args:
        region: AWS region to create and run the pipeline.
        role: IAM role to create and run steps and pipeline.
        default_bucket: the bucket to use for storing the artifacts

    Returns:
        an instance of a pipeline
    """
    sagemaker_session = get_session(region, default_bucket)
    if role is None:
        role = sagemaker.session.get_execution_role(sagemaker_session)

    # parameters for pipeline execution
#     processing_instance_count = ParameterInteger(name="ProcessingInstanceCount", default_value=1)
#     processing_instance_type = ParameterString(
#         name="ProcessingInstanceType", default_value="ml.m5.xlarge"
#     )
#     training_instance_type = ParameterString(
#         name="TrainingInstanceType", default_value="ml.m5.xlarge"
#     )
#     model_approval_status = ParameterString(
#         name="ModelApprovalStatus", default_value="PendingManualApproval"
#     )
#     input_data = ParameterString(
#         name="InputDataBucket",
#         default_value="s3://smartapps-studio-data-bucket",
#     )
#     output_data = ParameterString(
#         name="OutputDataBucket",
#         default_value="s3://smartapps-studio-model-building-bucket/data/",
#     )

    
#     config_path = ParameterString(
#         name="ConfigPath",
#         default_value="s3://smartapps-model-building-bucket/configs/configs.json",
#     )

#     with open(config_path, 'r') as f:
#         configs = json.load(f)


    s3_client = boto3.resource('s3')
    content_object = s3_client.Object(BUCKET_NAME, 'configs/configs.json')
    file_content = content_object.get()['Body'].read().decode('utf-8')
    configs = json.loads(file_content)
    
    processing_instance_count = configs.get('ProcessingInstanceCount')
    processing_instance_type = configs.get('ProcessingInstanceType')
    training_instance_type = configs.get('TrainingInstanceType')
    model_approval_status = configs.get('ModelApprovalStatus')
    input_data = configs.get('InputDataBucket')
    output_data = configs.get('OutputDataBucket')

    
    # processing step for feature engineering
    sklearn_processor = SKLearnProcessor(
        framework_version="0.23-1",
        instance_type=processing_instance_type,
        instance_count=processing_instance_count,
        base_job_name=f"{base_job_prefix}/sklearn-abalone-preprocess",
        sagemaker_session=sagemaker_session,
        role=role)
    
    step_process = ProcessingStep(
        name="DataProcessing",
        processor=sklearn_processor,
        code=os.path.join(BASE_DIR, "preprocess.py"),
        job_arguments=["--input-data", input_data, "--output-data", output_data]
    )
    #job_arguments=["--input-data", input_data, "--output-data", "s3://smartapps-studio-model-building-bucket/data/"]
    
    #------------------------------------------------
    
    # training step for generating model artifacts
    
    
    
    FRAMEWORK_VERSION = "0.23-1"
    script_path = "inference.py"
    
    sklearn = SKLearn(
        entry_point= script_path,
        framework_version=FRAMEWORK_VERSION,
        instance_type="ml.m4.xlarge",
        role=role,
        sagemaker_session=sagemaker_session,
        source_dir = "s3://"+BUCKET_NAME+"/model/model_trainer.tar.gz",
    )

    step_train = TrainingStep(
        name="TrainModel",
        estimator=sklearn,
    )
    
#    step_train.add_depends_on([step_process])
    
    
    
    sklearn_model = SKLearnModel(
    model_data =  "s3://"+BUCKET_NAME+"/model_artifact/model.tar.gz",
    source_dir= "s3://"+BUCKET_NAME+"/model/model_trainer.tar.gz",
    framework_version=FRAMEWORK_VERSION,
    role=role,
    entry_point='inference.py',
    sagemaker_session=sagemaker_session,
    py_version='py3')
    
    step_register = RegisterModel(
        name="RegisterModel",
        model=sklearn_model,
        content_types=["application/json"],
        response_types=["application/json"],

        inference_instances=["ml.t2.large"],
        transform_instances=["ml.m4.xlarge"],
        model_package_group_name=model_package_group_name,
        approval_status="Approved",
    )

#     pipeline = Pipeline(
#         name=pipeline_name,
#         parameters=[
#             processing_instance_type,
#             processing_instance_count,
#             training_instance_type,
#             model_approval_status,
#             input_data,
#             output_data,
#         ],
#         steps=[step_process, step_train, step_register],
#         sagemaker_session=sagemaker_session,
#     )


    
    
#     script_eval = ScriptProcessor(
#         image_uri=image_uri,
#         command=["python3"],
#         instance_type=processing_instance_type,
#         instance_count=1,
#         base_job_name=f"{base_job_prefix}/script-CustomerChurn-eval",
#         sagemaker_session=sagemaker_session,
#         role=role,
#     )
    
    
#     evaluation_report = PropertyFile(
#         name="EvaluationReport",model
#         output_name="evaluation",
#         path="evaluation.json",
#     )

    sklearn_evaluator = SKLearnProcessor(
        framework_version="0.23-1",
        instance_type=processing_instance_type,
        instance_count=processing_instance_count,
        base_job_name=f"{base_job_prefix}/sklearn-abalone-eval",
        sagemaker_session=sagemaker_session,
        role=role)

    evaluation_report = PropertyFile(
        name="EvaluationReport",
        output_name="evaluation",
        path="evaluation_kids.json",
    )
    
    step_eval = ProcessingStep(
        name="Eval",
        processor=sklearn_processor,
        code=os.path.join(BASE_DIR, "evaluate.py"),
        outputs=[ProcessingOutput(output_name="evaluation", source="/opt/ml/processing/evaluation"),],
        property_files=[evaluation_report],
        depends_on = [step_train],
    )

#     evaluation_path = "/opt/ml/processing/evaluation/evaluation.json"
    
#     with open(evaluation_path) as json_file:
#         similarity = json.load(json_file)
    
    cond_lte = ConditionGreaterThanOrEqualTo(  
        left=JsonGet(
            step_name=step_eval.name,
            property_file=evaluation_report,
            json_path="Similarity",
        ),
        right=0.9,  # You can change the threshold here
    )
    
    step_cond = ConditionStep(
        name="SimilarityEvalCondition",
        conditions=[cond_lte],
        if_steps=[step_register],
        else_steps=[],
        depends_on = [step_eval],
    )
    
    pipeline = Pipeline(
        name=pipeline_name,
        parameters=[
        ],
        # steps=[step_process, step_train, step_eval, step_cond],
        steps=[step_train],
        sagemaker_session=sagemaker_session,
    )


#     pipeline = Pipeline(
#         name=pipeline_name,
#         parameters=[
#         ],
#         steps=[step_process, step_train, step_register],
#         sagemaker_session=sagemaker_session,
#     )
    
    return pipeline

