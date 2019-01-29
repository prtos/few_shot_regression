import os
import json
import time
import boto3, boto
import argparse
import numpy as np
import sagemaker as sage
from os.path import join, splitext
from sagemaker.session import Session
from boto.s3.connection import S3Connection
from sklearn.model_selection import ParameterGrid


def get_role_arn(role_name):
    # set role
    roles = boto3.client('iam').list_roles()['Roles']
    if roles == []:
        print('No role available.')
        exit()
    for r in roles:
        if role_name in r['Arn']:
            arn = r['Arn']
    return arn


def monitor_jobs(running_jobs, session):
    for job in running_jobs:
        # Checking the job current status
        desc = session.sagemaker_client.describe_training_job(TrainingJobName=job)
        status = desc['TrainingJobStatus']

        # Case where there is a problem with the job
        if status in ["Failed", "Stopped"]:
            raise RuntimeError("Job {} has {}. Abandonning dispatching".format(job, status))

        # Case where the job is finished
        elif status == "Completed":
            print("[{}] Job: {}".format(status, job))
            running_jobs.remove(job)
    return running_jobs


def is_hp_already_done(output_path):
    # todo: I don't get the logic of this part of the code.
    # todo: Gael make this more general please
    conn = boto.s3.connect_to_region('us-east-2',
                                     aws_access_key_id=os.environ['AWS_ACCESS_KEY_ID'],
                                     aws_secret_access_key=os.environ['AWS_SECRET_ACCESS_KEY'],
                                     is_secure=True,
                                     calling_format=boto.s3.connection.OrdinaryCallingFormat())
    bucket = conn.get_bucket(bucket_name)

    # Checking which datasets results are already saved by extracting the job name in the experiment output directory structure
    s3_results = [key.name.replace(prefix, "") for key in bucket.list(prefix=prefix)]
    s3_results = [r.split("/")[0] for r in s3_results if r]

    # Extracting the target and aid of the job which has the following pattern : [projet]-[target]-[aid]-[date]
    s3_results = [(r.split("-")[1], r.split("-")[2]) for r in s3_results]

    # Identifying which dataset are left to dispatch
    metatest_datasets_to_process = [(t, a) for (t, a) in metatest_datasets if not((t.replace('_', ''), a) in s3_results)]

    # Datasets which failed on m4.xlarge intances
    problematic_datasets = [('Max_Response', '485364')] # Busted RAM (359839 examples)
    metatest_datasets_to_process = [d for d in metatest_datasets_to_process if not(d in problematic_datasets)]

    return False


if __name__ == '__main__':
    """
    # command for launch:
    python aws_dispatcher.py -n merck -e test-merck-baseline -t ml.p2.xlarge -c 1 -x 76000 \
    -i s3://datasets-ressources/merck -o s3://sagemaker-artifacts/merck
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--imagename', type=str,
                        help='The name of the image that will be used to execute your code')
    parser.add_argument('-e', '--experiment_name', type=str,
                        help='The name of the experiment')
    parser.add_argument('-r', '--role', type=str, default='AmazonSageMaker',
                        help="""An AWS IAM role (either name or full ARN). The Amazon SageMaker training jobs and APIs
                                 that create Amazon SageMaker endpoints use this role to access training data and model
                                 artifacts. After the endpoint is created, the inference code might use the IAM role,
                                 if it needs to access an AWS resource.""")
    parser.add_argument('-t', '--instance_type', type=str, default='local',     # gpu_local
                        help="""Type of EC2 instance to use for training, for example 'ml.c4.xlarge',
                                (default: local).""")
    parser.add_argument('-p', '--config_file', type=str, default='config.json',
                    help="""The name/path of the config file (in json format) that contains all the 
                            parameters for the experiment.""")
    parser.add_argument('-v', '--volume_size', type=int, default=30,
                        help="""Size in GB of the EBS volume to use for storing input data
                                during training (default: 30). Must be large enough to store training data
                                if File Mode is used (which is the default)""")
    parser.add_argument('-x', '--max_runtime', type=int, default=60*60,
                        help="""Timeout in seconds for training (default: 60 * 60).""")
    parser.add_argument('-c', '--max_concurrent_instances', type=int, default=1,
                        help="""Number of Sagemaker instances to use concurrently (default: 1)""")
    parser.add_argument('-i', '--input_path', type=str, default='',
                        help="""S3 location that contains the train files. If instance_type is 'local' 
                                it must be a local emplacement""")
    parser.add_argument('-o', '--output_path', type=str, default='s3://sagemaker-us-east-2-707251684090/',
                        help="""S3 location for saving the training result (model artifacts and output files).
                                If not specified, results are stored to a default bucket. If the bucket with the
                                specific name does not exist, the estimator creates the bucket during the
                                sagemaker.estimator.EstimatorBase.fit method execution.""")

    args = parser.parse_args()
    role_arn = get_role_arn(args.role)
    # Dispatching on Sagemaker instances
    if args.instance_type != 'local':
        session = Session()
        account = session.boto_session.client('sts').get_caller_identity()['Account']
        region = session.boto_session.region_name
        image = '{}.dkr.ecr.{}.amazonaws.com/{}:latest'.format(account, region, args.imagename)
    else:
        session = None
        image = args.imagename
    # Variable for used when dispatching multiple instances at the same time
    asynchronous_dispatching = (args.max_concurrent_instances > 1) and (args.instance_type != 'local')

    with open(args.config_file, 'r') as fd:
        config = json.load(fd)
    hp_grid = list(ParameterGrid(config))
    print("Quick summary")
    print("Experiment name: {}".format(args.experiment_name))
    print("Number of tasks: {}".format(len(hp_grid)))
    print("Instance type: {}".format(args.instance_type))

    running_jobs = []
    # hp_grid = [hp_grid[1]]
    # print('Did you change the hp_id')
    # exit()
    for i, hp in enumerate(hp_grid):    # for i, hp in hp_grid:
        job_name = args.experiment_name
        print(hp)
        # the input_path is a folder thus only its content will be copied.
        # The destination is /opt/ml/input/data/training/, where training is called a channel name and
        # usually one can have different channel for training testing and validation.
        # I don't have it here because I my code I handle the split of the training myself
        input_path = args.input_path

        model = sage.estimator.Estimator(image,
                                         role=role_arn, train_instance_count=1,
                                         train_instance_type=args.instance_type,
                                         train_volume_size=args.volume_size,
                                         train_max_run=args.max_runtime,
                                         base_job_name=job_name,
                                         hyperparameters=hp,
                                         output_path=args.output_path,
                                         sagemaker_session=session)
        # If we want more than one sagemaker instances running at the same time, we need to launch asynchronously.
        # Thus, we do not stay attached to the instance during the training.
        model.fit(input_path, wait=not(asynchronous_dispatching))

        if asynchronous_dispatching:
            training_job_name = model.latest_training_job.name
            print("[Started] Job: {}".format(training_job_name))
            running_jobs.append(training_job_name)
            monitor_jobs(running_jobs, session)

            while len(running_jobs) >= args.max_concurrent_instances:
                monitor_jobs(running_jobs, session)
                time.sleep(30)