import argparse
import boto3
import os
import sagemaker as sage
from sagemaker.session import Session
from .expts_utils import get_config_params

def get_hps(dataset, algo):
    return get_config_params(dataset)[algo]

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--imagename', type=str,
                        help='The name of the image that will be used to execute your code')
    parser.add_argument('-a', '--algos', default=['metakrr_mk'],
                        type=str, nargs='+',
                        help="""The name of the algos: 
                                metakrr_sk|metakrr_mk|metagp_sk|metagp_mk|deep_prior|snail|mann|maml""")
    parser.add_argument('-d', '--datasets', default=['uci'],
                        type=str, nargs='+',
                        help='The name of the dataset: easytoy|toy|mhc|uci|bindingdb|movielens')
    parser.add_argument('-r', '--role', type=str, default='AmazonSageMaker',
                        help="""An AWS IAM role (either name or full ARN). The Amazon SageMaker training jobs and APIs
                                 that create Amazon SageMaker endpoints use this role to access training data and model 
                                 artifacts. After the endpoint is created, the inference code might use the IAM role,
                                 if it needs to access an AWS resource.""")
    parser.add_argument('-n', '--instance_count', type=int, default=1,
                        help="""Number of Amazon EC2 instances to use for training. (default: 1)""")
    parser.add_argument('-t', '--instance_type', type=str, default='local',     # gpu_local
                        help="""type of EC2 instance to use for training, for example 'ml.c4.xlarge', 
                                (default: local).""")
    parser.add_argument('-v', '--volume_size', type=int, default=30,
                        help="""Size in GB of the EBS volume to use for storing input data
                                during training (default: 30). Must be large enough to store training data 
                                if File Mode is used (which is the default)""")
    parser.add_argument('-x', '--max_runtime', type=int, default=60*60,
                        help="""Timeout in seconds for training (default: 60 * 60).""")
    parser.add_argument('-o', '--output_path', type=str, default='',
                        help="""S3 location for saving the training result (model artifacts and output files).
                                If not specified, results are stored to a default bucket. If the bucket with the 
                                specific name does not exist, the estimator creates the bucket during the
                                sagemaker.estimator.EstimatorBase.fit method execution.""")
    parser.add_argument('-j', '--job_name', type=str, default=None,
                        help="""Prefix for training job name when the sagemaker.estimator.EstimatorBase.fit
            method launches. If not specified, the estimator generates a default job name, based on
            the training image name and current timestamp.""")
    parser.add_argument('-p', '--hyperparameters', type=str, default='',
                        help="""Dictionary containing the hyperparameters to initialize this estimator with.""")

    args = parser.parse_args()
    # set role
    roles = boto3.client('iam').list_roles()['Roles']
    if roles == []:
        print('No role available.')
        exit()
    for r in roles:
        if args.role in r['Arn']:
            role = r['Arn']

    # setup hyperparameter grid using args.datasets and args.algos
    datasets = args.datasets
    algos = args.algos
    hp_grid = []
    for dataset in datasets:
        for algo in algos:
            hp_grid.extend(get_hps(dataset, algo))

    print(hp_grid)
    exit()
    for hp in hp_grid:
        dataset_name = hp['dataset_name']
        if 'local' in args.instance_type:
            session = None
            image = args.imagename
            input_path = "file:///home/prtos/workspace/code/few_shot_regression/datasets"
            output_path = "file:///home/prtos/workspace/code/few_shot_regression/results"
        else:
            session = Session()
            account = session.boto_session.client('sts').get_caller_identity()['Account']
            region = session.boto_session.region_name
            image = '{}.dkr.ecr.{}.amazonaws.com/{}:latest'.format(account, region, args.imagename)
            input_path = "s3://prudencio/metalearning/{}".format(dataset_name)
            output_path = "s3://prudencio/metalearning/results/output"

        model = sage.estimator.Estimator(image,
                                         role=role, train_instance_count=args.instance_count,
                                         train_instance_type=args.instance_type,
                                         train_volume_size=args.volume_size,
                                         train_max_run=args.max_runtime,
                                         base_job_name=args.job_name,
                                         hyperparameters=args.hyperparameters,
                                         output_path=output_path,
                                         sagemaker_session=session)
        print(args)
        model.fit(input_path)