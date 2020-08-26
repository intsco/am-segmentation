import argparse
from pathlib import Path

import yaml
import boto3
import sagemaker
from sagemaker.estimator import Estimator

from am.sage_maker import copy_training_data, upload_fine_tuning_data, delete_data, \
    download_training_artifacts


def fit_estimator():
    pytorch_estimator = Estimator(
        hyperparameters=hyperparameters,
        image_uri=image_uri,
        instance_type='ml.p2.xlarge',
        # instance_type='local',
        instance_count=1,
        # use_spot_instances=False,
        # max_wait=None,
        role=role,
        sagemaker_session=sagemaker_session,
        # sagemaker_session=sagemaker.LocalSession(boto_session=session),
        base_job_name='sagemaker-pytorch-train-gpu'
    )
    ds_types = ['train', 'valid']
    pytorch_estimator.fit(
        {
            ds_type: f's3://{sagemaker_bucket}/{sagemaker_input_prefix}/{ds_type}'
            for ds_type in ds_types
        }
    )
    return pytorch_estimator


def parse_args():
    parser = argparse.ArgumentParser(description='Train AM segmentation model with SageMaker')
    parser.add_argument(
        '--fine-tuning-path',
        type=str,
        default='./data/fine-tuning',
        help="Path to fine tuning data"
    )
    parser.add_argument('--matrix', type=str, required=True, help="'DHB' or 'DAN'")
    return parser.parse_args()


def init_aws_clients():
    config = yaml.full_load(open('config/config.yml'))
    session = boto3.Session(
        aws_access_key_id=config['aws']['aws_access_key_id'],
        aws_secret_access_key=config['aws']['aws_secret_access_key'],
        region_name=config['aws']['aws_default_region']
    )
    sagemaker_session = sagemaker.Session(boto_session=session)
    s3 = session.client('s3')
    return sagemaker_session, s3


if __name__ == '__main__':
    args = parse_args()

    sagemaker_session, s3 = init_aws_clients()

    role = 'arn:aws:iam::236062312728:role/AM-SegmSageMakerRole'
    image_uri = '236062312728.dkr.ecr.eu-west-1.amazonaws.com/am-segm/sagemaker-pytorch-train-gpu:latest'
    am_bucket = 'am-segm'
    training_data_prefix = 'training-data'
    sagemaker_bucket = sagemaker_session.default_bucket()
    sagemaker_input_prefix = 'input'
    fine_tuning_data_path = Path(args.fine_tuning_path)
    output_artifacts_path = Path('./model')

    copy_training_data(
        s3, am_bucket, training_data_prefix, sagemaker_bucket, sagemaker_input_prefix
    )
    upload_fine_tuning_data(sagemaker_session, fine_tuning_data_path, sagemaker_input_prefix)

    hyperparameters = {
        'epochs': 10, 'batch-size': 4, 'num-workers': 4, 'lr-dec-1': 3e-2, 'lr-enc-2': 3e-4
    }
    pytorch_estimator = fit_estimator()

    estimator_output_prefix = f'{pytorch_estimator.latest_training_job.name}/output'
    download_training_artifacts(
        sagemaker_session, estimator_output_prefix, output_artifacts_path, args.matrix
    )

    delete_data(s3, sagemaker_bucket, sagemaker_input_prefix)
