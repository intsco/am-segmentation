import argparse
from pathlib import Path

import boto3
import sagemaker
from sagemaker.estimator import Estimator

from am.config import Config
from am.sage_maker import copy_training_data, upload_fine_tuning_data, delete_data, \
    download_training_artifacts, copy_model


def parse_args():
    parser = argparse.ArgumentParser(description='Train AM segmentation model with SageMaker')
    parser.add_argument('fine_tuning_path', type=str, help='Path to fine tuning data')
    parser.add_argument('--matrix', type=str, required=True, help="'DHB' or 'DAN'")
    parser.add_argument('--local', action='store_true', help='Run training locally')
    return parser.parse_args()


def init_aws_clients(config):
    session = boto3.Session(
        aws_access_key_id=config['aws_access_key_id'],
        aws_secret_access_key=config['aws_secret_access_key'],
        region_name=config['aws_default_region']
    )
    return sagemaker.Session(boto_session=session), session.client('s3')


if __name__ == '__main__':
    args = parse_args()

    config = Config('config/config.yml')
    sagemaker_session, s3 = init_aws_clients(config)
    sagemaker_bucket = sagemaker_session.default_bucket()

    hyperparameters = config['sagemaker']['hyperparameters']
    if args.local:
        hyperparameters['num_workers'] = 1

    pytorch_estimator = Estimator(
        hyperparameters=hyperparameters,
        image_uri=config['sagemaker']['image_uri'],
        instance_type='ml.p3.2xlarge' if not args.local else 'local',
        instance_count=1,
        # use_spot_instances=False,
        # max_wait=None,
        role=config['sagemaker']['role'],
        sagemaker_session=(sagemaker_session
                           if not args.local
                           else sagemaker.LocalSession(boto_session=sagemaker_session.boto_session)),
        base_job_name='sagemaker-pytorch-train'
    )

    # Prepare input data
    copy_training_data(
        s3,
        from_bucket=config['am_bucket'],
        from_prefix=config.training_data_prefix(args.matrix),
        to_bucket=config['am_bucket'],
        to_prefix=config['sagemaker']['input_prefix'],
    )
    upload_fine_tuning_data(
        sagemaker_session,
        local_path=Path(args.fine_tuning_path),
        bucket=config['am_bucket'],
        prefix=config['sagemaker']['input_prefix'],
    )

    # Fit the model
    ds_types = ['train', 'valid']
    pytorch_estimator.fit(
        {
            ds_type: f's3://{config["am_bucket"]}/{config["sagemaker"]["input_prefix"]}/{ds_type}'
            for ds_type in ds_types
        }
    )

    # Copy model file and download training output
    copy_model(
        s3,
        from_bucket=sagemaker_bucket,
        from_key=f'{pytorch_estimator.latest_training_job.name}/output/model.tar.gz',
        to_bucket=config['am_bucket'],
        to_key=config['model_path'],
    )
    download_training_artifacts(
        sagemaker_session,
        prefix=f'{pytorch_estimator.latest_training_job.name}/output/output.tar.gz',
        local_dir=config['sagemaker']['output_artifacts_path'],
    )
    delete_data(s3, bucket=sagemaker_bucket, prefix=config['sagemaker']['input_prefix'])
