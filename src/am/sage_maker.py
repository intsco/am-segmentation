import shutil
import tarfile
from pathlib import Path


def download_training_artifacts(sagemaker_session, prefix, local_dir, matrix):
    bucket = sagemaker_session.default_bucket()
    print(f'Downloading training artifacts from s3://{bucket}/{prefix} to {local_dir}')

    sagemaker_session.download_data(local_dir, bucket, prefix)

    for archive_path in local_dir.iterdir():
        if archive_path.name.endswith('tar.gz'):
            with tarfile.open(archive_path, 'r:gz') as f:
                f.extractall(local_dir)
            archive_path.unlink()

    shutil.copy(local_dir / 'model.pt', local_dir / f'model-{matrix.lower()}.pt')


def copy_training_data(s3, from_bucket, from_prefix, to_bucket, to_prefix):
    print(
        f'Copying training data from s3://{from_bucket}/{from_prefix} '
        f'to s3://{to_bucket}/{to_prefix}'
    )
    for doc in s3.list_objects_v2(Bucket=from_bucket, Prefix=from_prefix)['Contents']:
        copy_source = {'Bucket': from_bucket, 'Key': doc['Key']}
        relative_key = doc['Key'].replace(from_prefix + '/', '')
        new_key = f'{to_prefix}/{relative_key}'
        s3.copy(copy_source, to_bucket, new_key)


def upload_fine_tuning_data(sagemaker_session, local_path, prefix):
    print(
        f'Uploading fine tuning data from {local_path} '
        f'to s3://{sagemaker_session.default_bucket()}/{prefix}'
    )
    for path in local_path.iterdir():
        ds_type = path.name
        sagemaker_session.upload_data(path=str(path), key_prefix=f'{prefix}/{ds_type}')


def delete_data(s3, bucket, prefix):
    print(f'Deleting data at s3://{bucket}/{prefix}')
    for doc in s3.list_objects_v2(Bucket=bucket, Prefix=prefix)['Contents']:
        s3.delete_object(Bucket=bucket, Key=doc['Key'])
