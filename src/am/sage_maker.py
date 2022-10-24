import shutil
import tarfile
from pathlib import Path


def download_training_artifacts(sagemaker_session, prefix, local_dir):
    bucket = sagemaker_session.default_bucket()
    print(f'Downloading training artifacts from s3://{bucket}/{prefix} to {local_dir}')

    sagemaker_session.download_data(local_dir, bucket, prefix)

    for archive_path in Path(local_dir).iterdir():
        if archive_path.name.endswith('tar.gz'):
            with tarfile.open(archive_path, 'r:gz') as f:
                
                import os
                
                def is_within_directory(directory, target):
                    
                    abs_directory = os.path.abspath(directory)
                    abs_target = os.path.abspath(target)
                
                    prefix = os.path.commonprefix([abs_directory, abs_target])
                    
                    return prefix == abs_directory
                
                def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
                
                    for member in tar.getmembers():
                        member_path = os.path.join(path, member.name)
                        if not is_within_directory(path, member_path):
                            raise Exception("Attempted Path Traversal in Tar File")
                
                    tar.extractall(path, members, numeric_owner=numeric_owner) 
                    
                
                safe_extract(f, local_dir)
            archive_path.unlink()


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


def copy_model(s3, from_bucket, from_key, to_bucket, to_key):
    print(
        f'Copying model file from s3://{from_bucket}/{from_key} '
        f'to s3://{to_bucket}/{to_key}'
    )
    copy_source = {'Bucket': from_bucket, 'Key': from_key}
    s3.copy(copy_source, to_bucket, to_key)


def upload_fine_tuning_data(sagemaker_session, local_path, bucket, prefix):
    print(f'Uploading fine tuning data from {local_path} to s3://{bucket}/{prefix}')
    for path in local_path.iterdir():
        ds_type = path.name
        sagemaker_session.upload_data(
            path=str(path), bucket=bucket, key_prefix=f'{prefix}/{ds_type}'
        )


def delete_data(s3, bucket, prefix):
    print(f'Deleting data at s3://{bucket}/{prefix}')
    for doc in s3.list_objects_v2(Bucket=bucket, Prefix=prefix)['Contents']:
        s3.delete_object(Bucket=bucket, Key=doc['Key'])
