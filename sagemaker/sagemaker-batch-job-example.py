from time import time
from pathlib import Path
import boto3
from sagemaker.session import Session
from sagemaker.pytorch import PyTorchModel

# Batch Transform

bucket = 'am-segm'
dataset = 'test_AM_image'
input_prefix = Path(f'input-data/{dataset}')
output_prefix = f'output-data/{dataset}'

s3 = boto3.client('s3')

local_input = Path('data/test_AM_image/source_tiles')

for group_path in local_input.iterdir():
    for file_path in (group_path / 'source').iterdir():
        s3_file_path = input_prefix / group_path.name / file_path.name
        print(f'Uploading {file_path} to {s3_file_path}')
        s3.upload_file(str(file_path), bucket, str(s3_file_path))


session = Session()
s3_input = f's3://{bucket}/{input_prefix}'
s3_output = f's3://{bucket}/{output_prefix}'
pytorch_model = PyTorchModel(model_data='s3://am-segm/unet.tar.gz',
                             image='236062312728.dkr.ecr.eu-west-1.amazonaws.com/intsco/am-segm',
                             role='AM-SegmSageMakerRole',
                             entry_point='sagemaker/main.py',
                             sagemaker_session=session)
transformer = pytorch_model.transformer(instance_count=3,
                                        instance_type='ml.c4.xlarge',
                                        # instance_type='ml.p2.xlarge',
                                        output_path=s3_output,
                                        accept='application/x-image',
                                        strategy='SingleRecord',
                                        env={'MODEL_SERVER_TIMEOUT': '180'})
start = time()
transformer.transform(data=s3_input,
                      data_type='S3Prefix',
                      content_type='application/x-image')
transformer.wait()
print('{} min {} sec'.format(*divmod(int(time() - start), 60)))
