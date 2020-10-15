import os

import yaml


class Config:
    _config = dict(
        aws_default_region='eu-west-1',
        am_bucket='am-segm',
        model_path_suffix='ecs/model.pt',
        input_bucket='am-segm-input',
        output_bucket='am-segm-output',
        queue_base_url='https://sqs.eu-west-1.amazonaws.com/236062312728',
        queue_base_name='am-segm-inference',
        container_name='am-segm-batch',
        ecs=dict(
            cluster='am-segm',
            taskDefinition='am-segm-batch',
            launchType='FARGATE',
            networkConfiguration=dict(
                awsvpcConfiguration=dict(
                    subnets=['subnet-2619c87f'],  # eu-west-1a availability zone in SM VPC
                    securityGroups=['sg-73462d16'],  # default in SM VPC
                    assignPublicIp='ENABLED',
                )
            ),
        ),
        overrides=dict()
    )
    _user_config_keys = ['aws_access_key_id', 'aws_secret_access_key', 'user']

    def __init__(self, path):
        user_config = yaml.full_load(open(path))
        assert all(user_config.get(key, None) for key in self._user_config_keys)

        self._config.update(user_config)

        queue_name = f'{self._config["queue_base_name"]}-{self._config["user"]}'
        self._dynamic_items = dict(
            model_path=f'{self._config["am_bucket"]}/{self._config["user"]}/'
                       f'{self._config["model_path_suffix"]}',
            queue_name=queue_name,
            queue_url=f'{self._config["queue_base_url"]}/{queue_name}',
        )

        os.environ['AWS_ACCESS_KEY_ID'] = self._config['aws_access_key_id']
        os.environ['AWS_SECRET_ACCESS_KEY'] = self._config['aws_secret_access_key']
        os.environ['AWS_DEFAULT_REGION'] = self._config['aws_default_region']

    def __getitem__(self, key):
        return self._dynamic_items.get(key, None) or self._config[key]

    def task_config(self):
        task_config = self._config['ecs']
        task_config['overrides'] = {
            'containerOverrides': [
                {
                    'name': self['container_name'],
                    'environment': [
                        {'name': 'MODEL_PATH', 'value': self['model_path']},
                        {'name': 'AWS_ACCESS_KEY_ID', 'value': self['aws_access_key_id']},
                        {'name': 'AWS_SECRET_ACCESS_KEY', 'value': self['aws_secret_access_key']},
                        {'name': 'AWS_DEFAULT_REGION', 'value': self['aws_default_region']},
                        {'name': 'QUEUE_URL', 'value': self['queue_url']},
                    ]
                }
            ]
        }
        return task_config
