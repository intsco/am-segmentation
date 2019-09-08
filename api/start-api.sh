#!/usr/bin/env bash

/etc/init.d/redis-server start

exec gunicorn --reload 'api.app:get_app()'