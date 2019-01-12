#!/usr/bin/env bash

/etc/init.d/redis-server start

exec python segm_worker.py --model-path unet.pth --data-path data/tasks &>logs/worker.log &

exec gunicorn --reload --bind 0.0.0.0:8000 'api.app:get_app()' &> logs/api.log
