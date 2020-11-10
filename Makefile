VERSION=v1.6.1
BASE_ENV_VERSION=v1.7.6
PROJECT_ID=neuro-project-60319926

##### CONSTANTS #####

#DATA_DIR=data
#CONFIG_DIR=config
#CODE_DIR=src
#NOTEBOOKS_DIR=notebooks
#RESULTS_DIR=results

#PROJECT_PATH_STORAGE=storage:am_segm
#PROJECT_PATH_ENV=/project

#PROJECT=am-segm

SETUP_JOB=setup-$(PROJECT)
#TRAIN_JOB=train-$(PROJECT)
DEVELOP_JOB=develop-$(PROJECT)
JUPYTER_JOB=jupyter-$(PROJECT)
TENSORBOARD_JOB=tensorboard-$(PROJECT)
FILEBROWSER_JOB=filebrowser-$(PROJECT)
_PROJECT_TAGS=--tag "kind:project" \
              --tag "project:$(PROJECT)" \
              --tag "project-id:$(PROJECT_ID)"

BASE_ENV=neuromation/base:$(BASE_ENV_VERSION)
#CUSTOM_ENV?=image:neuromation-$(PROJECT):$(VERSION)

##### VARIABLES #####

# To overload a variable use either way:
# - change its default value in Makefile,
# - export variable: `export VAR=value`,
# - or change its value for a single run only: `make <target> VAR=value`.

# Allows to set the `neuro` executable:
#   make setup NEURO=/usr/bin/neuro
#   make setup NEURO="neuro --verbose --show-traceback"
NEURO?=neuro

# Location of your dataset on the platform storage:
#   make setup DATA_DIR_STORAGE=storage:datasets/cifar10
DATA_DIR_STORAGE?=$(PROJECT_PATH_STORAGE)/$(DATA_DIR)

# The type of the training machine (run `neuro config show`
# to see the list of available types):
#   make jupyter PRESET=cpu-small
#PRESET?=gpu-small

# Extra options for `neuro run` targets:
#   make train RUN_EXTRA="--env MYVAR=value"
RUN_EXTRA?=

# HTTP authentication (via cookies) for the job's HTTP link.
# Applied only to jupyter, tensorboard and filebrowser jobs.
# Set `HTTP_AUTH=--no-http-auth` to disable any authentication.
# WARNING: removing authentication might disclose your sensitive data stored in the job.
#   make jupyter HTTP_AUTH=--no-http-auth
HTTP_AUTH?=--http-auth

# Command to run training inside the environment:
#   make train TRAIN_CMD="python ./train.py"
#TRAIN_CMD?=python -u $(CODE_DIR)/train.py --data $(DATA_DIR)

# Postfix of training jobs:
#   make train RUN=experiment-2
#   make kill RUN=experiment-2
RUN?=base

# Local port to use in `port-forward`:
#   make port-forward-develop LOCAL_PORT=2233
LOCAL_PORT?=2211

# Jupyter mode. Available options: `notebook` (to
# run Jupyter Notebook), `lab` (to run JupyterLab):
#   make jupyter JUPYTER_MODE=LAB
JUPYTER_MODE?=notebook
# Maximum running time for Jupyter. Available opitons:
# `1d`, `1h`, `8h`, `0` (to disable):
#   make jupyter JUPYTER_LIFE_SPAN=4h
JUPYTER_LIFE_SPAN?=0

# Storage synchronization:
#  make jupyter SYNC=""
SYNC?=upload-code upload-config

SECRETS?=



##### AM-SEGM CUSTOM #####

PROJECT=am-segm

DATA_DIR=data
#CONFIG_DIR=config
CODE_DIR=src
NOTEBOOKS_DIR=notebooks
RESULTS_DIR=results

PROJECT_PATH_STORAGE=storage:$(PROJECT)
# project path within Docker container
PROJECT_PATH_ENV=/$(PROJECT)

TRAIN_JOB=train-$(PROJECT)
PREDICT_JOB=predict-$(PROJECT)

CUSTOM_ENV=image:am-segm/pytorch:latest
PRESET=gpu-k80-small

TRAIN_DATA_DIR=  # must be provided
PREDICT_DATA_DIR=  # must be provided


.PHONY: am-setup
am-setup: ### Setup remote environment
	$(NEURO) mkdir --parents $(PROJECT_PATH_STORAGE)

.PHONY: build-push
build-push: ### Build image locally and push it to registry
	docker build -t am-segm/neuro-pytorch neuro
	$(NEURO) push am-segm/neuro-pytorch $(CUSTOM_ENV)

.PHONY: upload-code
upload-code: _check_setup  ### Upload code directory to the platform storage
	$(NEURO) cp \
		--recursive \
		--update \
		--no-target-directory \
		$(CODE_DIR) $(PROJECT_PATH_STORAGE)/$(CODE_DIR)

.PHONY: upload-train-data
upload-train-data: _check_setup  ### Upload training data directory to the platform storage
	$(NEURO) cp \
		--recursive \
		--update \
		--no-target-directory \
		$(TRAIN_DATA_DIR) $(PROJECT_PATH_STORAGE)/$(TRAIN_DATA_DIR)

.PHONY: train
train: _check_setup $(SYNC)   ### Run a training job (set up env var 'RUN' to specify the training job),
	$(NEURO) run $(RUN_EXTRA) \
		$(SECRETS) \
		--name $(TRAIN_JOB)-$(RUN) \
		--tag "target:train" $(_PROJECT_TAGS) \
		--preset gpu-k80-small \
		--wait-start \
		--volume $(PROJECT_PATH_STORAGE)/$(DATA_DIR):$(PROJECT_PATH_ENV)/$(DATA_DIR):ro \
		--volume $(PROJECT_PATH_STORAGE)/$(CODE_DIR):$(PROJECT_PATH_ENV)/$(CODE_DIR):ro \
		--volume $(PROJECT_PATH_STORAGE)/$(CONFIG_DIR):$(PROJECT_PATH_ENV)/$(CONFIG_DIR):ro \
		--volume $(PROJECT_PATH_STORAGE)/$(RESULTS_DIR):$(PROJECT_PATH_ENV)/$(RESULTS_DIR):rw \
		--env PYTHONPATH=$(PROJECT_PATH_ENV)/$(CODE_DIR) \
		--env EXPOSE_SSH=yes \
		--life-span=1d \
		$(CUSTOM_ENV) \
		bash -c 'cd $(PROJECT_PATH_ENV) && python -u src/train.py --data-dir ${TRAIN_DATA_DIR}'

.PHONY: download-train-results
download-train-results: _check_setup  ### Download results directory from the platform storage
	$(NEURO) cp \
		--recursive \
		--update \
		--no-target-directory \
		$(PROJECT_PATH_STORAGE)/$(RESULTS_DIR) $(RESULTS_DIR)

.PHONY: upload-predict-data
upload-predict-data: _check_setup  ### Upload training data directory to the platform storage
	$(NEURO) mkdir --parents $(PROJECT_PATH_STORAGE)/$(PREDICT_DATA_DIR)
	$(NEURO) cp \
		--recursive \
		--update \
		--no-target-directory \
		$(PREDICT_DATA_DIR) $(PROJECT_PATH_STORAGE)/$(PREDICT_DATA_DIR)

.PHONY: predict
predict: _check_setup $(SYNC)   ### Run an inference job
	$(NEURO) run $(RUN_EXTRA) \
		$(SECRETS) \
		--name $(PREDICT_JOB)-$(RUN) \
		--tag "target:train" $(_PROJECT_TAGS) \
		--preset cpu-large \
		--wait-start \
		--volume $(PROJECT_PATH_STORAGE)/$(DATA_DIR):$(PROJECT_PATH_ENV)/$(DATA_DIR):ro \
		--volume $(PROJECT_PATH_STORAGE)/$(CODE_DIR):$(PROJECT_PATH_ENV)/$(CODE_DIR):ro \
		--volume $(PROJECT_PATH_STORAGE)/$(CONFIG_DIR):$(PROJECT_PATH_ENV)/$(CONFIG_DIR):ro \
		--volume $(PROJECT_PATH_STORAGE)/$(RESULTS_DIR):$(PROJECT_PATH_ENV)/$(RESULTS_DIR):rw \
		--env PYTHONPATH=$(PROJECT_PATH_ENV)/$(CODE_DIR) \
		--env EXPOSE_SSH=yes \
		--life-span=1d \
		$(CUSTOM_ENV) \
		bash -c 'cd $(PROJECT_PATH_ENV) && python -u src/predict.py --data-dir ${PREDICT_DATA_DIR}'

.PHONY: download-predict-results
download-predict-results: _check_setup  ### Download results directory from the platform storage
	$(NEURO) cp \
		--recursive \
		--update \
		$(PROJECT_PATH_STORAGE)/$(RESULTS_DIR)/predictions/* $(PREDICT_DATA_DIR)


##### HELP #####

.PHONY: help
help:
	@# generate help message by parsing current Makefile
	@# idea: https://marmelab.com/blog/2016/02/29/auto-documented-makefile.html
	@grep -hE '^[a-zA-Z_-]+:[^#]*?### .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'

##### SETUP #####

.PHONY: setup
setup: ### Setup remote environment
	$(NEURO) mkdir --parents $(PROJECT_PATH_STORAGE) \
		$(PROJECT_PATH_STORAGE)/$(CODE_DIR) \
		$(DATA_DIR_STORAGE) \
		$(PROJECT_PATH_STORAGE)/$(CONFIG_DIR) \
		$(PROJECT_PATH_STORAGE)/$(NOTEBOOKS_DIR) \
		$(PROJECT_PATH_STORAGE)/$(RESULTS_DIR)
	$(NEURO) cp requirements.txt $(PROJECT_PATH_STORAGE)
	$(NEURO) cp apt.txt $(PROJECT_PATH_STORAGE)
	$(NEURO) cp setup.cfg $(PROJECT_PATH_STORAGE)
	$(NEURO) run $(RUN_EXTRA) \
		$(SECRETS) \
		--name $(SETUP_JOB) \
		--tag "target:setup" $(_PROJECT_TAGS) \
		--preset cpu-small \
		--detach \
		--life-span=1h \
		--volume $(PROJECT_PATH_STORAGE):$(PROJECT_PATH_ENV):ro \
		$(BASE_ENV) \
		'sleep infinity'
	$(NEURO) exec --no-key-check -T $(SETUP_JOB) "bash -c 'export DEBIAN_FRONTEND=noninteractive && apt-get -qq update && cat $(PROJECT_PATH_ENV)/apt.txt | tr -d \"\\r\" | xargs -I % apt-get -qq install --no-install-recommends % && apt-get -qq clean && apt-get autoremove && rm -rf /var/lib/apt/lists/*'"
	$(NEURO) exec --no-key-check -T $(SETUP_JOB) "bash -c 'pip install --progress-bar=off -U --no-cache-dir -r $(PROJECT_PATH_ENV)/requirements.txt'"
	$(NEURO) exec --no-key-check -T $(SETUP_JOB) "bash -c 'ssh-keygen -f /id_rsa -t rsa -N neuromation -q'"
	$(NEURO) --network-timeout 300 job save $(SETUP_JOB) $(CUSTOM_ENV)
	$(NEURO) kill $(SETUP_JOB) || :
	@touch .setup_done

.PHONY: kill-setup
kill-setup:  ### Terminate the setup job (if it was not killed by `make setup` itself)
	$(NEURO) kill $(SETUP_JOB) || :

.PHONY: _check_setup
_check_setup:
	@test -f .setup_done || { echo "Please run 'make setup' first"; false; }

##### STORAGE #####

#.PHONY: upload-code
#upload-code: _check_setup  ### Upload code directory to the platform storage
#	$(NEURO) cp \
#		--recursive \
#		--update \
#		--no-target-directory \
#		$(CODE_DIR) $(PROJECT_PATH_STORAGE)/$(CODE_DIR)

.PHONY: download-code
download-code: _check_setup  ### Download code directory from the platform storage
	$(NEURO) cp \
		--recursive \
		--update \
		--no-target-directory \
		$(PROJECT_PATH_STORAGE)/$(CODE_DIR) $(CODE_DIR)

.PHONY: clean-code
clean-code: _check_setup  ### Delete code directory from the platform storage
	$(NEURO) rm --recursive $(PROJECT_PATH_STORAGE)/$(CODE_DIR)/*

#.PHONY: upload-data
#upload-data: _check_setup  ### Upload data directory to the platform storage
#	$(NEURO) cp \
#		--recursive \
#		--update \
#		--no-target-directory \
#		$(DATA_DIR) $(DATA_DIR_STORAGE)

.PHONY: download-data
download-data: _check_setup  ### Download data directory from the platform storage
	$(NEURO) cp \
		--recursive \
		--update \
		--no-target-directory \
		$(DATA_DIR_STORAGE) $(DATA_DIR)

.PHONY: clean-data
clean-data: _check_setup  ### Delete data directory from the platform storage
	$(NEURO) rm --recursive $(DATA_DIR_STORAGE)/*

.PHONY: upload-config
upload-config: _check_setup  ### Upload config directory to the platform storage
#	$(NEURO) cp \
#		--recursive \
#		--update \
#		--no-target-directory \
#		$(CONFIG_DIR) $(PROJECT_PATH_STORAGE)/$(CONFIG_DIR)

.PHONY: download-config
download-config: _check_setup  ### Download config directory from the platform storage
	$(NEURO) cp \
		--recursive \
		--update \
		--no-target-directory \
		$(PROJECT_PATH_STORAGE)/$(CONFIG_DIR) $(CONFIG_DIR)

.PHONY: clean-config
clean-config: _check_setup  ### Delete config directory from the platform storage
	$(NEURO) rm --recursive $(PROJECT_PATH_STORAGE)/$(CONFIG_DIR)/*

.PHONY: upload-notebooks
upload-notebooks: _check_setup  ### Upload notebooks directory to the platform storage
	$(NEURO) cp \
		--recursive \
		--update \
		--no-target-directory \
		--exclude="*" \
		--include="*.ipynb" \
		$(NOTEBOOKS_DIR) $(PROJECT_PATH_STORAGE)/$(NOTEBOOKS_DIR)

.PHONY: download-notebooks
download-notebooks: _check_setup  ### Download notebooks directory from the platform storage
	$(NEURO) cp \
		--recursive \
		--update \
		--no-target-directory \
		--exclude="*" \
		--include="*.ipynb" \
		$(PROJECT_PATH_STORAGE)/$(NOTEBOOKS_DIR) $(NOTEBOOKS_DIR)

.PHONY: clean-notebooks
clean-notebooks: _check_setup  ### Delete notebooks directory from the platform storage
	$(NEURO) rm --recursive $(PROJECT_PATH_STORAGE)/$(NOTEBOOKS_DIR)/*

.PHONY: upload-results
upload-results: _check_setup  ### Upload results directory to the platform storage
	$(NEURO) cp \
		--recursive \
		--update \
		--no-target-directory \
		$(RESULTS_DIR)/ $(PROJECT_PATH_STORAGE)/$(RESULTS_DIR)

#.PHONY: download-results
#download-results: _check_setup  ### Download results directory from the platform storage
#		$(NEURO) cp \
#		--recursive \
#		--update \
#		--no-target-directory \
#		$(PROJECT_PATH_STORAGE)/$(RESULTS_DIR)/ $(RESULTS_DIR)

.PHONY: clean-results
clean-results: _check_setup  ### Delete results directory from the platform storage
	$(NEURO) rm --recursive $(PROJECT_PATH_STORAGE)/$(RESULTS_DIR)/*

.PHONY: upload-all
upload-all: upload-code upload-data upload-config upload-notebooks upload-results  ### Upload code, data, config, notebooks, and results directories to the platform storage

.PHONY: download-all
download-all: download-code download-data download-config download-notebooks download-results  ### Download code, data, config, notebooks, and results directories from the platform storage

.PHONY: clean-all
clean-all: clean-code clean-data clean-config clean-notebooks clean-results  ### Delete code, data, config, notebooks, and results directories from the platform storage

##### JOBS #####
.PHONY: develop
develop: _check_setup $(SYNC)  ### Run a development job
	$(NEURO) run $(RUN_EXTRA) \
		$(SECRETS) \
		--name $(DEVELOP_JOB) \
		--tag "target:develop" $(_PROJECT_TAGS) \
		--preset $(PRESET) \
		--detach \
		--volume $(DATA_DIR_STORAGE):$(PROJECT_PATH_ENV)/$(DATA_DIR):ro \
		--volume $(PROJECT_PATH_STORAGE)/$(CODE_DIR):$(PROJECT_PATH_ENV)/$(CODE_DIR):rw \
		--volume $(PROJECT_PATH_STORAGE)/$(CONFIG_DIR):$(PROJECT_PATH_ENV)/$(CONFIG_DIR):ro \
		--volume $(PROJECT_PATH_STORAGE)/$(RESULTS_DIR):$(PROJECT_PATH_ENV)/$(RESULTS_DIR):rw \
		--env PYTHONPATH=$(PROJECT_PATH_ENV) \
		--env EXPOSE_SSH=yes \
		--life-span=1d \
		$(CUSTOM_ENV) \
		sleep infinity

.PHONY: connect-develop
connect-develop:  ### Connect to the remote shell running on the development job
	$(NEURO) exec --no-key-check $(DEVELOP_JOB) bash

.PHONY: logs-develop
logs-develop:  ### Connect to the remote shell running on the development job
	$(NEURO) logs $(DEVELOP_JOB)

.PHONY: port-forward-develop
port-forward-develop:  ### Forward SSH port to localhost for remote debugging
	@test $(LOCAL_PORT) || { echo 'Please set up env var LOCAL_PORT'; false; }
	$(NEURO) port-forward $(DEVELOP_JOB) $(LOCAL_PORT):22

.PHONY: kill-develop
kill-develop:  ### Terminate the development job
	$(NEURO) kill $(DEVELOP_JOB) || :

#.PHONY: train
#train: _check_setup $(SYNC)   ### Run a training job (set up env var 'RUN' to specify the training job),
#	$(NEURO) run $(RUN_EXTRA) \
#		$(SECRETS) \
#		--name $(TRAIN_JOB)-$(RUN) \
#		--tag "target:train" $(_PROJECT_TAGS) \
#		--preset $(PRESET) \
#		--wait-start \
#		--volume $(DATA_DIR_STORAGE):$(PROJECT_PATH_ENV)/$(DATA_DIR):ro \
#		--volume $(PROJECT_PATH_STORAGE)/$(CODE_DIR):$(PROJECT_PATH_ENV)/$(CODE_DIR):ro \
#		--volume $(PROJECT_PATH_STORAGE)/$(CONFIG_DIR):$(PROJECT_PATH_ENV)/$(CONFIG_DIR):ro \
#		--volume $(PROJECT_PATH_STORAGE)/$(RESULTS_DIR):$(PROJECT_PATH_ENV)/$(RESULTS_DIR):rw \
#		--env PYTHONPATH=$(PROJECT_PATH_ENV) \
#		--env EXPOSE_SSH=yes \
#		--life-span=0 \
#		$(CUSTOM_ENV) \
#		bash -c 'cd $(PROJECT_PATH_ENV) && $(TRAIN_CMD)'

.PHONY: kill-train
kill-train:  ### Terminate the training job (set up env var 'RUN' to specify the training job)
	$(NEURO) kill $(TRAIN_JOB)-$(RUN) || :

.PHONY: kill-train-all
kill-train-all:  ### Terminate all training jobs you have submitted
	jobs=`neuro -q ps --tag "target:train" $(_PROJECT_TAGS) | tr -d "\r"` && \
	[ ! "$$jobs" ] || $(NEURO) kill $$jobs

.PHONY: connect-train
connect-train: _check_setup  ### Connect to the remote shell running on the training job (set up env var 'RUN' to specify the training job)
	$(NEURO) exec --no-key-check $(TRAIN_JOB)-$(RUN) bash

.PHONY: jupyter
jupyter: _check_setup $(SYNC) ### Run a job with Jupyter Notebook and open UI in the default browser
	$(NEURO) run $(RUN_EXTRA) \
		$(SECRETS) \
		--name $(JUPYTER_JOB) \
		--tag "target:jupyter" $(_PROJECT_TAGS) \
		--preset $(PRESET) \
		--http 8888 \
		$(HTTP_AUTH) \
		--browse \
		--detach \
		--volume $(DATA_DIR_STORAGE):$(PROJECT_PATH_ENV)/$(DATA_DIR):ro \
		--volume $(PROJECT_PATH_STORAGE)/$(CODE_DIR):$(PROJECT_PATH_ENV)/$(CODE_DIR):rw \
		--volume $(PROJECT_PATH_STORAGE)/$(CONFIG_DIR):$(PROJECT_PATH_ENV)/$(CONFIG_DIR):ro \
		--volume $(PROJECT_PATH_STORAGE)/$(NOTEBOOKS_DIR):$(PROJECT_PATH_ENV)/$(NOTEBOOKS_DIR):rw \
		--volume $(PROJECT_PATH_STORAGE)/$(RESULTS_DIR):$(PROJECT_PATH_ENV)/$(RESULTS_DIR):rw \
		--life-span=$(JUPYTER_LIFE_SPAN) \
		--env PYTHONPATH=$(PROJECT_PATH_ENV) \
		$(CUSTOM_ENV) \
		jupyter $(JUPYTER_MODE) \
			--no-browser \
			--ip=0.0.0.0 \
			--allow-root \
			--NotebookApp.token= \
			--notebook-dir=$(PROJECT_PATH_ENV)/$(NOTEBOOKS_DIR)

.PHONY: kill-jupyter
kill-jupyter:  ### Terminate the job with Jupyter Notebook
	$(NEURO) kill $(JUPYTER_JOB) || :

.PHONY: jupyterlab
jupyterlab:  ### Run a job with JupyterLab and open UI in the default browser
	@make --silent jupyter JUPYTER_MODE=lab

.PHONY: kill-jupyterlab
kill-jupyterlab:  ### Terminate the job with JupyterLab
	@make --silent kill-jupyter

.PHONY: tensorboard
tensorboard: _check_setup  ### Run a job with TensorBoard and open UI in the default browser
	$(NEURO) run $(RUN_EXTRA) \
		$(SECRETS) \
		--name $(TENSORBOARD_JOB) \
		--preset cpu-small \
		--tag "target:tensorboard" $(_PROJECT_TAGS) \
		--http 6006 \
		$(HTTP_AUTH) \
		--browse \
		--life-span=1d \
		--volume $(PROJECT_PATH_STORAGE)/$(RESULTS_DIR):$(PROJECT_PATH_ENV)/$(RESULTS_DIR):ro \
		tensorflow/tensorflow:latest \
		tensorboard --host=0.0.0.0 --logdir=$(PROJECT_PATH_ENV)/$(RESULTS_DIR)

.PHONY: kill-tensorboard
kill-tensorboard:  ### Terminate the job with TensorBoard
	$(NEURO) kill $(TENSORBOARD_JOB) || :

.PHONY: filebrowser
filebrowser: _check_setup  ### Run a job with File Browser and open UI in the default browser
	$(NEURO) run $(RUN_EXTRA) \
		$(SECRETS) \
		--name $(FILEBROWSER_JOB) \
		--tag "target:filebrowser" $(_PROJECT_TAGS) \
		--preset cpu-small \
		--http 80 \
		$(HTTP_AUTH) \
		--browse \
		--life-span=1d \
		--volume $(PROJECT_PATH_STORAGE):/srv:rw \
		filebrowser/filebrowser:latest \
		--noauth

.PHONY: kill-filebrowser
kill-filebrowser:  ### Terminate the job with File Browser
	$(NEURO) kill $(FILEBROWSER_JOB) || :

.PHONY: kill-all
kill-all:  ### Terminate all jobs of this project
	jobs=`neuro -q ps $(_PROJECT_TAGS) | tr -d "\r"` && \
	[ ! "$$jobs" ] || $(NEURO) kill $$jobs

##### LOCAL #####

.PHONY: setup-local
setup-local:  ### Install pip requirements locally
	pip install -r requirements.txt

.PHONY: format-local
format-local:  ### Automatically format the code
	isort -rc modules
	black modules

.PHONY: lint-local
lint-local:  ### Run static code analysis locally
	isort -c -rc modules
	black --check modules
	mypy modules
	flake8 modules

##### MISC #####

.PHONY: ps
ps:  ### List all running and pending jobs
	$(NEURO) ps $(_PROJECT_TAGS)

.PHONY: ps-train-all
ps-train-all:  ### List all running and pending training jobs
	$(NEURO) ps --tag "target:train" $(_PROJECT_TAGS)


.PHONY: _upgrade
_upgrade:
	@if ! (git status | grep "nothing to commit"); then echo "Please commit or stash changes before upgrade."; exit 1; fi
	@echo "Applying the latest Neuro Project Template to this project..."
	cookiecutter \
		--output-dir .. \
		--no-input \
		--overwrite-if-exists \
		--checkout release \
		gh:neuromation/cookiecutter-neuro-project \
		project_slug=$(PROJECT) \
		code_directory=$(CODE_DIR)
	git checkout -- $(DATA_DIR) $(CODE_DIR) $(CONFIG_DIR) $(NOTEBOOKS_DIR) $(RESULTS_DIR)
	git checkout -- .gitignore requirements.txt apt.txt setup.cfg README.md
	@echo "Some files are successfully changed. Please review the changes using git diff."