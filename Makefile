.PHONY: help clean check autoformat
.DEFAULT: help

# Generates a useful overview/help message for various make features - add to this as necessary!
help:
	@echo "make clean"
	@echo "    Remove all temporary pyc/pycache files"
	@echo "make check"
	@echo "    Run code style and linting (black, ruff) *without* changing files!"
	@echo "make autoformat"
	@echo "    Run code styling (black, ruff) and update in place - committing with pre-commit also does this."

clean:
	find . -name "*.pyc" | xargs rm -f && \
	find . -name "__pycache__" | xargs rm -rf

check:
	black --check .
	ruff check --show-source .

autoformat:
	black .
	ruff check --fix --show-fixes .

builddocker:
	DOCKER_BUILDKIT=0 docker build -f Dockerfile -t openvla:latest .
	
launchdocker:
	docker run -it --rm -v /datasets/openx_embodiment:/datasets/openx_embodiment -e HF_TOKEN -e WANDB_API_KEY -e XAUTHORITY -e DISPLAY=$(DISPLAY) -e NVIDIA_DRIVER_CAPABILITIES=video,compute,utility --shm-size 32G -v /tmp/.X11-unix:/tmp/.X11-unix -v $(PWD):/opt/ml/code/  -w /opt/ml/code/ --gpus all openvla:latest bash
