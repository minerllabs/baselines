.PHONY: build rm run bash
CONTAINER_NAME=chainerrl-baselines
CONTAINER_TAG=:v0.0.0
IMAGE_NAME=kumezawa/$(CONTAINER_NAME)$(CONTAINER_TAG)

DOCKER_CMD=tail -f /dev/null

build:
	docker build -t $(IMAGE_NAME) .

run:
	docker run $(DOCKER_OPTS) -d --name $(CONTAINER_NAME) $(IMAGE_NAME) $(DOCKER_CMD)

rm:
	docker rm -f $(CONTAINER_NAME) || echo

bash:
	docker exec -i -t $(CONTAINER_NAME) bash

