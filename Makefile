.PHONY: sandbox all

all: sandbox

sandbox:
	docker build . -f sandbox.Dockerfile --tag mapta-sandbox:latest

run:
	uv run --env-file=.env main.py
