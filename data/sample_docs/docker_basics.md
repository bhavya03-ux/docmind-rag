# Docker Basics

Docker packages an application together with its dependencies into a container, so it runs the same way on a laptop, a test server and in production.

## Images and containers
An image is a read-only template built from instructions in a Dockerfile. A container is a running instance of an image. You can start many containers from one image.

## Dockerfile essentials
A Dockerfile starts with a FROM line that selects a base image, such as python:3.11-slim. The COPY instruction adds files, RUN executes commands at build time, and CMD sets the default command when a container starts. Copying the requirements file and installing packages before copying the rest of the code lets Docker reuse cached layers and speeds up rebuilds.

## Volumes and ports
Data written inside a container disappears when the container is removed. A volume stores data outside the container's writable layer so it survives restarts. To reach a service from the host, publish a port with the -p flag, for example -p 8501:8501.

## Docker Compose
Docker Compose describes several services in one docker-compose.yml file. Running docker compose up starts them together on a shared network where each service can reach the others by its service name.

## Useful commands
docker build creates an image, docker run starts a container, docker ps lists running containers, and docker logs shows a container's output.
