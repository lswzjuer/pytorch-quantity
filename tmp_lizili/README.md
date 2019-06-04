# Roadstar

Welcome to the Roadstar.

## Installation

We strongly recommend building Roadstar in our pre-specified Docker environment.
See the following instructions on how to set up the docker environment and build from source.

### The docker environment can be set by the commands below.

Attention!!! The first, check your env variable USER, it should be equal to your user name, if not, please
set it in your .bashrc

```shell
# Check docker if exists
$ which docker
# If the os has NOT installed the docker, install docker first otherwise skip this step.
bash docker/scripts/install_docker.sh
# logout and login to make sure to run docker command without sudo
$ docker ps
# If there is an permission deny error message, add the user into docker group if you have the permission.
$ sudo usermod -aG docker ${USER}
# Start a docker container.
$ bash docker/scripts/dev_start.sh
# Get into the container.
bash docker/scripts/dev_into.sh
```
### To build from source

```Shell
# Build without optimization
$ bash roadstar.sh build
# Build with optimization
$ bash roadstar.sh build_opt
# Build with GPU support
$ bash roadstar.sh build_opt_gpu
```

## Run Roadstar
Follow the steps below to launch Roadstar:
### Start Roadstar

```shell
# Get the latest release version.
$ bash scripts/update_release.sh
# Start the dreamview.
$ cd release && bash scripts/dreamview.sh
# Checkout the dreamview port (For develop container only).
# The release dreamview will be 8888 port.
# Exit the docker container first.
$ ./docker/scripts/check_port.sh
ssh:32773
dreamview:32772
# Now the dreamview is running on the 32773 port (The port will change if you restart the docker).
```
### Access Dreamview
Access dreamview by opening your favorite browser (Chrome recommended).
