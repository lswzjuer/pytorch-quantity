We define two Docker images: build and release.
The build image provides an environment where
`bash roadstar.sh build` runs successfully,
the release image provides an environment where
`bash scripts/<module_name>.sh start` runs successfully.

The standard workflow for getting these containers
running is (from the main roadstar directory):
```bash
bash docker/scripts/{dev/release}_start.sh
bash docker/scripts/{dev/release}_into.sh
```
The development docker will only map several ports from host to
docker and this mapping will be changed when restart the docker. And you can
check this mapping by:
```bash
bash docker/scripts/check_port.sh
```
Considering the size of docker, we also provide the save and load scripts to
save the docker into a `.img` file and load from it. The default save or load path
is `/tmp/roadstar_docker.img` and you can also specify the docker image path when run
the scripts.
```bash
bash docker/scripts/dump_docker.sh
bash docker/scripts/load_docker.sh
```
Advanced users wishing to create their own build/release
images can do this by running:
```bash
bash docker/scripts/{dev/release}_create.sh
```
Note that, within the scripts in this directory,
only standard tools that are expected
to be available in most Linux distributions
should be used (e.g., don't use realpath).
