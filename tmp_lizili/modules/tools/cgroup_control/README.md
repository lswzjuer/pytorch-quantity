## CPU Resource Mangement

We use `cpuset` subsystem to assign individual CPUs to cgroups.([details](https://access.redhat.com/documentation/en-us/red_hat_enterprise_linux/6/html/resource_management_guide/sec-cpuset)) This tool allow us assign individual CPUs to individual modules via a config file.

### Usage 

```bash
sudo bash cgroup.sh [config_file]
```
**Important**:
 * Root permission is needed
 * If you run this script in a container, you need add `--privileged` when you start the container. ([details](https://stackoverflow.com/questions/32534203/mounting-cgroups-inside-a-docker-container))

In config file, use `:` to seprate module name and cpu id. Here is an example.

```config
# module_name : cpu_id(starts from 0)
planning    : 0
planning_v2 : 1-3
drivers_v2  : 0-2,7 
```

