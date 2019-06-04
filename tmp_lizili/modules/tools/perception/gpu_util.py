# coding=utf-8

import subprocess
import re
import sys
import log_util


# Input a cmd string or string list
# Return a stdout
def GetCommandOutput(cmd):
    cmd = cmd.split()
    try:
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    except:
        raise RuntimeError("Invalid Command")
    output = p.communicate()[0]
    # Python2.x and Python3.x compatible
    if sys.version_info.major == 3:
        return str(output, 'utf8')
    else:
        return output


# Call nvidia-smi cmd and return the query result
def QueryGPU():
    return GetCommandOutput("nvidia-smi")


# Caculate remain memory of every GPU
def GetGPURemain(nvidia_output):
    regex = re.compile(r"(\d+)MiB / (\d+)MiB")
    result = regex.findall(nvidia_output)
    gpu_remain = []
    for (used, total) in result:
        gpu_remain.append(int(total) - int(used))
    return gpu_remain


def AssignValidGPU(assign_memory, assign_id, nvidia_output=None):
    if nvidia_output is None:
        nvidia_output = QueryGPU()
    gpu_remain = GetGPURemain(nvidia_output)

    # Checkout Input
    if assign_id != -1 and (assign_id < 0 or assign_id > len(gpu_remain)-1):
        raise RuntimeError("Invalid GPU num")

    # Assign GPU id
    if gpu_remain[assign_id] >= assign_memory:
        return assign_id

    # Scan other GPU
    for i, remain in enumerate(gpu_remain):
        if remain >= assign_memory:
            if assign_id != -1:
                logger = log_util.logger
                logger.info("GPU %d is full, switch to GPU %d" %
                            (assign_id, i))
            return i

    # Otherwise
    raise RuntimeError("No avaliable GPU")
