#!/usr/bin/env python

import glob
import os

import torch
from setuptools import find_packages
from setuptools import setup
from torch.utils.cpp_extension import CUDA_HOME
from torch.utils.cpp_extension import CppExtension
from torch.utils.cpp_extension import CUDAExtension


def get_extensions():
    this_dir = os.path.dirname(os.path.abspath(__file__))
    # common csrc
    extensions_dir = os.path.join(this_dir, "roadtensor/common", "csrc")

    main_file = glob.glob(os.path.join(extensions_dir, "*.cpp"))
    source_cpu = glob.glob(os.path.join(extensions_dir, "cpu", "*.cpp"))
    source_cpu += glob.glob(os.path.join(extensions_dir, "plugins", "*.cpp"))
    source_cuda = glob.glob(os.path.join(extensions_dir, "cuda", "*.cu"))

    sources = main_file + source_cpu
    extension = CppExtension

    extra_compile_args = {"cxx": []}
    define_macros = []

    if torch.cuda.is_available() and CUDA_HOME is not None:
        extension = CUDAExtension
        sources += source_cuda
        define_macros += [("WITH_CUDA", None)]
        extra_compile_args["nvcc"] = [
            '--expt-extended-lambda', 
            '--use_fast_math',
            '-Xcompiler', '-Wall',  # set up for g++
            '-arch=sm_70',  # sm_70 for TITAN V , sm_61 for GeForce GTX 1080 Ti
        ]

    include_dirs = [extensions_dir]

    ext_modules = [
        extension(
            "roadtensor._C",
            sources,
            include_dirs=include_dirs,
            define_macros=define_macros,
            extra_compile_args=extra_compile_args,
            libraries=['nvinfer', 'nvinfer_plugin', 'nvonnxparser']  #TensorRT support
        )
    ]

    return ext_modules

setup(
    name="roadtensor",
    version="0.1",
    author="fabu",
    url="http://git.fabu.ai/roadtensor/roadtensor",
    description="auto perception in pytorch",
    packages=find_packages(exclude=("tests",)),
    ext_modules=get_extensions(), # if no need c++ source op, you can comment this line
    cmdclass={"build_ext": torch.utils.cpp_extension.BuildExtension},
)
