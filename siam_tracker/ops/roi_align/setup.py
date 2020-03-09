# Copyright 2018-2019 Open-MMLab. All rights reserved.
# Licensed under the Apache License, Version 2.0.

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='roi_align_backend',
    ext_modules=[
        CUDAExtension('roi_align_backend', [
            'src/roi_align.cpp',
            'src/roi_align_cpu.cpp',
            'src/roi_align_kernel.cu',
        ]),
    ],
    cmdclass={'build_ext': BuildExtension})
