from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='nms_torch_backend',
    ext_modules=[
        CUDAExtension('nms_torch_backend', [
            'src/nms.cpp',
            'src/nms_cpu.cpp',
            'src/nms_cuda_kernel.cu',
        ]),
    ],
    cmdclass={'build_ext': BuildExtension}
)
