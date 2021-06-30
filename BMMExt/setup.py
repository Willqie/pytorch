from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension

setup(
    name='BMMExt',
    ext_modules=[
        CUDAExtension(
            'BMMExt', ['BMMExt.cpp'],
            extra_cuda_cflags = ['-lcubls']
            )
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
