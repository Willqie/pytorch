from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension

setup(
    name='BMMExt',
    ext_modules=[
        CUDAExtension(
            'BMMExt', sources=['BMMExt.cpp'],
            extra_compile_args = {'cxx': ['-lcubls', '-g']}
            )
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
