from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize

ext_modules = [
    Extension(
        name='region',
        sources=[
            'region.pyx',
            'src/region.c',
        ],
        include_dirs=[
            'src'
        ]
    )
]

setup(
    name='region',
    packages=['region'],
    ext_modules=cythonize(ext_modules)
)
