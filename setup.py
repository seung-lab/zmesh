#!/usr/bin/env python

import os
import setuptools
import sys

class NumpyImport:
  def __repr__(self):
    import numpy as np

    return np.get_include()

  __fspath__ = __repr__


# NOTE: If zmesh.cpp does not exist, you must run
# cython --cplus -I./zi_lib/ zmesh.pyx

extra_compile_args = []
if sys.platform == 'win32':
  extra_compile_args += [
    '/std:c++17', '/O2'
  ]
else:
  extra_compile_args += [
    '-std=c++17', '-O3', '-Wno-unused-local-typedefs', 
    '-DNDEBUG',
  ]

include_dirs = [ str(NumpyImport()), 'zi_lib/', './' ]

setuptools.setup(
  setup_requires=['pbr', 'numpy', 'cython'],
  python_requires=">=3.8",
  pbr=True,
  define_macros=[ ("NDEBUG", 1) ],
  extras_require={
    "viewer": [ "vtk" ],
  },
  ext_modules=[
    setuptools.Extension(
      'zmesh._zmesh',
      sources=[ 'zmesh/_zmesh.pyx' ],
      depends=[ 'cMesher.hpp' ],
      language='c++',
      include_dirs=include_dirs,
      extra_compile_args=extra_compile_args
    ),
  ],
)

