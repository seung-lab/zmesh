#!/usr/bin/env python

import os
import setuptools
import numpy as np
import sys

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

include_dirs = [ np.get_include(), 'zi_lib/', './' ]

setuptools.setup(
  setup_requires=['pbr', 'numpy'],
  python_requires=">=3.7", # >= 3.6 < 4.0
  pbr=True,
  define_macros=[ ("NDEBUG", 1) ],
  ext_modules=[
    setuptools.Extension(
      'zmesh._zmesh',
      sources=[ 'zmesh/_zmesh.cpp' ],
      depends=[ 'cMesher.hpp' ],
      language='c++',
      include_dirs=include_dirs,
      extra_compile_args=extra_compile_args
    ),
  ],
)

