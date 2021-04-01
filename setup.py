#!/usr/bin/env python

import os
import setuptools
import numpy as np

# NOTE: If zmesh.cpp does not exist, you must run
# cython --cplus -I./zi_lib/ zmesh.pyx

setuptools.setup(
  setup_requires=['pbr', 'numpy'],
  extras_require={
     ':python_version == "2.7"': ['futures'],
  },
  pbr=True,
  ext_modules=[
    setuptools.Extension(
      'zmesh._zmesh',
      sources=[ 'zmesh/_zmesh.cpp' ],
      depends=[ 'cMesher.hpp' ],
      language='c++',
      # Note: On MacOS add before zi_lib:
      # '/opt/homebrew/Cellar/boost/1.75.0_2/include',
      include_dirs=[ np.get_include(), 'zi_lib/', './' ],
      extra_compile_args=[
        '-std=c++11','-O3', '-ffast-math'
      ]),
  ],
)

