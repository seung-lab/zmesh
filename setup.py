#!/usr/bin/env python

import os
import setuptools
import numpy as np

# NOTE: If zmesh.cpp does not exist, you must run
# cython --cplus -I./zi_lib/ zmesh.pyx

include_dirs = [ np.get_include(), 'zi_lib/', './' ]

# Note: On MacOS add boost before zi_lib:
# /opt/homebrew/Cellar/boost/1.75.0_2/include
boost_dir = os.environ.get("BOOST_ROOT", None) 
if boost_dir:
  boost_dir = os.path.join(boost_dir, 'include')
  include_dirs.insert(1, boost_dir)

setuptools.setup(
  setup_requires=['pbr', 'numpy'],
  python_requires="~=3.6", # >= 3.6 < 4.0
  pbr=True,
  ext_modules=[
    setuptools.Extension(
      'zmesh._zmesh',
      sources=[ 'zmesh/_zmesh.cpp' ],
      depends=[ 'cMesher.hpp' ],
      language='c++',
      include_dirs=include_dirs,
      extra_compile_args=[
        '-std=c++11','-O3', '-ffast-math'
      ]),
  ],
)

