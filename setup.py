#!/usr/bin/env python

import os
import setuptools
import sys

class NumpyImport:
  def __repr__(self):
    import numpy as np

    return np.get_include()

  __fspath__ = __repr__

extra_compile_args = []
if sys.platform == 'win32':
  extra_compile_args += [
    '/std:c++20', '/O2'
  ]
else:
  extra_compile_args += [
    '-std=c++20', '-O3', '-Wno-unused-local-typedefs',
  ]

if sys.platform == 'darwin':
  extra_compile_args += [ '-stdlib=libc++', '-mmacosx-version-min=10.9' ]

include_dirs = [ str(NumpyImport()), 'zi_lib/', './' ]

define_macros = [
    ("NPY_NO_DEPRECATED_API", 1),
    ("NPY_1_7_API_VERSION", 1),
]


def is_truthy(v):
    return v is not None and str(v).lower() in ("1", "true", "yes", "on")

check_asserts = os.environ.get("ZMESH_CHECK_ASSERTS", False)
if not is_truthy(check_asserts):
  define_macros.insert(0, ("NDEBUG", 1))

setuptools.setup(
  setup_requires=['pbr', 'numpy', 'cython'],
  python_requires=">=3.8",
  pbr=True,
  define_macros=define_macros,
  extras_require={
    "viewer": [ "vtk" ],
  },
  ext_modules=[
    setuptools.Extension(
      'zmesh._zmesh',
      sources=[ 'zmesh/_zmesh.pyx' ],
      depends=[ 'zmesh/cMesher.hpp' ],
      language='c++',
      include_dirs=include_dirs,
      extra_compile_args=extra_compile_args
    ),
  ],
)

