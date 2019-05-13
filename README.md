## zmesh: Multi-Label Marching Cubes &amp; Mesh Simplification
[![Build Status](https://travis-ci.org/seung-lab/zmesh.svg?branch=master)](https://travis-ci.org/seung-lab/zmesh) [![PyPI version](https://badge.fury.io/py/zmesh.svg)](https://badge.fury.io/py/zmesh)  



```python
from zmesh import Mesher

labels = ... # some dense volumetric labeled image
mesher = Mesher( (4,4,40) ) # anisotropy of image
mesher.mesh(labels) # initial marching cubes pass

meshes = []
for obj_id in mesher.ids():
  meshes.append(
    mesher.get_mesh(
      obj_id, 
      normals=False, # whether to calculate normals or not

      # tries to reduce triangles by this factor
      # 0 disables simplification
      simplification_factor=100, 

      # Max tolerable error in physical distance
      max_simplification_error=8
    )
  )
  mesher.erase(obj_id) # delete high res mesh

mesher.clear() # clear memory retained by mesher

mesh = meshes[0]

mesh.vertices
mesh.faces 
mesh.normals

# Extremely common obj format
with open('iconic_doge.obj', 'wb') as f:
  f,write(mesh.to_obj())

# Common binary format
with open('iconic_doge.ply', 'wb') as f:
  f,write(mesh.to_ply())

# Neuroglancer Precomputed format
with open('10001001:0', 'wb') as f:
  f.write(mesh.to_precomputed())
```

## Installation 

If binaries are available for your system:

```bash
pip install zmesh
```

*Requires a C++ compiler*

```bash
sudo apt-get install python3-dev libboost-all-dev
pip install zmesh --no-binary :all:
```

## Performance Tuning

- The mesher will consume about double memory in 64 bit mode if the size of the 
object exceeds <511, 1023, 511> on the x, y, or z axes. This is due to a limitation 
of the 32-bit format. It might be possible to get x to 1023 as well.
- Input labels are converted to uint32 or uint64. Use one of these data types to avoid a copy.
- The mesher processes in C order.

## Related Projects 

- [zi_lib](https://github.com/zlateski/zi_lib) - zmesh makes heavy use of Aleks' C++ library.
- [Igneous](https://github.com/seung-lab/igneous) - Visualization of connectomics data using cloud computing.

## Credits

Thanks to Aleks Zlateski for creating and sharing this beautiful mesher.  

Later changes by Will Silversmith and Nico Kemnitz.  

## References  

1. W. Lorensen and H. Cline. "Marching Cubes: A High Resolution 3D Surface Construction Algorithm". pp 163-169. Computer Graphics, Volume 21, Number 4, July 1987.
2. TK Quadratic Edge Collapse Paper
