## zmesh: Multi-Label Marching Cubes &amp; Mesh Simplification
[![Tests](https://github.com/seung-lab/zmesh/actions/workflows/test.yml/badge.svg?branch=master)](https://github.com/seung-lab/zmesh/actions/workflows/test.yml) [![PyPI version](https://badge.fury.io/py/zmesh.svg)](https://badge.fury.io/py/zmesh)  

```python
from zmesh import Mesher

labels = ... # some dense volumetric labeled image
mesher = Mesher( (4,4,40) ) # anisotropy of image

# initial marching cubes pass
# close controls whether meshes touching
# the image boundary are left open or closed
mesher.mesh(labels, close=False) 

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
      max_simplification_error=8,
      # whether meshes should be centered in the voxel
      # on (0,0,0) [False] or (0.5,0.5,0.5) [True]
      voxel_centered=False, 
    )
  )
  mesher.erase(obj_id) # delete high res mesh

mesher.clear() # clear memory retained by mesher

mesh = meshes[0]
mesh = mesher.simplify(
  mesh, 
  # same as simplification_factor in get_mesh
  reduction_factor=100, 
  # same as max_simplification_error in get_mesh
  max_error=40, 
  compute_normals=False, # whether to also compute face normals
) # apply simplifier to a pre-existing mesh

# compute normals without simplifying
mesh = mesher.compute_normals(mesh) 

mesh.vertices
mesh.faces 
mesh.normals
mesh.triangles() # compute triangles from vertices and faces

# Extremely common obj format
with open('iconic_doge.obj', 'wb') as f:
  f.write(mesh.to_obj())

# Common binary format
with open('iconic_doge.ply', 'wb') as f:
  f.write(mesh.to_ply())

# Neuroglancer Precomputed format
with open('10001001:0', 'wb') as f:
  f.write(mesh.to_precomputed())
```

## Installation 

If binaries are available for your system:

```bash
pip install zmesh
```

*Requires a C++ compiler and boost*

Note that you may need to set the environment variable `BOOST_ROOT`.

```bash
sudo apt-get install python3-dev libboost-all-dev
pip install zmesh --no-binary :all:
```

## Performance Tuning

- The mesher will consume about double memory in 64 bit mode if the size of the 
object exceeds <1023, 1023, 511> on the x, y, or z axes. This is due to a limitation 
of the 32-bit format. 
- The mesher processes in C order.

## Related Projects 

- [zi_lib](https://github.com/zlateski/zi_lib) - zmesh makes heavy use of Aleks' C++ library.
- [Igneous](https://github.com/seung-lab/igneous) - Visualization of connectomics data using cloud computing.

## Credits

Thanks to Aleks Zlateski for creating and sharing this beautiful mesher.  

Later changes by Will Silversmith, Nico Kemnitz, and Jingpeng Wu. 

## References  

1. W. Lorensen and H. Cline. "Marching Cubes: A High Resolution 3D Surface Construction Algorithm". pp 163-169. Computer Graphics, Volume 21, Number 4, July 1987. ([link](https://people.eecs.berkeley.edu/~jrs/meshpapers/LorensenCline.pdf))  
2. M. Garland and P. Heckbert. "Surface simplification using quadric error metrics". SIGGRAPH '97: Proceedings of the 24th annual conference on Computer graphics and interactive techniques. Pages 209–216. August 1997. doi: 10.1145/258734.258849 ([link](https://mgarland.org/files/papers/quadrics.pdf))  
3. H. Hoppe. "New Quadric Metric for Simplifying Meshes with Appearance Attributes". IEEE Visualization 1999 Conference. pp. 59-66. doi: 10.1109/VISUAL.1999.809869 ([link](http://hhoppe.com/newqem.pdf))  
