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

# If you don't mind shuffling the vertices and faces
# from older versions of zmesh, this option unlocks
# some performance optimizations.
mesher.mesh(labels, preserve_order=False) 

meshes = []
for obj_id in mesher.ids():
  meshes.append(
    mesher.get(
      obj_id, 
      normals=False, # whether to calculate normals or not

      # tries to reduce triangles by this factor
      # 0 disables simplification
      reduction_factor=100, 

      # Max tolerable error in physical distance
      # note: if max_error is not set, the max error
      # will be set equivalent to one voxel along the 
      # smallest dimension.
      max_error=8,
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
  # same as reduction_factor in get
  reduction_factor=100, 
  # same as max_error in get
  max_error=40, 
  compute_normals=False, # whether to also compute face normals
) # apply simplifier to a pre-existing mesh

# use an fqmr derived non-topology preserving algorithm
# this useful particularly for multi-resolution meshes
# where visual appearence is more important than connectivity
# This function has many parameters, see help(zmesh.simplify_fqmr)
mesh = zmesh.simplify_fqmr(
  mesh, 
  triangle_count=(mesh.faces.shape[0] // 10),
)

# compute normals on a pre-existing mesh
mesh = zmesh.compute_normals(mesh) 

# run face based connected components
ccls = zmesh.face_connected_components(mesh)
# run vertex based connected components
ccls = zmesh.vertex_connected_components(mesh)

# remove small components based on vertices or faces
mesh = zmesh.dust(mesh, threshold=100, metric="vertices")
# remove components bigger than the threshold using invert
mesh = zmesh.dust(mesh, threshold=100, metric="vertices", invert=True)
# retain only the largest k connected components
mesh = zmesh.largest_k(mesh, k=1, metric="vertices")
# retain only the smallest k connected components
mesh = zmesh.largest_k(mesh, k=1, metric="vertices", invert=True)

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

Note: `mesher.get_mesh` has been deprecated in favor of `mesher.get` which fixed a long standing bug where you needed to transpose your data in order to get a mesh in the correct orientation.

## Installation 

If binaries are not available for your system, ensure you have a C++ compiler installed.

```bash
pip install zmesh
```

## Performance Tuning & Notes

- The mesher will consume about double memory in 64 bit mode if the size of the 
object exceeds <1023, 1023, 511> on the x, y, or z axes. This is due to a limitation 
of the 32-bit format. 
- The mesher is ambidextrous, it can handle C or Fortran order arrays.
- The maximum vertex range supported `.simplify` after converting to voxel space is 2<sup>20</sup> (appx. 1M) due to the packed 64-bit vertex format.

## Related Projects 

- [zi_lib](https://github.com/zlateski/zi_lib) - zmesh makes heavy use of Aleks' C++ library.
- [Igneous](https://github.com/seung-lab/igneous) - Visualization of connectomics data using cloud computing.

## Example Performance 

 This was a limited experiment conducted on a Macbook Pro M3 comparing Zmesh 1.12.0 
 and scikit-image==0.26.0. The volume is a 512^3 uint32 segmentation of a mouse visual cortex containing 2523 shapes of various sizes including parts of dendrites, a nucleus, and a glia.  

 Note that this is not really an apples-to-apples comparison because scikit-image is specialized for continuous value images like CT scans not segmentation, and so the resulting meshes are very different.

 This is `mesher.mesh(image)`.

| Marching Cubes Data     | ZMESH Time (s) | ZMESH MVx/sec | SKIMAGE Time (s) | SKIMAGE MVx/sec |  N  |
|-------------------------|----------------|---------------|------------------|-----------------|-----|
| Black                   | 0.891          | 451.35        | NOT HANDLED      | —               | 3   |
| Filled                  | 0.961          | 418.12        | NOT HANDLED      | —               | 3   |
| connectomics.npy        | 4.107          | 97.89         | 9.861            | 40.77           | 3   |
| random                  | 6.950          | 12.81         | 40.509           | 2.20            | 1   |

The meshes can then be extracted (`mesher.get`):

| Simplification Factor | Max Error | Labels per Second | N  |
|-----------------------|-----------|-------------------|----|
| 0                     | N/A       | 478.2             | 3  |
| 100                   | 40        |  14.7             | 3  |

## Credits

Thanks to Aleks Zlateski for creating and sharing this beautiful mesher.  

Later changes by Will Silversmith, Nico Kemnitz, and Jingpeng Wu. 

Thank you to Sven Forstmann, Kristof S., Br&eacute;nainn Woodsend, and others for pyfqmr which we have adapted here for non-topology preserving simplification and fast OBJ reading. See https://github.com/Kramer84/pyfqmr-Fast-Quadric-Mesh-Reduction/

## References  

1. W. Lorensen and H. Cline. "Marching Cubes: A High Resolution 3D Surface Construction Algorithm". pp 163-169. Computer Graphics, Volume 21, Number 4, July 1987. ([link](https://people.eecs.berkeley.edu/~jrs/meshpapers/LorensenCline.pdf))  
2. M. Garland and P. Heckbert. "Surface simplification using quadric error metrics". SIGGRAPH '97: Proceedings of the 24th annual conference on Computer graphics and interactive techniques. Pages 209–216. August 1997. doi: 10.1145/258734.258849 ([link](https://mgarland.org/files/papers/quadrics.pdf))  
3. H. Hoppe. "New Quadric Metric for Simplifying Meshes with Appearance Attributes". IEEE Visualization 1999 Conference. pp. 59-66. doi: 10.1109/VISUAL.1999.809869 ([link](http://hhoppe.com/newqem.pdf))  
