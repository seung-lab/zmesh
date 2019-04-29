# zmesh
[![PyPI version](https://badge.fury.io/py/zmesh.svg)](https://badge.fury.io/py/zmesh)

Multi-Label Marching Cubes &amp; Simplification

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
```

## Installation 

*Requires a C++ compiler*

```bash
sudo apt-get install python3-dev libboost-all-dev
pip install zmesh
```

## Performance Tuning

- The mesher will consume about double memory in 64 bit mode if the size of the 
object exceeds <511, 1023, 511> on the x, y, or z axes. This is due to a limitation 
of the 32-bit format. It might be possible to get x to 1023 as well.

## Related Projects 

- [zi_lib](https://github.com/zlateski/zi_lib) - zmesh makes heavy use of Alek's C++ library.
- [Igneous](https://github.com/seung-lab/igneous) - Visualization of connectomics data using cloud computing.

## Credits

Thanks to Aleks Zlateski for creating and sharing this beautiful mesher.  

Later changes by Will Silversmith and Nico Kemnitz.  

## References  

1. W. Lorensen and H. Cline. "Marching Cubes: A High Resolution 3D Surface Construction Algorithm". pp 163-169. Computer Graphics, Volume 21, Number 4, July 1987.
2. TK Quadratic Edge Collapse Paper