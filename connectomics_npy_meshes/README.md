These meshes were produced from connectomics.npy using the following script with zmesh 1.1.0

```python
import numpy as np
import zmesh
from tqdm import tqdm

labels = np.load("connectomics.npy")
mesher = zmesh.Mesher( (32, 32, 40) )
mesher.mesh(labels)


for lbl in tqdm(mesher.ids()):
  mesh = mesher.get_mesh(lbl, normals=False, simplification_factor=0, max_simplification_error=40)

  with open(f"./connectomics_npy_meshes/unsimplified/{lbl}.ply", "wb") as f:
    f.write(mesh.to_ply())

  mesh = mesher.get_mesh(lbl, normals=False, simplification_factor=100, max_simplification_error=40)

  with open(f"./connectomics_npy_meshes/simplified/{lbl}.ply", "wb") as f:
    f.write(mesh.to_ply())
```