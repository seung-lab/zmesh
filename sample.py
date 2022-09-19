import numpy as np
import zmesh
import time

labels = np.load("connectomics.npy")
mesher = zmesh.Mesher( (32, 32, 40) )

s = time.time()
mesher.mesh(labels)
print(f"MC: {time.time() - s:.3f} sec")

s = time.time()
mesh = mesher.get_mesh(labels[400,400,128], normals=False, simplification_factor=0, max_simplification_error=40)
print(f"Simplify: {time.time() - s:.3f} sec")

with open('mesh.ply', 'bw') as f:
  f.write(mesh.to_ply())
