import numpy as np
import zmesh

labels = np.zeros( (50, 50, 50), dtype=np.uint32)
labels[1:-1, 1:-1, 1:-1] = 1

mesher = zmesh.Mesher( (1,1,1) )
mesher.mesh(labels)
mesh = mesher.get_mesh(1, normals=False, simplification_factor=0, max_simplification_error=100)

print(mesh)

with open('wow.obj', 'bw') as f:
  f.write(mesh.to_obj())

with open('wow.ply', 'bw') as f:
  f.write(mesh.to_ply())

