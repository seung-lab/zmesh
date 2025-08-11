import numpy as np
import zmesh

labels = np.zeros( (20, 20, 20), dtype=np.uint32)
labels[1:-1, 1:-1, 1:-1] = 1

mesher = zmesh.Mesher( (1, 1, 1) )
mesher.mesh(labels)
mesh = mesher.get_mesh(1, normals=False, simplification_factor=0, max_simplification_error=100)


meshes = zmesh.chunk_mesh(
  mesh,
  [10.,10.,10.],
)
m = zmesh.Mesh.concatenate(*list(meshes.values()))

with open('cube.obj', 'bw') as f:
  f.write(m.to_obj())
