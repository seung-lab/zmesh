from typing import Literal

from zmesh.mesh import Mesh
from zmesh._zmesh import *

import numpy as np

def dust(
  mesh:Mesh,
  threshold:float,
  metric:Literal["vertices", "faces", "surface_area", "volume"] = "vertices",
  ccl:Literal["vertices", "faces"] = "vertices",
  invert:bool = False,
) -> Mesh:
  """
  Remove connected components smaller than threshold.

  You can pick which kind of connected components you want to trace.
  vertices means use the vertex graph, faces means use common edges between faces.

  metric means which type of threshold to apply. 
  vertices means number of vertices, faces means number of faces, 
  surface area and volume are somewhat self explainatory.
  """
  if metric in ["surface_area", "volume"]:
    raise ValueError(f"Metric {metric} not yet implemented.")
  elif metric not in ["faces", "vertices"]:
    raise ValueError(f"Metric {metric} not supported. Must be one of: 'vertices', 'faces', 'surface_area', 'volume'.")

  if ccl == "vertices":
    ccls = vertex_connected_components(mesh)
  elif ccl == "faces":
    ccls = face_connected_components(mesh)
  else:
    raise ValueError(f"Connected components type {ccl} not supported.")

  if metric == "vertices":
    if invert:
      ccls = [
        cc for cc in ccls
        if cc.vertices.shape[0] < threshold
      ]
    else:
      ccls = [
        cc for cc in ccls
        if cc.vertices.shape[0] >= threshold
      ]
  else:
    if invert:
      ccls = [
        cc for cc in ccls
        if cc.faces.shape[0] < threshold
      ]
    else:
      ccls = [
        cc for cc in ccls
        if cc.faces.shape[0] >= threshold
      ]

  if len(ccls) == 0:
    return Mesh(id=mesh.id)
  else:
    return Mesh.concatenate(*ccls)

def largest_k(
  mesh:Mesh, 
  k:int,
  metric:Literal["vertices", "faces", "surface_area", "volume"] = "vertices",
  ccl:Literal["vertices", "faces"] = "vertices",
  invert:bool = False,
) -> Mesh:
  """Keep only the largest k components in the mesh."""

  if metric in ["surface_area", "volume"]:
    raise ValueError(f"Metric {metric} not yet implemented.")
  elif metric not in ["faces", "vertices"]:
    raise ValueError(
      f"Metric {metric} not supported. Must be one of: 'vertices', 'faces', 'surface_area', 'volume'."
    )

  if ccl == "vertices":
    ccls = vertex_connected_components(mesh)
  elif ccl == "faces":
    ccls = face_connected_components(mesh)
  else:
    raise ValueError(f"Connected components type {ccl} not supported.")

  if k >= len(ccls):
    return mesh.clone()

  if metric == "vertices":
    scores = [ (i, cc.vertices.shape[0]) for i, cc in enumerate(ccls) ]
  else:
    scores = [ (i, cc.faces.shape[0]) for i, cc in enumerate(ccls) ]

  scores.sort(key=lambda x: x[1], reverse=(not invert))
  idx = [ i for i, score in scores[:k] ]

  return Mesh.concatenate(*[ ccls[i] for i in idx ])



