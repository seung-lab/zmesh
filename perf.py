# marching cubes (connectomics.npy 512^3, uint32): 24.395s, 5.49 MVx/sec, N=1
# marching cubes (random, 448^3 uint64): 72.481s, 1.23 MVx/sec, N=1

import numpy as np

import zmesh
import time

def result(label, dt, data, N):
	voxels = data.size
	mvx = voxels // (10 ** 6)
	print(f"{label}: {dt:02.3f}s, {N * mvx / dt:.2f} MVx/sec, N={N}")

def test_marching_cubes():
	labels = np.load("./connectomics.npy")
	labels = np.ascontiguousarray(labels)
	mesher = zmesh.Mesher((1,1,1))

	N = 1
	start = time.time()
	for _ in range(N):
		mesher.mesh(labels)
	end = time.time()
	result("marching cubes (connectomics.npy)", end - start, labels, N=N)

	labels = np.random.randint(0,1000, size=(448,448,448), dtype=np.uint32)
	labels = np.ascontiguousarray(labels)
	mesher = zmesh.Mesher((1,1,1))

	N = 1
	start = time.time()
	for _ in range(N):
		mesher.mesh(labels)
	end = time.time()
	result("marching cubes (random)", end - start, labels, N=N)

test_marching_cubes()