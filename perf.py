# ZMESH
# marching cubes (blank): 0.897s, 149.42 MVx/sec, N=1
# marching cubes (filled): 0.966s, 138.78 MVx/sec, N=1
# marching cubes (connectomics.npy 512^3, uint32): 3.132s, 42.78 MVx/sec, N=1
# marching cubes (random 448^3 uint64): 25.092s, 3.55 MVx/sec, N=1
# SKIMAGE
# marching cubes (blank) NOT HANDLED
# marching cubes (filled) NOT HANDLED
# marching cubes (connectomics.npy 512^3, uint32): 5.359s, 25.00 MVx/sec, N=1
# marching cubes (random 448^3 uint64): 69.927s, 1.27 MVx/sec, N=1


import numpy as np

import zmesh
import time

def result(label, dt, data, N):
    voxels = data.size
    mvx = voxels // (10 ** 6)
    print(f"{label}: {dt:02.3f}s, {N * mvx / dt:.2f} MVx/sec, N={N}")

def test_zmesh_marching_cubes():
    labels = np.zeros((512,512,512), dtype=np.uint8, order="C")
    mesher = zmesh.Mesher((1,1,1))
    N = 1
    start = time.time()
    for _ in range(N):
        mesher.mesh(labels)
    end = time.time()
    result("marching cubes (blank)", end - start, labels, N=N)

    labels = np.ones((512,512,512), dtype=np.uint8, order="C")
    mesher = zmesh.Mesher((1,1,1))
    N = 1
    start = time.time()
    for _ in range(N):
        mesher.mesh(labels, close=True)
    end = time.time()
    result("marching cubes (filled)", end - start, labels, N=N)

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

def test_scikit_marching_cubes():
    import skimage.measure

    print("marching cubes (blank) NOT HANDLED")
    print("marching cubes (filled) NOT HANDLED")
    
    labels = np.ones((512,512,512))
    labels = np.load("./connectomics.npy")
    labels = np.ascontiguousarray(labels)

    N = 1
    start = time.time()
    for _ in range(N):
        skimage.measure.marching_cubes(labels)
    end = time.time()
    result("marching cubes (connectomics.npy)", end - start, labels, N=N)

    labels = np.random.randint(0,1000, size=(448,448,448), dtype=np.uint32)
    labels = np.ascontiguousarray(labels)

    N = 1
    start = time.time()
    for _ in range(N):
        skimage.measure.marching_cubes(labels)
    end = time.time()
    result("marching cubes (random)", end - start, labels, N=N)

print("ZMESH")
test_zmesh_marching_cubes()
print("SKIMAGE")
test_scikit_marching_cubes()