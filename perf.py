# On a Macbook Pro M3 2026-08-05
# ZMESH
# marching cubes (blank): 0.587s, 684.86 MVx/sec, N=3
# marching cubes (filled): 0.717s, 560.39 MVx/sec, N=3
# marching cubes (connectomics.npy): 3.278s, 122.64 MVx/sec, N=3
# marching cubes (connectomics.npy, compressed): 4.339s, 92.66 MVx/sec, N=3
# marching cubes (random): 7.033s, 12.65 MVx/sec, N=1
# SKIMAGE
# marching cubes (blank) NOT HANDLED
# marching cubes (filled) NOT HANDLED
# marching cubes (connectomics.npy): 9.861s, 40.77 MVx/sec, N=3
# marching cubes (random): 39.807s, 2.24 MVx/sec, N=1

# simplification (connectomics.npy): 434.549s, 0.31 MVx/sec, N=1

import numpy as np
import crackle

import zmesh
import time
from tqdm import tqdm

def result(label, dt, data, N):
    voxels = data.size
    mvx = voxels // (10 ** 6)
    print(f"{label}: {dt:02.3f}s, {N * mvx / dt:.2f} MVx/sec, N={N}")

def test_zmesh_marching_cubes():
    labels = np.zeros((512,512,512), dtype=np.uint8, order="C")
    mesher = zmesh.Mesher((1,1,1))
    N = 3
    start = time.time()
    for _ in range(N):
        mesher.mesh(labels)
    end = time.time()
    result("marching cubes (blank)", end - start, labels, N=N)

    labels = np.ones((512,512,512), dtype=np.uint8, order="C")
    mesher = zmesh.Mesher((1,1,1))
    N = 3
    start = time.time()
    for _ in range(N):
        mesher.mesh(labels, close=True)
    end = time.time()
    result("marching cubes (filled)", end - start, labels, N=N)

    labels = crackle.load("./connectomics.npy.ckl.gz")
    labels = np.ascontiguousarray(labels)
    mesher = zmesh.Mesher((1,1,1))

    N = 3
    start = time.time()
    for _ in range(N):
        mesher.mesh(labels)
    end = time.time()
    result("marching cubes (connectomics.npy)", end - start, labels, N=N)

    labels = crackle.compressa(labels)
    mesher = zmesh.Mesher((1,1,1))

    N = 3
    start = time.time()
    for _ in range(N):
        mesher.mesh(labels)
    end = time.time()
    result("marching cubes (connectomics.npy, compressed)", end - start, labels, N=N)

    labels = np.random.randint(0,1000, size=(448,448,448), dtype=np.uint32)
    # labels = np.ascontiguousarray(labels)
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
    labels = crackle.load("connectomics.npy.ckl.gz")
    labels = np.ascontiguousarray(labels)

    N = 3
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

# Ran zmesh simplification and summed the sizes
# of the meshes.
# factor 0 max error 0:  1614121164 bytes (1.0x)
# factor 100 max error 0: 503561448 bytes (3.2x)
# factor 100 max error 1: 350636148 bytes (4.6x)

def test_zmesh_simplification():
    labels = crackle.load("connectomics.npy.ckl.gz")
    mesher = zmesh.Mesher((1,1,1))
    mesher.mesh(labels)

    N = 3
    start = time.time()
    for i in range(N):
        for label in tqdm(mesher.ids()):
            mesher.get(label, 
                reduction_factor=100, 
                # Max tolerable error in physical distance
                max_error=40,
            )
    end = time.time()
    result("simplification (connectomics.npy)", end - start, labels, N=N)

print("ZMESH")
test_zmesh_marching_cubes()
print("SKIMAGE")
test_scikit_marching_cubes()
print("ZMESH SIMPLIFICATION")
test_zmesh_simplification()