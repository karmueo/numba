# @File  : CUDA.py
# @Author: 沈昌力
# @Date  : 2018/5/17
# @Desc  :
from numba import cuda, float32
import numpy as np
import math

# Controls threads per block and shared memory usage.
# The computation will be done on blocks of TPBxTPB elements.
TPB = 16

@cuda.jit
def fast_matmul(A, B, C):
    # Define an array in the shared memory
    # The size and type of the arrays must be known at compile time
    sA = cuda.shared.array(shape=(TPB, TPB), dtype=float32)
    sB = cuda.shared.array(shape=(TPB, TPB), dtype=float32)

    x, y = cuda.grid(2)

    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y
    bpg = cuda.gridDim.x    # blocks per grid

    if x >= C.shape[0] and y >= C.shape[1]:
        # Quit if (x, y) is outside of valid C boundary
        return

    # Each thread computes one element in the result matrix.
    # The dot product is chunked into dot products of TPB-long vectors.
    tmp = 0.
    for i in range(bpg):
        # Preload data into shared memory
        sA[tx, ty] = A[x, ty + i * TPB]
        sB[tx, ty] = B[tx + i * TPB, y]

        # Wait until all threads finish preloading
        cuda.syncthreads()

        # Computes partial product on the shared memory
        for j in range(TPB):
            tmp += sA[tx, j] * sB[j, ty]

        # Wait until all threads finish computing
        cuda.syncthreads()

    C[x, y] = tmp


@cuda.jit
def increment_a_2D_array(an_array):
    x, y = cuda.grid(2)
    if x < an_array.shape[0] and y < an_array.shape[1]:
       an_array[x, y] += 1

threadsperblock = (TPB, TPB)

an_array = np.random.rand(100, 100)
blockspergrid_x = math.ceil(an_array.shape[0] / threadsperblock[0])
blockspergrid_y = math.ceil(an_array.shape[1] / threadsperblock[1])
blockspergrid = (blockspergrid_x, blockspergrid_y)
increment_a_2D_array[blockspergrid, threadsperblock](an_array)

# import time
#
# a = np.random.rand(100, 100)
# b = np.random.rand(100, 100)
# c = np.zeros(shape=(100,100))
# blockspergrid_x = math.ceil(a.shape[0] / threadsperblock[0])
# blockspergrid_y = math.ceil(a.shape[1] / threadsperblock[1])
# blockspergrid = (blockspergrid_x, blockspergrid_y)
# start = time.time()
# fast_matmul[blockspergrid, threadsperblock](a, b, c)
# end = time.time()
# print(c)
# print('计算时间=%f' %(end-start))
