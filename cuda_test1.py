# @File  : cuda_test1.py
# @Author: 沈昌力
# @Date  : 2018/4/20
# @Desc  :
import numba.cuda as cuda
import time
import numpy as np
from numba.cuda.random import create_xoroshiro128p_states, xoroshiro128p_normal_float64

@cuda.jit
def cudaNormalVariateKernel(rng_states, an_array):
    threadId = cuda.grid(1) #cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
    base = threadId*1000
    for i in range(base+1, base+1000):
        an_array[i] = an_array[i-1] + xoroshiro128p_normal_float64(rng_states, threadId)


def cudaTask():
    T = 1000    #threadNum per block
    B = 10      #blockNum per grid
    AS = 1000   #loopNum in one thread
    res = np.zeros(T*B*AS,dtype='float64')  #创建数组，初始化0，参考numpy使用手册

    rng_states = create_xoroshiro128p_states(T*B, seed=1)   #参见http://numba.pydata.org/numba-doc/0.35.0/cuda/random.html
    cudaNormalVariateKernel[B, T](rng_states, res)

    return res

if __name__ == "__main__":

    t1 = time.clock()
    res = cudaTask()
    t2 = time.clock()