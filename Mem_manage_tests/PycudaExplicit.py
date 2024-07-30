# Pycuda Square Function using explicit HtoD and DtoH (memcpy)

import pycuda.autoinit
from pycuda.compiler import SourceModule
import pycuda.driver as cuda
import numpy as np

module = SourceModule("""
__global__
void square(size_t n, float *out, float *in){
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = index; i < n; i += stride)
      out[i] = in[i] * in[i];
}
""")

square = module.get_function("square")

# Constants
d_type = np.float32
ARRAY_SIZE = 1 << 20
ARRAY_BYTES = ARRAY_SIZE * np.float32().nbytes
loope = 10

# Declare GPU arrays
inp_gpu = cuda.mem_alloc(ARRAY_BYTES)
out_gpu = cuda.mem_alloc(ARRAY_BYTES)
# Declare host arrays
inp = np.empty(ARRAY_SIZE).astype(d_type)
out = np.empty_like(inp)

# Init host input array
for i in range(ARRAY_SIZE):
    inp[i] = float(i)

# Host to Device
cuda.memcpy_htod(inp_gpu, inp)

# Kernel
numThreads = 1024
numBlocks = (ARRAY_SIZE + numThreads - 1) // numThreads

print("Square Function")
print(f"numElements: {ARRAY_SIZE}")
print(f"numBlocks: {numBlocks}, numThreads: {numThreads}")
for i in range(0,loope):
    square(np.uintp(ARRAY_SIZE), out_gpu, inp_gpu, block=(numThreads, 1, 1), grid=(numBlocks, 1)) 
cuda.Context.synchronize()

# Device to Host
cuda.memcpy_dtoh(out, out_gpu)

# Error checking
errCount = 0
for i in range(ARRAY_SIZE):
    if inp[i] * inp[i] != out[i]:
        errCount += 1
print(f"CUDA Error count: {errCount}")

# Free memory (automatically managed by Python garbage collector)