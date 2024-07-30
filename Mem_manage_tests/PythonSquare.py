import numpy as np
from timeit import default_timer as timer

def square(n, out, inp):
    for i in range(n):
        out[i] = inp[i] * inp[i]

# Constants
d_type = np.float32
ARRAY_SIZE = 1 << 20
ARRAY_BYTES = np.zeros(ARRAY_SIZE).astype(d_type).nbytes
loope = 10

# Init arrays
inp = np.zeros(ARRAY_SIZE).astype(d_type)
out = np.empty_like(inp)
for i in range(ARRAY_SIZE):
    inp[i] = float(i)

# Function
print("Square Function")
print(f"numElements: {ARRAY_SIZE}")
exec_time = 0
for i in range(0,loope):
    start = timer()
    square(ARRAY_SIZE, out, inp) 
    end = timer()
    exec_time += end - start

exec_time = exec_time / 10   
print(f"Avg Execution Time: {exec_time}")

# Free memory (automatically managed by Python garbage collector)