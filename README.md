# GPU Puzzles
- by [Sasha Rush](http://rush-nlp.com) - [srush_nlp](https://twitter.com/srush_nlp)

![](https://github.com/srush/GPU-Puzzles/raw/main/cuda.png)

GPU architectures are critical to machine learning, and seem to be
becoming even more important every day. However, you can be an expert
in machine learning without ever touching GPU code. It is hard to gain
intuition working through abstractions. 

This notebook is an attempt to teach beginner GPU programming in a
completely interactive fashion. Instead of providing text with
concepts, it throws you right into coding and building GPU
kernels. The exercises use NUMBA which directly maps Python
code to CUDA kernels. It looks like Python but is basically
identical to writing low-level CUDA code. 
In a few hours, I think you can go from basics to
understanding the real algorithms that power 99% of deep learning
today. If you do want to read the manual, it is here:

[NUMBA CUDA Guide](https://numba.readthedocs.io/en/stable/cuda/index.html)

I recommend doing these in Colab, as it is easy to get started.  Be
sure to make your own copy, turn on GPU mode in the settings (`Runtime / Change runtime type`, then set `Hardware accelerator` to `GPU`), and
then get to coding.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/srush/GPU-Puzzles/blob/main/GPU_puzzlers.ipynb)

(If you are into this style of puzzle, also check out my [Tensor
Puzzles](https://github.com/srush/Tensor-Puzzles) for PyTorch.)


```python
!pip install -qqq git+https://github.com/chalk-diagrams/planar git+https://github.com/danoneata/chalk@srush-patch-1
!wget -q https://github.com/srush/GPU-Puzzles/raw/main/robot.png https://github.com/srush/GPU-Puzzles/raw/main/lib.py
```


```python
import numba
import numpy as np
import warnings
from lib import CudaProblem, Coord
```


```python
warnings.filterwarnings(
    action="ignore", category=numba.NumbaPerformanceWarning, module="numba"
)
```

## Puzzle 1: Map

Implement a "kernel" (GPU function) that adds 10 to each position of vector `a`
and stores it in vector `out`.  You have 1 thread per position.

**Warning** This code looks like Python but it is really CUDA! You cannot use
standard python tools like list comprehensions or ask for Numpy properties
like shape or size (if you need the size, it is given as an argument).
The puzzles only require doing simple operations, basically
+, *, simple array indexing, for loops, and if statements.
You are allowed to use local variables. 
If you get an
error it is probably because you did something fancy :). 

*Tip: Think of the function `call` as being run 1 time for each thread.
The only difference is that `cuda.threadIdx.x` changes each time.*


```python
def map_spec(a):
    return a + 10


def map_test(cuda):
    def call(out, a) -> None:
        local_i = cuda.threadIdx.x
        # FILL ME IN (roughly 1 lines)
        out[local_i] = a[local_i] + 10

    return call


SIZE = 4
out = np.zeros((SIZE,))
a = np.arange(SIZE)
problem = CudaProblem(
    "Map", map_test, [a], out, threadsperblock=Coord(SIZE, 1), spec=map_spec
)
problem.show()
```

    # Map
     
       Score (Max Per Thread):
       |  Global Reads | Global Writes |  Shared Reads | Shared Writes |
       |             1 |             1 |             0 |             0 | 
    





    
![svg](GPU_puzzlers_files/GPU_puzzlers_14_1.svg)
    




```python
problem.check()
```

    Passed Tests!






<video alt="test" controls autoplay=1>
    <source src="https://openpuppies.com/mp4/fqHxOGI.mp4"  type="video/mp4">
</video>




## Puzzle 2 - Zip

Implement a kernel that adds together each position of `a` and `b` and stores it in `out`.
You have 1 thread per position.


```python
def zip_spec(a, b):
    return a + b


def zip_test(cuda):
    def call(out, a, b) -> None:
        local_i = cuda.threadIdx.x
        # FILL ME IN (roughly 1 lines)
        out[local_i] = a[local_i] + b[local_i]

    return call


SIZE = 4
out = np.zeros((SIZE,))
a = np.arange(SIZE)
b = np.arange(SIZE)
problem = CudaProblem(
    "Zip", zip_test, [a, b], out, threadsperblock=Coord(SIZE, 1), spec=zip_spec
)
problem.show()
```

    # Zip
     
       Score (Max Per Thread):
       |  Global Reads | Global Writes |  Shared Reads | Shared Writes |
       |             2 |             1 |             0 |             0 | 
    





    
![svg](GPU_puzzlers_files/GPU_puzzlers_17_1.svg)
    




```python
problem.check()
```

    Passed Tests!






<video alt="test" controls autoplay=1>
    <source src="https://openpuppies.com/mp4/aydRUz8.mp4"  type="video/mp4">
</video>




## Puzzle 3 - Guards

Implement a kernel that adds 10 to each position of `a` and stores it in `out`.
You have more threads than positions.


```python
def map_guard_test(cuda):
    def call(out, a, size) -> None:
        local_i = cuda.threadIdx.x
        # FILL ME IN (roughly 2 lines)
        if local_i < size:
            out[local_i] = a[local_i] + 10

    return call


SIZE = 4
out = np.zeros((SIZE,))
a = np.arange(SIZE)
problem = CudaProblem(
    "Guard",
    map_guard_test,
    [a],
    out,
    [SIZE],
    threadsperblock=Coord(8, 1),
    spec=map_spec,
)
problem.show()
```

    # Guard
     
       Score (Max Per Thread):
       |  Global Reads | Global Writes |  Shared Reads | Shared Writes |
       |             1 |             1 |             0 |             0 | 
    





    
![svg](GPU_puzzlers_files/GPU_puzzlers_20_1.svg)
    




```python
problem.check()
```

    Passed Tests!






<video alt="test" controls autoplay=1>
    <source src="https://openpuppies.com/mp4/HlaTE8H.mp4"  type="video/mp4">
</video>




## Puzzle 4 - Map 2D

Implement a kernel that adds 10 to each position of `a` and stores it in `out`.
Input `a` is 2D and square. You have more threads than positions.


```python
def map_2D_test(cuda):
    def call(out, a, size) -> None:
        local_i = cuda.threadIdx.x
        local_j = cuda.threadIdx.y
        # FILL ME IN (roughly 2 lines)
        if local_i < size and local_j < size:
            out[local_i, local_j] = a[local_i, local_j] + 10

    return call


SIZE = 2
out = np.zeros((SIZE, SIZE))
a = np.arange(SIZE * SIZE).reshape((SIZE, SIZE))
problem = CudaProblem(
    "Map 2D", map_2D_test, [a], out, [SIZE], threadsperblock=Coord(3, 3), spec=map_spec
)
problem.show()
```

    # Map 2D
     
       Score (Max Per Thread):
       |  Global Reads | Global Writes |  Shared Reads | Shared Writes |
       |             1 |             1 |             0 |             0 | 
    





    
![svg](GPU_puzzlers_files/GPU_puzzlers_23_1.svg)
    




```python
problem.check()
```

    Passed Tests!






<video alt="test" controls autoplay=1>
    <source src="https://openpuppies.com/mp4/MQCIwzT.mp4"  type="video/mp4">
</video>




## Puzzle 5 - Broadcast

Implement a kernel that adds `a` and `b` and stores it in `out`.
Inputs `a` and `b` are vectors. You have more threads than positions.


```python
def broadcast_test(cuda):
    def call(out, a, b, size) -> None:
        local_i = cuda.threadIdx.x
        local_j = cuda.threadIdx.y
        # FILL ME IN (roughly 2 lines)
        if local_i < size and local_j < size:
            out[local_i, local_j] = a[local_i, 0] + b[0, local_j]

    return call


SIZE = 2
out = np.zeros((SIZE, SIZE))
a = np.arange(SIZE).reshape(SIZE, 1)
b = np.arange(SIZE).reshape(1, SIZE)
problem = CudaProblem(
    "Broadcast",
    broadcast_test,
    [a, b],
    out,
    [SIZE],
    threadsperblock=Coord(3, 3),
    spec=zip_spec,
)
problem.show()
```

    # Broadcast
     
       Score (Max Per Thread):
       |  Global Reads | Global Writes |  Shared Reads | Shared Writes |
       |             2 |             1 |             0 |             0 | 
    





    
![svg](GPU_puzzlers_files/GPU_puzzlers_26_1.svg)
    




```python
problem.check()
```

    Passed Tests!






<video alt="test" controls autoplay=1>
    <source src="https://openpuppies.com/mp4/F1SChho.mp4"  type="video/mp4">
</video>




## Puzzle 6 - Blocks

Implement a kernel that adds 10 to each position of `a` and stores it in `out`.
You have fewer threads per block than the size of `a`.

*Tip: A block is a group of threads. The number of threads per block is limited, but we can
have many different blocks. Variable `cuda.blockIdx` tells us what block we are in.*


```python
def map_block_test(cuda):
    def call(out, a, size) -> None:
        i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
        # FILL ME IN (roughly 2 lines)
        if i < size:
            out[i] = a[i] + 10

    return call


SIZE = 9
out = np.zeros((SIZE,))
a = np.arange(SIZE)
problem = CudaProblem(
    "Blocks",
    map_block_test,
    [a],
    out,
    [SIZE],
    threadsperblock=Coord(4, 1),
    blockspergrid=Coord(3, 1),
    spec=map_spec,
)
problem.show()
```

    # Blocks
     
       Score (Max Per Thread):
       |  Global Reads | Global Writes |  Shared Reads | Shared Writes |
       |             1 |             1 |             0 |             0 | 
    





    
![svg](GPU_puzzlers_files/GPU_puzzlers_30_1.svg)
    




```python
problem.check()
```

    Passed Tests!






<video alt="test" controls autoplay=1>
    <source src="https://openpuppies.com/mp4/3V37Hqr.mp4"  type="video/mp4">
</video>




## Puzzle 7 - Blocks 2D

Implement the same kernel in 2D.  You have fewer threads per block
than the size of `a` in both directions.


```python
def map_block2D_test(cuda):
    def call(out, a, size) -> None:
        i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
        # FILL ME IN (roughly 4 lines)
        j = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y
        if i < size and j < size:
            out[i, j] = a[i, j] + 10

    return call


SIZE = 5
out = np.zeros((SIZE, SIZE))
a = np.ones((SIZE, SIZE))

problem = CudaProblem(
    "Blocks 2D",
    map_block2D_test,
    [a],
    out,
    [SIZE],
    threadsperblock=Coord(3, 3),
    blockspergrid=Coord(2, 2),
    spec=map_spec,
)
problem.show()
```

    # Blocks 2D
     
       Score (Max Per Thread):
       |  Global Reads | Global Writes |  Shared Reads | Shared Writes |
       |             1 |             1 |             0 |             0 | 
    





    
![svg](GPU_puzzlers_files/GPU_puzzlers_33_1.svg)
    




```python
problem.check()
```

    Passed Tests!






<video alt="test" controls autoplay=1>
    <source src="https://openpuppies.com/mp4/fqHxOGI.mp4"  type="video/mp4">
</video>




## Puzzle 8 - Shared

Implement a kernel that adds 10 to each position of `a` and stores it in `out`.
You have fewer threads per block than the size of `a`.

**Warning**: Each block can only have a *constant* amount of shared
 memory that threads in that block can read and write to. This needs
 to be a literal python constant not a variable. After writing to
 shared memory you need to call `cuda.syncthreads` to ensure that
 threads do not cross.

(This example does not really need shared memory or syncthreads, but it is a demo.)


```python
TPB = 4
def shared_test(cuda):
    def call(out, a, size) -> None:
        shared = cuda.shared.array(TPB, numba.float32)
        i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
        local_i = cuda.threadIdx.x

        if i < size:
            shared[local_i] = a[i]
            cuda.syncthreads()

        # FILL ME IN (roughly 2 lines)
        out[i] = shared[local_i] + 10

    return call


SIZE = 8
out = np.zeros(SIZE)
a = np.ones(SIZE)
problem = CudaProblem(
    "Shared",
    shared_test,
    [a],
    out,
    [SIZE],
    threadsperblock=Coord(TPB, 1),
    blockspergrid=Coord(2, 1),
    spec=map_spec,
)
problem.show()
```

    # Shared
     
       Score (Max Per Thread):
       |  Global Reads | Global Writes |  Shared Reads | Shared Writes |
       |             1 |             1 |             1 |             1 | 
    





    
![svg](GPU_puzzlers_files/GPU_puzzlers_38_1.svg)
    




```python
problem.check()
```

    Passed Tests!






<video alt="test" controls autoplay=1>
    <source src="https://openpuppies.com/mp4/9hRi2jN.mp4"  type="video/mp4">
</video>




## Puzzle 9 - Pooling

Implement a kernel that sums together the last 3 position of `a` and stores it in `out`.
You have 1 thread per position. You only need 1 global read and 1 global write per thread.

*Tip: Remember to be careful about syncing.*


```python
def pool_spec(a):
    out = np.zeros(*a.shape)
    for i in range(a.shape[0]):
        out[i] = a[max(i - 2, 0) : i + 1].sum()
    return out


TPB = 8
def pool_test(cuda):
    def call(out, a, size) -> None:
        shared = cuda.shared.array(TPB, numba.float32)
        i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
        local_i = cuda.threadIdx.x
        # FILL ME IN (roughly 8 lines)
        if i < size:
            shared[i] = a[i]
            cuda.syncthreads()
        if i < 1:
            out[i] = shared[i]
        elif i < 2:
            out[i] = shared[i] + shared[i - 1]
        else:
            out[i] = shared[i - 2] + shared[i - 1] + shared[i]

    return call


SIZE = 8
out = np.zeros(SIZE)
a = np.arange(SIZE)
problem = CudaProblem(
    "Pooling",
    pool_test,
    [a],
    out,
    [SIZE],
    threadsperblock=Coord(TPB, 1),
    blockspergrid=Coord(1, 1),
    spec=pool_spec,
)
problem.show()
```

    # Pooling
     
       Score (Max Per Thread):
       |  Global Reads | Global Writes |  Shared Reads | Shared Writes |
       |             1 |             1 |             3 |             1 | 
    





    
![svg](GPU_puzzlers_files/GPU_puzzlers_42_1.svg)
    




```python
problem.check()
```

    Passed Tests!






<video alt="test" controls autoplay=1>
    <source src="https://openpuppies.com/mp4/sWp0Dqd.mp4"  type="video/mp4">
</video>




## Puzzle 10 - Dot Product

Implement a kernel that computes the dot-product of `a` and `b` and stores it in `out`.
You have 1 thread per position. You only need 2 global reads and 1 global write per thread.

*Note: For this problem you don't need to worry about number of shared reads. We will
 handle that challenge later.*


```python
def dot_spec(a, b):
    return a @ b

TPB = 8
def dot_test(cuda):
    def call(out, a, b, size) -> None:
        shared = cuda.shared.array(TPB, numba.float32)

        i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
        local_i = cuda.threadIdx.x
        # FILL ME IN (roughly 9 lines)
        if i < size:
            shared[i] = a[i] * b[i]
            cuda.syncthreads()
        threshold = size // 2
        while threshold > 0:
            j = i + threshold
            if i < threshold:
                shared[i] = shared[i] + shared[j]
            cuda.syncthreads()
            threshold = threshold // 2

        if i == 0:
            out[0] = shared[0]

    return call


SIZE = 8
out = np.zeros(1)
a = np.arange(SIZE)
b = np.arange(SIZE)
problem = CudaProblem(
    "Dot",
    dot_test,
    [a, b],
    out,
    [SIZE],
    threadsperblock=Coord(SIZE, 1),
    blockspergrid=Coord(1, 1),
    spec=dot_spec,
)
problem.show()
```

    # Dot
     
       Score (Max Per Thread):
       |  Global Reads | Global Writes |  Shared Reads | Shared Writes |
       |             2 |             1 |             7 |             4 | 
    





    
![svg](GPU_puzzlers_files/GPU_puzzlers_46_1.svg)
    




```python
problem.check()
```

    Passed Tests!






<video alt="test" controls autoplay=1>
    <source src="https://openpuppies.com/mp4/aydRUz8.mp4"  type="video/mp4">
</video>




## Puzzle 11 - 1D Convolution

Implement a kernel that computes a 1D convolution between `a` and `b` and stores it in `out`.
You need to handle the general case. You only need 2 global reads and 1 global write per thread.


```python
def conv_spec(a, b):
    out = np.zeros(*a.shape)
    len = b.shape[0]
    for i in range(a.shape[0]):
        out[i] = sum([a[i + j] * b[j] for j in range(len) if i + j < a.shape[0]])
    return out


MAX_CONV = 4
TPB = 8
TPB_MAX_CONV = TPB + MAX_CONV
SHARE_MEM_FULL_SIZE = 16
def conv_test(cuda):
    def call(out, a, b, a_size, b_size) -> None:
        i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
        local_i = cuda.threadIdx.x

        # FILL ME IN (roughly 17 lines)
        shared_a = cuda.shared.array(SHARE_MEM_FULL_SIZE, numba.float32)
        shared_b = cuda.shared.array(SHARE_MEM_FULL_SIZE, numba.float32)
        for idx in range(i, a_size, TPB):
            shared_a[idx] = a[idx]
        for idx in range(i, b_size, TPB):
            shared_b[idx] = b[idx]
        cuda.syncthreads()
        for idx in range(i, a_size, TPB):
            ans = 0
            for j in range(b_size):
                if idx + j < a_size:
                    ans += shared_a[idx + j] * shared_b[j]
            out[idx] = ans


    return call


# Test 1

SIZE = 6
CONV = 3
out = np.zeros(SIZE)
a = np.arange(SIZE)
b = np.arange(CONV)
problem = CudaProblem(
    "1D Conv (Simple)",
    conv_test,
    [a, b],
    out,
    [SIZE, CONV],
    Coord(1, 1),
    Coord(TPB, 1),
    spec=conv_spec,
)
problem.show()
```

    # 1D Conv (Simple)
     
       Score (Max Per Thread):
       |  Global Reads | Global Writes |  Shared Reads | Shared Writes |
       |             2 |             1 |             6 |             2 | 
    





    
![svg](GPU_puzzlers_files/GPU_puzzlers_49_1.svg)
    




```python
problem.check()
```

    Passed Tests!






<video alt="test" controls autoplay=1>
    <source src="https://openpuppies.com/mp4/g9I2ZmK.mp4"  type="video/mp4">
</video>




Test 2


```python
out = np.zeros(15)
a = np.arange(15)
b = np.arange(4)
problem = CudaProblem(
    "1D Conv (Full)",
    conv_test,
    [a, b],
    out,
    [15, 4],
    Coord(2, 1),
    Coord(TPB, 1),
    spec=conv_spec,
)
problem.show()
```

    # 1D Conv (Full)
     
       Score (Max Per Thread):
       |  Global Reads | Global Writes |  Shared Reads | Shared Writes |
       |             3 |             2 |            16 |             3 | 
    





    
![svg](GPU_puzzlers_files/GPU_puzzlers_52_1.svg)
    




```python
problem.check()
```

    Passed Tests!






<video alt="test" controls autoplay=1>
    <source src="https://openpuppies.com/mp4/bRKfspn.mp4"  type="video/mp4">
</video>




## Puzzle 12 - Prefix Sum

Implement a kernel that computes a sum over `a` and stores it in `out`.
If the size of `a` is greater than the block size, only store the sum of
each block.

We will do this using the [parallel prefix sum](https://en.wikipedia.org/wiki/Prefix_sum) algorithm in shared memory.
That is, each step of the algorithm should sum together half the remaining numbers.
Follow this diagram:

![](https://user-images.githubusercontent.com/35882/178757889-1c269623-93af-4a2e-a7e9-22cd55a42e38.png)


```python
TPB = 8
def sum_spec(a):
    out = np.zeros((a.shape[0] + TPB - 1) // TPB)
    for j, i in enumerate(range(0, a.shape[-1], TPB)):
        out[j] = a[i : i + TPB].sum()
    return out


def sum_test(cuda):
    def call(out, a, size: int) -> None:
        cache = cuda.shared.array(TPB, numba.float32)
        i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
        local_i = cuda.threadIdx.x
        # FILL ME IN (roughly 12 lines)
        if i < size:
            cache[local_i] = a[i]
        cuda.syncthreads()
        # stride = TPB // 2
        # while stride > 0:
        #     if local_i < stride and local_i + stride < size:
        #         cache[local_i] = cache[local_i] + cache[local_i + stride]
        #     cuda.syncthreads()
        #     stride = stride // 2
        stride = 1
        while stride < TPB:
            if local_i % (stride * 2) == 0 and local_i + stride < size:
                cache[local_i] = cache[local_i] + cache[local_i + stride]
            cuda.syncthreads()
            stride *= 2


        if local_i == 0:
            out[cuda.blockIdx.x] = cache[local_i]

    return call


# Test 1

SIZE = 6
out_num = (SIZE // TPB) + 1
out = np.zeros(out_num)
inp = np.arange(SIZE)
problem = CudaProblem(
    "Sum (Simple)",
    sum_test,
    [inp],
    out,
    [SIZE],
    Coord(out_num, 1),
    Coord(TPB, 1),
    spec=sum_spec,
)
problem.show()
```

    # Sum (Simple)
     
       Score (Max Per Thread):
       |  Global Reads | Global Writes |  Shared Reads | Shared Writes |
       |             1 |             1 |             7 |             4 | 
    





    
![svg](GPU_puzzlers_files/GPU_puzzlers_57_1.svg)
    




```python
problem.check()
```

    Passed Tests!






<video alt="test" controls autoplay=1>
    <source src="https://openpuppies.com/mp4/wHVpHVG.mp4"  type="video/mp4">
</video>




Test 2


```python
SIZE = 15
out = np.zeros(2)
inp = np.arange(SIZE)
problem = CudaProblem(
    "Sum (Full)",
    sum_test,
    [inp],
    out,
    [SIZE],
    Coord(2, 1),
    Coord(TPB, 1),
    spec=sum_spec,
)
problem.show()
```

    # Sum (Full)
     
       Score (Max Per Thread):
       |  Global Reads | Global Writes |  Shared Reads | Shared Writes |
       |             1 |             1 |             7 |             4 | 
    





    
![svg](GPU_puzzlers_files/GPU_puzzlers_60_1.svg)
    




```python
problem.check()
```

    Passed Tests!






<video alt="test" controls autoplay=1>
    <source src="https://openpuppies.com/mp4/pn1e9TO.mp4"  type="video/mp4">
</video>




## Puzzle 13 - Axis Sum

Implement a kernel that computes a sum over each column of `a` and stores it in `out`.


```python
TPB = 8
def sum_spec(a):
    out = np.zeros((a.shape[0], (a.shape[1] + TPB - 1) // TPB))
    for j, i in enumerate(range(0, a.shape[-1], TPB)):
        out[..., j] = a[..., i : i + TPB].sum(-1)
    return out


def axis_sum_test(cuda):
    def call(out, a, size: int) -> None:
        cache = cuda.shared.array(TPB, numba.float32)
        i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
        local_i = cuda.threadIdx.x
        batch = cuda.blockIdx.y
        # FILL ME IN (roughly 12 lines)
        if local_i < size:
            cache[local_i] = a[batch, local_i]
        cuda.syncthreads()
        # stride = TPB // 2
        # while stride > 0:
        #     if local_i < stride and local_i + stride < size:
        #         cache[local_i] = cache[local_i] + cache[local_i + stride]
        #     cuda.syncthreads()
        #     stride = stride // 2
        stride = 1
        while stride < TPB:
            if local_i % (stride * 2) == 0 and local_i + stride < size:
                cache[local_i] = cache[local_i] + cache[local_i + stride]
            cuda.syncthreads()
            stride *= 2

        if local_i == 0:
            out[batch, 0] = cache[0]

    return call


BATCH = 4
SIZE = 6
out = np.zeros((BATCH, 1))
inp = np.arange(BATCH * SIZE).reshape((BATCH, SIZE))
problem = CudaProblem(
    "Axis Sum",
    axis_sum_test,
    [inp],
    out,
    [SIZE],
    Coord(1, BATCH),
    Coord(TPB, 1),
    spec=sum_spec,
)
problem.show()
```

    # Axis Sum
     
       Score (Max Per Thread):
       |  Global Reads | Global Writes |  Shared Reads | Shared Writes |
       |             1 |             1 |             7 |             4 | 
    





    
![svg](GPU_puzzlers_files/GPU_puzzlers_63_1.svg)
    




```python
problem.check()
```

    Passed Tests!






<video alt="test" controls autoplay=1>
    <source src="https://openpuppies.com/mp4/pCAIlxD.mp4"  type="video/mp4">
</video>




## Puzzle 14 - Matrix Multiply!

Implement a kernel that multiplies square matrices `a` and `b` and
stores the result in `out`.

*Tip: The most efficient algorithm here will copy a block into
 shared memory before computing each of the individual row-column
 dot products. This is easy to do if the matrix fits in shared
 memory.  Do that case first. Then update your code to compute
 a partial dot-product and iteratively move the part you
 copied into shared memory.* You should be able to do the hard case
 in 6 global reads.


```python
def matmul_spec(a, b):
    return a @ b


TPB = 3
def mm_oneblock_test(cuda):
    def call(out, a, b, size: int) -> None:
        a_shared = cuda.shared.array((TPB, TPB), numba.float32)
        b_shared = cuda.shared.array((TPB, TPB), numba.float32)

        i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
        j = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y
        local_i = cuda.threadIdx.x
        local_j = cuda.threadIdx.y
        # FILL ME IN (roughly 14 lines)
        sum = 0
        for tile in range(0, size, TPB):
            if i < size and tile + local_j < size:
                a_shared[local_i, local_j] = a[i, tile + local_j]
            if j < size and tile + local_i < size:
                b_shared[local_i, local_j] = b[tile + local_i, j]
            cuda.syncthreads()

            for k in range(TPB):
                if tile + k < size:
                    sum += a_shared[local_i, k] * b_shared[k, local_j]

        if i < size and j < size:
            out[i, j] = sum

    return call

# Test 1
SIZE = 2
out = np.zeros((SIZE, SIZE))
inp1 = np.arange(SIZE * SIZE).reshape((SIZE, SIZE))
inp2 = np.arange(SIZE * SIZE).reshape((SIZE, SIZE)).T

problem = CudaProblem(
    "Matmul (Simple)",
    mm_oneblock_test,
    [inp1, inp2],
    out,
    [SIZE],
    Coord(1, 1),
    Coord(TPB, TPB),
    spec=matmul_spec,
)
problem.show(sparse=True)
```

    # Matmul (Simple)
     
       Score (Max Per Thread):
       |  Global Reads | Global Writes |  Shared Reads | Shared Writes |
       |             2 |             1 |             4 |             2 | 
    





    
![svg](GPU_puzzlers_files/GPU_puzzlers_66_1.svg)
    




```python
problem.check()
```

    Passed Tests!






<video alt="test" controls autoplay=1>
    <source src="https://openpuppies.com/mp4/eyxH0Wc.mp4"  type="video/mp4">
</video>




Test 2


```python
SIZE = 8
out = np.zeros((SIZE, SIZE))
inp1 = np.arange(SIZE * SIZE).reshape((SIZE, SIZE))
inp2 = np.arange(SIZE * SIZE).reshape((SIZE, SIZE)).T

problem = CudaProblem(
    "Matmul (Full)",
    mm_oneblock_test,
    [inp1, inp2],
    out,
    [SIZE],
    Coord(3, 3),
    Coord(TPB, TPB),
    spec=matmul_spec,
)
problem.show(sparse=True)
```

    # Matmul (Full)
     
       Score (Max Per Thread):
       |  Global Reads | Global Writes |  Shared Reads | Shared Writes |
       |             6 |             1 |            16 |             6 | 
    





    
![svg](GPU_puzzlers_files/GPU_puzzlers_69_1.svg)
    




```python
problem.check()
```

    Passed Tests!






<video alt="test" controls autoplay=1>
    <source src="https://openpuppies.com/mp4/Wj2PGRl.mp4"  type="video/mp4">
</video>



