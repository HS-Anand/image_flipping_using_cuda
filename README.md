
# CUDA Image Flipping — CPU vs GPU Performance

A parallel image-processing project using **NVIDIA CUDA and C++** to implement horizontal image flipping on both the CPU and GPU, then compare their execution times.

## Overview

This project implements horizontal image flipping using two approaches:

- **CPU implementation:** Processes image pixels sequentially using nested loops.
- **GPU implementation:** Uses a custom CUDA kernel to process multiple pixels in parallel across GPU threads.

The project measures CPU execution time and GPU end-to-end execution time, including memory transfers between host and device, to explore the performance characteristics of GPU-accelerated image processing.

## Features

- CPU-based horizontal image flipping.
- Custom CUDA kernel for parallel pixel processing.
- Two-dimensional CUDA thread indexing.
- Configurable thread block dimensions.
- GPU memory allocation and host-device memory transfers.
- CPU and GPU execution-time measurement.
- Performance speedup calculation.
- Output image generation using OpenCV.

## Tech Stack

- **Language:** C++
- **Parallel Computing:** NVIDIA CUDA C/C++
- **Image Processing:** OpenCV
- **Performance Measurement:** C++ `clock()` and CUDA Events

## How It Works

### 1. CPU Implementation

The CPU implementation iterates through each pixel and its color channels. Each pixel is written to its horizontally mirrored position.

The destination column is calculated as:

$$
x_{\text{dest}} = W - x - 1
$$

where `W` is the image width and `x` is the source column.

### 2. CUDA GPU Implementation

The GPU implementation distributes pixel-processing work across CUDA threads. Each thread calculates its corresponding pixel coordinates using its block and thread indices.

```cpp
int x = blockIdx.x * blockDim.x + threadIdx.x;
int y = blockIdx.y * blockDim.y + threadIdx.y;
```

Threads outside the image boundaries are excluded using bounds checks.

The kernel uses a **2D grid of 2D thread blocks**, with each block configured to contain 16 × 16 threads. Each valid thread reads a source pixel and writes it to its mirrored destination position.

### 3. Memory Management

The program allocates GPU memory for the input and output images using `cudaMalloc`.

The image data is transferred from host memory to device memory, processed by the CUDA kernel, and copied back to host memory for output.

### 4. Performance Measurement

The program measures:

- CPU processing time using `clock()`.
- GPU end-to-end time using CUDA Events, covering the intended transfer and kernel execution sequence.
- Speedup as the ratio of CPU processing time to GPU end-to-end time.

$$
\text{Speedup} =
\frac{\text{CPU Execution Time}}
{\text{GPU End-to-End Time}}
$$

A speedup above 1× indicates that the measured GPU path was faster for the tested image and system configuration.

## Project Structure

```text
image-flipping-using-cuda/
├── image_flipping_using_cuda.cu
├── input.jpg
├── flipped.jpg
└── README.md
```

`input.jpg` is the source image, and `flipped.jpg` is the generated horizontally flipped image.

## Requirements

- NVIDIA GPU with CUDA support.
- NVIDIA CUDA Toolkit and `nvcc`.
- C++ compiler compatible with the installed CUDA Toolkit.
- OpenCV development libraries.

## Build and Run

### 1. Verify CUDA availability

```bash
nvcc --version
nvidia-smi
```

### 2. Compile

On Linux, with OpenCV 4 available through `pkg-config`:

```bash
nvcc image_flipping_using_cuda.cu -o image_flip \
    $(pkg-config --cflags --libs opencv4)
```

### 3. Run

Place `input.jpg` in the working directory, then execute:

```bash
./image_flip
```

The program prints CPU time, GPU total time, and measured speedup. The output image is saved as `flipped.jpg`.

## Performance Analysis

The project compares sequential CPU processing against parallel GPU processing while accounting for the overhead of transferring image data.

Actual performance depends on image dimensions, GPU architecture, CPU performance, memory-transfer overhead, and CUDA launch overhead.

For small images, GPU overhead can outweigh the benefits of parallel execution. Larger images may provide more opportunities for parallel processing, although the actual speedup must be measured.

**Benchmark results:** To be added after testing on the target hardware.

## Future Improvements

- Benchmark multiple image resolutions and report measured speedups.
- Compare different CUDA block dimensions, such as 8 × 8, 16 × 16, and 32 × 32.
- Add CUDA allocation, memory-copy, and kernel-launch error handling.
- Validate GPU output against the CPU output for correctness.
- Measure kernel-only execution time separately from end-to-end GPU time.
- Evaluate pinned host memory and asynchronous transfers for larger images.
- Explore memory-access patterns and GPU profiling with NVIDIA Nsight Systems or Nsight Compute.

## Learning Outcomes

- Fundamentals of CUDA thread and block indexing.
- Parallel decomposition of an image-processing workload.
- GPU memory allocation and host-device data transfers.
- CUDA kernel execution and event-based timing.
- CPU-versus-GPU performance analysis and optimization trade-offs.

## License

This project is available for educational and portfolio purposes.
