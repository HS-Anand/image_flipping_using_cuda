#include <stdio.h>
#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>
#include <time.h>

// ---------------- CPU FUNCTION ----------------
void flipCPU(unsigned char *input, unsigned char *output,
             int width, int height, int channels) {

    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            for (int c = 0; c < channels; c++) {

                int in_idx  = (y * width + x) * channels + c;
                int out_idx = (y * width + (width - x - 1)) * channels + c;

                output[out_idx] = input[in_idx];
            }
        }
    }
}

// ---------------- CUDA KERNEL ----------------
__global__ void flipGPU(unsigned char *input, unsigned char *output,
                        int width, int height, int channels) {

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        for (int c = 0; c < channels; c++) {

            int in_idx  = (y * width + x) * channels + c;
            int out_idx = (y * width + (width - x - 1)) * channels + c;

            output[out_idx] = input[in_idx];
        }
    }
}

// ---------------- MAIN ----------------
int main() {

    // -------- LOAD IMAGE --------
    cv::Mat img = cv::imread("input.jpg");

    if (img.empty()) {
        printf("Error: Image not found\n");
        return -1;
    }

    int width = img.cols;
    int height = img.rows;
    int channels = img.channels();

    size_t size = width * height * channels * sizeof(unsigned char);

    cv::Mat cpu_output(height, width, img.type());
    cv::Mat gpu_output(height, width, img.type());

    // ---------------- CPU TIMING ----------------
    clock_t start_cpu = clock();

    flipCPU(img.data, cpu_output.data, width, height, channels);

    clock_t end_cpu = clock();

    double cpu_time = ((double)(end_cpu - start_cpu)) / CLOCKS_PER_SEC * 1000.0;
    printf("CPU Time: %f ms\n", cpu_time);

    // ---------------- GPU MEMORY ----------------
    unsigned char *d_input, *d_output;

    cudaMalloc((void**)&d_input, size);
    cudaMalloc((void**)&d_output, size);

    // Include transfer time (fair comparison)
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    cudaMemcpy(d_input, img.data, size, cudaMemcpyHostToDevice);

    dim3 block(16, 16);
    dim3 grid((width + 15) / 16, (height + 15) / 16);

    flipGPU<<<grid, block>>>(d_input, d_output, width, height, channels);

    cudaMemcpy(gpu_output.data, d_output, size, cudaMemcpyDeviceToHost);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float gpu_time = 0;
    cudaEventElapsedTime(&gpu_time, start, stop);

    printf("GPU Total Time (with transfer): %f ms\n", gpu_time);

    // ---------------- SPEEDUP ----------------
    printf("Speedup: %f x\n", cpu_time / gpu_time);

    // ---------------- SAVE FINAL IMAGE ----------------
    cv::imwrite("flipped.jpg", gpu_output);
    printf("Final flipped image saved as flipped.jpg\n");

    // ---------------- CLEANUP ----------------
    cudaFree(d_input);
    cudaFree(d_output);

    return 0;
}