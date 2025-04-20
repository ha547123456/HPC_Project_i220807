# HPC_Project_i220807 - Performance Evaluation of CUDA Optimizations for MNIST Neural Network

## Overview

This project presents a comparative analysis of four versions of a neural network model trained on the MNIST dataset, with each version incorporating different levels of optimization to enhance performance using CUDA and GPU. The goal of the project is to evaluate how different optimizations impact the speed and accuracy of training on the MNIST handwritten digits dataset.

## System Configuration

The experiments were conducted on a system equipped with an NVIDIA GPU, utilizing CUDA for parallel processing. The neural network architecture consists of:
- **Input layer**
- **Hidden layer**
- **Output layer**

ReLU activation functions were employed for non-linearity.

## Versions Evaluated

### Version 1: Baseline Sequential Implementation
- **Description**: The neural network was executed sequentially on the CPU, serving as the baseline for performance comparison.

### Version 2: Naive GPU Implementation
- **Description**: The neural network was ported to the GPU using CUDA with minimal optimizations. The implementation involved straightforward memory allocations and kernel launches without advanced considerations for memory hierarchy or execution configuration.

### Version 3: Optimized GPU Implementation
- **Optimizations Applied**:
  - **Launch Configuration**: Optimized thread block size for maximum occupancy and minimized idle threads, ensuring efficient utilization of GPU resources.
  - **Occupancy**: Used `cudaOccupancyMaxPotentialBlockSize` to determine the optimal block size for better resource utilization.
  - **Communication Optimizations**: Asynchronous memory transfers with `cudaMemcpyAsync` were implemented to overlap computation and communication. Pinned memory was used to speed up data transfers between host and device.
  - **Memory Optimizations**: Shared memory within CUDA blocks was utilized to store intermediate results, reducing global memory accesses. Constant memory was employed to store fixed data like weights and biases, improving read access times. Memory accesses were coalesced to maximize bandwidth utilization.

### Version 4: Tensor Core Utilization
- **Description**: Built upon Version 3, this version leveraged NVIDIA's Tensor Cores to accelerate matrix operations. Tensor Cores provide high-throughput matrix multiplications, significantly reducing training time and improving performance.

## Performance Comparison

The table below summarizes the performance metrics across different versions:

| **Version** | **Epoch 1 Time** | **Epoch 2 Time** | **Epoch 3 Time** | **Total Training Time** | **Test Accuracy** |
|-------------|------------------|------------------|------------------|-------------------------|-------------------|
| **V2**      | 29.184s          | 29.178s          | 29.684s          | 88.046s                 | 95.80%            |
| **V3**      | 24.008s          | 23.844s          | 24.080s          | 72.034s                 | 98.00%            |

## Discussion

- **CPU vs GPU**: The transition from CPU-based execution (Version 1) to GPU-based execution (Version 2) resulted in a significant reduction in training time, showcasing the advantages of GPU parallel processing capabilities.
- **Optimizations**: Version 3 introduced performance gains through improved launch configurations and memory management. Further, Version 4's utilization of Tensor Cores provided the most significant reduction in training time, highlighting the benefits of hardware-specific optimizations in deep learning tasks.

## Conclusion

This project demonstrates how various CUDA optimizations impact the performance of training a neural network on the MNIST dataset. By leveraging GPU capabilities, optimizing memory, and utilizing specialized hardware features like Tensor Cores, training time can be reduced, and accuracy improved. Future work may explore additional techniques such as mixed-precision training and distributed computing for even better performance.

## GitHub Repository

You can access the full project and source code at the following GitHub repository:  
[HPC_Project_i220807 GitHub Repository](https://github.com/ha547123456/HPC_Project_i220807.git)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
