#if KERNEL_CUDA
#define EIGEN_USE_GPU

#include "tensorflow/core/framework/op_kernel.h"
#include "integrand.h"

using namespace tensorflow;
using GPUDevice = Eigen::GpuDevice;

// This is the kernel that does the actual computation on device
template<typename T>
__global__ void IntegrandOpKernel(const T *input, T *output, const int nevents, const int ndim) {
    const auto gid = blockIdx.x*blockDim.x + threadIdx.x;
    // note: this an example of usage, not an example of a very optimal anything...
    for (int i = gid; i < nevents; i += blockDim.x*gridDim.x) {
        output[i] = 0.0;
        for (int j = 0; j < ndim; j++) {
            output[i] += input[i,j];
        }
    }
}

// But it still needs to be launched from within C++
// this bit is to be compared with the functor at the top of integrand.cpp 
template <typename T>
void IntegrandOpFunctor<GPUDevice, T>::operator()(const GPUDevice &d, const T *input, T *output, const int nevents, const int dims) {
    const int block_count = 1024;
    const int thread_per_block = 20;
    IntegrandOpKernel<T><<<block_count, thread_per_block, 0, d.stream()>>>(input, output, nevents, dims);
}

template struct IntegrandOpFunctor<GPUDevice, double>;


#endif
